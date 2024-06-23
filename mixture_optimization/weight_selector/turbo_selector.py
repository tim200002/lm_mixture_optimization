import logging
from typing import List
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
import os
import math
import warnings
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

logger = logging.getLogger("experiment_runner")

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )



def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    device="cpu",
    dtype=torch.double,
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next



class BayesianWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict, run_history = None):
        super().__init__(config)
        self.no_weights = config['no_weights']
        assert self.no_weights > 1, "Bayesian optimization requires at least 2 weights"
        self.no_free_weights = self.no_weights - 1
        self.bathc_size = 1 # right now we dont support parallel processing
        self.state = TurboState(dim=self.no_free_weights, batch_size=self.bathc_size)
        
        # generate bounds for the weights
        self.bounds = torch.stack([torch.zeros(self.no_free_weights), torch.ones(self.no_free_weights)])
        
        # generate inequality constrain, i.e. sum of free weights <= 1
        # inequality_constraints (List[Tuple[Tensor, Tensor, float]] | None) – A list of tuples (indices, coefficients, rhs), with each tuple encoding an inequality constraint of the form sum_i (X[indices[i]] * coefficients[i]) >= rhs. indices and coefficients should be torch tensors.
        # as fom sum(X) <= 1 follows -sum(X) >= -1, the constraint takes the form
        constraint = (torch.arange(self.no_free_weights), -1* torch.ones(self.no_free_weights), -1)
        self.constraints = [constraint]

        # Tracking
        self.initialization_runs = [] # list of initialization runs, each dict with keys trial_idx, free_weights, value
        self.current_turbo_run = [] # list of dict with keys trial_idx, free_weights, value
        self.current_restart_idx = None
        self.best_result = None

        # Some turbo hyperparameters
        self.n_candiates = min(5000, max(2000, 200 * self.no_free_weights))
        self.num_restarts = 10
        self.raw_samples = 512
        self.acqf = "ts"

        # Initialization
        self.no_initialization_runs = config["no_initialization_runs"]
        self.desired_no_restarts = config["no_restarts"]
        # self.overwrite_start = config.get("overwrite_start", False), not yet supported for now

        # Reload initialization
        if "initialization_weights" in config:
            logger.info("Reloading initialization weights")
            self.initialization_weights = config["initialization_weights"]
            assert len(self.initialization_weights) == self.no_initialization_runs, "Wrong number of initialization weights"
        else:
            # Sobol initialization
            sobol = SobolEngine(self.no_free_weights, scramble=True)
            self.initialization_weights = sobol.draw(self.no_initialization_runs).tolist()
            config["initialization_weights"] = self.initialization_weights


        # Reload history
        if run_history is not None:
            self._parse_history(run_history)


    def _parse_history(self, run_history: List[dict]):
        history_initialization = []
        history_turbo = []
        for run in run_history:
            if run["type"] == "initialization":
                history_initialization.append(run)
            elif run["type"] == "turbo":
                history_turbo.append(run)
            else:
                raise ValueError("Unknown run type")
        
        self._parse_history_initialization(history_initialization)
        self._parse_history_turbo(history_turbo)
    
    def _parse_history_initialization(self, run_history: List[dict]):
        assert len(run_history) <= self.no_initialization_runs, "Too many initialization runs"

        for i, run in enumerate(run_history):
            trial_index = run["trial_index"]
            assert trial_index == i, "Trial index mismatch"
            mixing_weights = run["true_mixing_weights"]
            free_weights = mixing_weights[:-1]
            reference_weights = self.initialization_weights[i]
            assert torch.allclose(torch.tensor(free_weights), torch.tensor(reference_weights), atol=1e-3), "Weights mismatch"

            obj = {
                "trial_idx": trial_index,
                "free_weights": free_weights,
                "value": None
            }

            self.initialization_runs.append(obj)
            if "weighted_val_perplexity" in run:
                perplexity = run["weighted_val_perplexity"]
                self.add_evaluation(perplexity, trial_index)


    def _parse_history_turbo(self, run_history: List[dict]):
        # find latest restart index
        cur_restart_idx = max([run["restart_idx"] for run in run_history])
        self.current_restart_idx = cur_restart_idx
        self.state = TurboState(dim=self.no_free_weights, batch_size=self.bathc_size)
        cur_runs = [run for run in run_history if run["restart_idx"] == cur_restart_idx]
        for run in cur_runs:
            trial_index = run["trial_index"]
            logger.info(f"PArsing trial {trial_index}.")
            restart_idx = run["restart_idx"]
            assert restart_idx == cur_restart_idx, "Restart index mismatch"    
            mixing_weights = run["true_mixing_weights"]
            free_weights = mixing_weights[:-1]

            obj = {
                "trial_idx": trial_index,
                "free_weights": free_weights,
                "value": None
            }    
            # safte check
            for run in self.current_turbo_run:
                assert run["trial_idx"] != trial_index, "Trial index already exists"
            self.current_turbo_run.append(obj)
     
            if "weighted_val_perplexity" in run:
                perplexity = run["weighted_val_perplexity"]
                self.add_evaluation(perplexity, trial_index)
    
    def propose_next_weights(self):
        if self.completed_runs < self.no_initialization_runs:
            return self._propose_next_weights_initialization()
        else:
            return self._propose_next_weights_turbo()
    
    def _propose_next_weights_initialization(self):        
        trial_idx = self.completed_runs
        assert trial_idx < self.no_initialization_runs, "Too many initialization runs"
        free_weights = self.initialization_weights[trial_idx]
        weights = self._convert_free_weights_to_pdf(free_weights)

        obj = {
            "type": "initialization",
            "trial_idx": trial_idx,
        }

        return weights, obj


    def _propose_next_weights_turbo(self):
        # case 1 the first value is probably best found via sobol
        if self.current_turbo_run == []:
            
            current_turbo_idx = self.current_restart_idx
            # sort initialization runs
            sorted_initialization_runs = sorted(self.initialization_runs, key=lambda x: x["value"], reverse=True)
            initialization = sorted_initialization_runs[current_turbo_idx]
            best_value = initialization["value"]
            self.state = TurboState(dim=self.no_free_weights, batch_size=self.bathc_size, best_value=best_value)
            logger.info(f"Restarting turbo run. Current restart index {current_turbo_idx}. Starting from value {initialization['value']}, found for weights {initialization['free_weights']}")


        # Generate new sampling points    
        initialization_weights = [run["free_weights"] for run in self.initialization_runs]
        initialization_values = [run["value"] for run in self.initialization_runs]
        curr_turbo_run_weights = [run["free_weights"] for run in self.current_turbo_run]
        curr_turbo_run_values = [run["value"] for run in self.current_turbo_run]
        X = torch.tensor(initialization_weights + curr_turbo_run_weights, dtype=torch.float)
        Y = torch.tensor(initialization_values + curr_turbo_run_values, dtype=torch.float)
        print(f"X shape {X.shape}, Y shape {Y.shape}")
        # no need for normalizing unnormalizing as X is anyways on domain [0,1]
        
        train_y = (Y - Y.mean()) / Y.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=self.no_free_weights, lengthscale_constraint=Interval(0.005, 4.0)
                )
        )
        model = SingleTaskGP(
            X, train_y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Fit the model
        max_cholesky_size = float("inf")
        with gpytorch.gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)

            # Generate candidates
            next_batch = generate_batch(
                state=self.state,
                model=model,
                X=X,
                Y=train_y,
                batch_size=self.bathc_size,
                n_candidates=self.n_candiates,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                acqf=self.acqf,
            )

        # Convert weights
        assert len(next_batch) == 1, "Only batch size one supported so far"
        next_free_weights = next_batch[0].tolist()
        next_weights = self._convert_free_weights_to_pdf(next_free_weights)

        # get last run and check whether everything is fine
        next_trial_idx = self.completed_runs
        
        obj = {
            "type": "turbo",
            "trial_idx": next_trial_idx,
            "free_weights": next_free_weights,
            "value": None
        }
        self.current_turbo_run.append(obj)

        return next_weights, {"generation_strategy": "turbo", "trial_index": next_trial_idx, "restart_idx": self.current_restart_idx}


    def add_evaluation(self, perplexity, trial_index):
        logger.info(f"Adding evaluation for trial {trial_index} with perplexity {perplexity}")
        # check if trial index in initialization or turbo
        for run in self.initialization_runs:
            if run["trial_idx"] == trial_index:
                self._add_evaluation_initialization(perplexity, trial_index)
                return
        
        for run in self.current_turbo_run:
            if run["trial_idx"] == trial_index:
                self._add_evaluation_turbo(perplexity, trial_index)
                return
        
        raise ValueError("Trial index not found")

    def _add_evaluation_initialization(self, perplexity, trial_index):
        logger.info("Adding evaluation to latest initialization run")
        value_inv = - perplexity
        assert self.initialization_runs[-1]["trial_idx"] == trial_index, "Run index mismatch"
        self.initialization_runs[-1]["value"] = value_inv
        self.completed_runs += 1

    def _add_evaluation_turbo(self, perplexity, trial_index):
        logger.info("Adding evaluation to latest turbo run")
        value_inv = - perplexity
        self.state = update_state(self.state, value_inv) 
        assert self.current_turbo_run[-1]["trial_idx"] == trial_index, "Run index mismatch"
        self.current_turbo_run[-1]["value"] = value_inv
        # Experiment tracking
        if self.best_result is None or perplexity > self.best_result["value"]:
            self.best_result = {
                "trial_idx": trial_index,
                "free_weights": self.current_turbo_run[-1]["free_weights"],
                "value": perplexity
            }
        self.completed_runs += 1


        # Turbo 
        if self.state.restart_triggered:
            self.current_restart_idx += 1 # We must do this here to be able to return
            self.current_turbo_run = []


    def get_best_weights(self):
        if self.best_result is None:
            return None

        best_weights = self.best_result["free_weights"]
        return self._convert_free_weights_to_pdf(best_weights)

    def _convert_free_weights_to_pdf(self, free_weights):
        fixed_weight = 1 - sum(free_weights)
        return [*free_weights, fixed_weight]

    def experiment_done(self):
        assert self.completed_runs <= self.no_runs, "Too many runs completed"
        return self.completed_runs == self.no_runs