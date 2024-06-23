import logging
from typing import List
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
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
    success_tolerance: int = 3  # Note: The original paper uses 3
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

def generate_sample(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    num_restarts=10,
    raw_samples=512,
   inequality_constraints=None,

):
    batch_size = 1 # only batch size 1 currentuly supported
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # sharpen bounds to be within [0, 1]
    tr_lb = torch.max(tr_lb, torch.zeros_like(tr_lb))
    tr_ub = torch.min(tr_ub, torch.ones_like(tr_ub))

    ei = qExpectedImprovement(model, Y.max())
    X_next, acq_value = optimize_acqf(
        ei,
        bounds=torch.stack([tr_lb, tr_ub]),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
    )

    return X_next



class TurboWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict, run_history = None):
        super().__init__(config)
        self.no_weights = config['no_weights']
        assert self.no_weights > 1, "Bayesian optimization requires at least 2 weights"
        self.no_free_weights = self.no_weights - 1
        self.bathc_size = 1 # right now we dont support parallel processing
        self.state = None
        self.completed_runs = 0

        # Some turbo hyperparameters
        self.num_restarts = 10
        self.raw_samples = 512
        self.dtype = torch.double
        
        
        
       

        # Tracking
        self.initialization_runs = [] # list of initialization runs, each dict with keys trial_idx, free_weights, value
        self.current_turbo_run = [] # list of dict with keys trial_idx, free_weights, value
        self.best_result = None



        # Initialization
        self.no_initialization_runs = config["no_initialization_runs"]
        self.desired_no_restarts = config["no_restarts"]
        assert self.desired_no_restarts == 1, "Only one restart supported for now"

        # Reload initialization
        if "initialization_weights" in config:
            logger.info("Reloading initialization weights")
            self.initialization_weights = config["initialization_weights"]
            assert len(self.initialization_weights) == self.no_initialization_runs, "Wrong number of initialization weights"
        else:
            # Sobol initialization
            sobol = SobolEngine(self.no_weights, scramble=True)
            drawn_weights = sobol.draw(self.no_initialization_runs)
            row_sum = drawn_weights.sum(dim=1, keepdim=True)
            drawn_weights_normalized = drawn_weights / row_sum
            self.initialization_weights = drawn_weights_normalized[:,:-1].tolist()
            config["initialization_weights"] = self.initialization_weights

        # Reload history
        if run_history is not None:
            self._parse_history(run_history)


    def _parse_history(self, run_history: List[dict]):
        history_initialization = []
        history_turbo = []
        for run in run_history:
            if run["generation_strategy"] == "initialization":
                history_initialization.append(run)
            elif run["generation_strategy"] == "turbo":
                history_turbo.append(run)
            else:
                raise ValueError("Unknown generation_strategy")
        
        if len(history_initialization) > 0:
            self._parse_history_initialization(history_initialization)
        if len(history_turbo) > 0:
            self._parse_history_turbo(history_turbo)
    
    def _parse_history_initialization(self, run_history: List[dict]):
        assert len(run_history) <= self.no_initialization_runs, "Too many initialization runs"

        for i, run in enumerate(run_history):
            trial_index = run["idx"]
            assert trial_index == i, "Trial index mismatch"
            mixing_weights = run["true_mixing_weights"]
            free_weights = mixing_weights[:-1]
            reference_weights = self.initialization_weights[i]
            #assert torch.allclose(torch.tensor(free_weights), torch.tensor(reference_weights), atol=1e-3), "Weights mismatch"

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
        self.state = self._init_state()
        for run in run_history:
            trial_index = run["idx"]
            mixing_weights = run["true_mixing_weights"]
            free_weights = mixing_weights[:-1]

            obj = {
                "trial_idx": trial_index,
                "free_weights": free_weights,
                "value": None
            }    
            self.current_turbo_run.append(obj)
     
            if "weighted_val_perplexity" in run:
                perplexity = run["weighted_val_perplexity"]
                self.add_evaluation(perplexity, trial_index)

    def _init_state(self):
        best_value = max([run["value"] for run in self.initialization_runs])
        logger.info(f"Initializing turbo state from best value {best_value}")
        return TurboState(dim=self.no_free_weights, batch_size=self.bathc_size, best_value=best_value)
    
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

        return weights, {"generation_strategy": "initialization"}


    def _propose_next_weights_turbo(self):
        # case 1 the first value is probably best found via sobol
        if self.current_turbo_run == []:
            self.state = self._init_state()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Proposing next turbo weights. Using device {device}")
        # Generate new sampling points    
        initialization_weights = torch.tensor([run["free_weights"] for run in self.initialization_runs], dtype=self.dtype, device=device)
        initialization_values = torch.tensor([run["value"] for run in self.initialization_runs], dtype=self.dtype, device=device).unsqueeze(-1)
        curr_turbo_run_weights = torch.tensor([run["free_weights"] for run in self.current_turbo_run], dtype=self.dtype, device=device)
        curr_turbo_run_values = torch.tensor([run["value"] for run in self.current_turbo_run], dtype=self.dtype, device=device).unsqueeze(-1)
        X = torch.concatenate((initialization_weights, curr_turbo_run_weights), dim=0)
        Y = torch.concatenate((initialization_values, curr_turbo_run_values), dim=0)

         # generate inequality constrain, i.e. sum of free weights <= 1
        # inequality_constraints (List[Tuple[Tensor, Tensor, float]] | None) – A list of tuples (indices, coefficients, rhs), with each tuple encoding an inequality constraint of the form sum_i (X[indices[i]] * coefficients[i]) >= rhs. indices and coefficients should be torch tensors.
        # as fom sum(X) <= 1 follows -sum(X) >= -1, the constraint takes the form
        constraint = (torch.arange(self.no_free_weights, device=device), -1* torch.ones(self.no_free_weights, dtype=self.dtype, device=device), -1.0)
        constraints = [constraint]
        
        train_y = (Y - Y.mean()) / Y.std()
        likelihood = GaussianLikelihood()
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
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)

            # Generate candidates
            next_sample = generate_sample(
                state=self.state,
                model=model,
                X=X,
                Y=train_y,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
                inequality_constraints=constraints,
            )

        # Convert weights
        
        next_free_weights = next_sample.squeeze().tolist()
        next_weights = self._convert_free_weights_to_pdf(next_free_weights)

        # get last run and check whether everything is fine
        next_trial_idx = self.completed_runs
        
        obj = {
            "generation_strategy": "turbo",
            "trial_idx": next_trial_idx,
            "free_weights": next_free_weights,
            "value": None
        }
        self.current_turbo_run.append(obj)

        return next_weights, {"generation_strategy": "turbo"}


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


    def get_best_weights(self):
        if self.best_result is None:
            return None

        best_weights = self.best_result["free_weights"]
        return self._convert_free_weights_to_pdf(best_weights)

    def _convert_free_weights_to_pdf(self, free_weights):
        fixed_weight = 1 - sum(free_weights)
        return [*free_weights, fixed_weight]

    def experiment_done(self):
        if self.state is None:
            return False
        return self.state.restart_triggered