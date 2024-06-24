import logging
from typing import List, Tuple
from mixture_optimization.datamodels.trial_tracking_config import Experiment, ExperimentConfig, Trial, TrialType
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.utils.botorch_constraints import create_probability_constraint_free_weights
from mixture_optimization.weight_selector.weight_selector_interface import TrialMemoryUnit, WeightSelectorInterface
import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import sample_simplex

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

    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        samples = sample_simplex(config.no_weights, config.no_initializations, qmc=True).tolist()
        exp_config = ExperimentConfig(initialization_weights=samples, experiment_idx=exp_idx)
        return TurboWeightSelector(config, exp_config), exp_config
    
    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return TurboWeightSelector(config, experiment.config)

    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        assert config.no_optimizations == None, "Turbo selector uses custom logic for ending optimization"

        self.batch_size = 1 # right now we dont support parallel processing
        self.state = None
        self.no_free_weights = self.config.no_weights - 1
 
        # Some turbo hyperparameters
        self.num_restarts = 10
        self.raw_samples = 512
        self.dtype = torch.double
          

        # Tracking
        self.initialization_runs = [] # list of initialization runs, each dict with keys trial_idx, free_weights, value
        self.current_turbo_run = [] # list of dict with keys trial_idx, free_weights, value
        self.best_result = None
    

    def _propose_next_weights_optimization(self):
        # preliminary checks
        assert all([run.value is not None for run in self.trial_memory]), "All runs must be evaluated before proposing next weights"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Proposing next turbo weights. Using device {device}")
        # Generate new sampling points    
        X = torch.tensor([run.weights[:-1] for run in self.trial_memory], dtype=self.dtype, device=device) #! Only use free weights
        Y = torch.tensor([run.value for run in self.trial_memory], dtype=self.dtype, device=device).unsqueeze(-1)

        pdf_constraint = create_probability_constraint_free_weights(self.no_free_weights, self.dtype, device)
        constraints = [pdf_constraint]
        
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

        unit = TrialMemoryUnit(
            trial_index=self.get_next_trial_idx(),
            trial_type=TrialType.OPTIMIZATION,
            weights=next_weights
        )
        self.trial_memory.append(unit)

        return next_weights, TrialType.OPTIMIZATION


    def add_evaluation(self, perplexity, trial_index):
        super().add_evaluation(perplexity, trial_index)
        
        trial = self.trial_memory[-1]
        if trial.trial_type == TrialType.OPTIMIZATION:
            if self.state == None:
                self.state = self._init_state()
            
            value_tensor = torch.tensor([trial.value], dtype=self.dtype)
            self.state = update_state(self.state, value_tensor)

    def _init_state(self):
        initialization_trials_value = []
        for trial in self.trial_memory:
            if trial.trial_type == TrialType.INITIALIZATION:
                initialization_trials_value.append(trial.value)
            elif trial.trial_type == TrialType.OPTIMIZATION and trial.value is not None:
                raise ValueError("Turbo initialization must be completed before first optimization trial is completed")
            else:
                raise ValueError("Unknown trial type")
        
        best_value = max(initialization_trials_value)
        logger.info(f"Initializing turbo state from best value {best_value}")
        return TurboState(dim=self.no_free_weights, batch_size=self.batch_size, best_value=best_value)



    def experiment_done(self):
        if self.state is None:
            return False
        return self.state.restart_triggered