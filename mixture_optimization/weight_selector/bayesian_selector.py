import logging
from typing import Tuple

import torch
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls import ExactMarginalLogLikelihood

from mixture_optimization.datamodels.trial_tracking_config import (
    Experiment,
    ExperimentConfig,
    TrialType,
)
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.utils.botorch_constraints import (
    create_probability_constraint_free_weights,
    get_unit_bounds,
)
from mixture_optimization.weight_selector.weight_selector_interface import (
    TrialMemoryUnit,
    WeightSelectorInterface,
)

logger = logging.getLogger("experiment_runner")

class BayesianWeightSelector(WeightSelectorInterface):

    @staticmethod
    def from_scratch(config: WeightSelectorConfig, exp_idx: int) -> Tuple[WeightSelectorInterface, ExperimentConfig]:
        samples = sample_simplex(config.no_weights, config.no_initializations, qmc=True).tolist()
        exp_config = ExperimentConfig(initialization_weights=samples, experiment_idx=exp_idx)
        return BayesianWeightSelector(config, exp_config), exp_config

    @staticmethod
    def from_history(config: WeightSelectorConfig, experiment: Experiment) -> WeightSelectorInterface:
        return BayesianWeightSelector(config, experiment.config)

    def __init__(self, config: WeightSelectorConfig, experiment_config: ExperimentConfig):
        super().__init__(config, experiment_config)
        
        self.batch_size = 1 # Right now no support for parallelization
        self.no_free_weights = config.no_weights - 1
        self.dtype = torch.double

        self.num_restarts = 20
        self.raw_samples = 100

    
    def _propose_next_weights_optimization(self):
        # preliminary checks
        assert all([run.value is not None for run in self.trial_memory]), "All runs must be evaluated before proposing next weights"
        
        device = "cpu"
        logger.info(f"Proposing next weights. Using device {device}")
        
        X = torch.tensor([run.weights[:-1] for run in self.trial_memory], dtype=torch.double, device=device) #! Only use free weights
        Y = torch.tensor([run.value for run in self.trial_memory], dtype=torch.double, device=device).unsqueeze(-1)

        pdf_constraint = create_probability_constraint_free_weights(self.no_free_weights, self.dtype)
        constraints = [pdf_constraint]

        # normalize Y
        Y = (Y - Y.mean()) / Y.std()
        X = self._normalize(X)

        model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acqf = qNoisyExpectedImprovement(model, X)

        next_weight, _ = optimize_acqf(
            acq_function=acqf,
            bounds=get_unit_bounds(self.no_free_weights, dtype=self.dtype),
            inequality_constraints=constraints,
            q=self.batch_size,
            num_restarts = self.num_restarts,
            raw_samples = self.raw_samples,
        )
        
        next_weight = self._unnormalize(next_weight)
        weights = self._convert_free_weights_to_pdf(next_weight.squeeze().tolist())

        unit = TrialMemoryUnit(
            trial_index=self.get_next_trial_idx(),
            trial_type=TrialType.OPTIMIZATION,
            weights=weights
        )
        self.trial_memory.append(unit)

        return weights, TrialType.OPTIMIZATION