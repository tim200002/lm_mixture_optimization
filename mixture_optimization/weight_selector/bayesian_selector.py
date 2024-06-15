import logging
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from ax.service.ax_client import AxClient, ObjectiveProperties
from typing import List

logger = logging.getLogger("experiment_runner")

class BayesianWeightSelector(WeightSelectorInterface):
    def __init__(self, config: dict):
        super().__init__(config)
        self.no_weights = config['no_weights']
        assert self.no_weights > 1, "Bayesian optimization requires at least 2 weights"
        self.no_free_weights = self.no_weights - 1

        # Define the search space for the weights
        parameters = []
        for i in range(self.no_free_weights):
            name = f"w{i}"
            parameters.append({
                "name": name,
                "type": "range",
                "bounds": [0, 1],
                "value_type": "float"
            })
        
        parameters_sum_str = " + ".join([param["name"] for param in parameters])
        constraint_upper = f"{parameters_sum_str} <= 1"
        parameter_constraints = [constraint_upper]
        objectives = {"perplexity": ObjectiveProperties(minimize=True)}

        self.client = AxClient()
        self.client.create_experiment(
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            objectives=objectives,
        )

    def parse_history(self, run_history: List[dict]):
        for i, run in enumerate(run_history):
            mixing_weights = run["true_mixing_weights"]
            free_mixing_weights = mixing_weights[:-1]
            parameters = {f"w{i}": weight for i, weight in enumerate(free_mixing_weights)}
            param, trial_index = self.client.attach_trial(parameters=parameters)
            assert trial_index == i, "Trial index mismatch" 
            logger.info(f"Attached trial {trial_index} with parameters {parameters}")
            if "weighted_val_perplexity" in run:
                perplexity = run["weighted_val_perplexity"]
                self.add_evaluation(perplexity, i)
    

    def propose_next_weights(self):
        parameters, trial_index = self.client.get_next_trial()
        free_parameters_list = [parameters[f"w{i}"] for i in range(self.no_free_weights)]
        logger.info(f"Proposing next weights for trial {trial_index} with free parameters {free_parameters_list}")
        return self._convert_free_weights_to_pdf(free_parameters_list)
    
    def add_evaluation(self, perplexity, run_index):
        logger.info(f"Adding evaluation for trial {run_index} with perplexity {perplexity}")
        self.client.complete_trial(trial_index=run_index, raw_data={"perplexity": perplexity})

    def get_best_weights(self):
        best_parameters, values = self.client.get_best_parameters()
        best_weights = [best_parameters[f"w{i}"] for i in range(self.no_free_weights)]
        self._convert_free_weights_to_pdf(best_weights)

    def _convert_free_weights_to_pdf(self, free_weights):
        fixed_weight = 1 - sum(free_weights)
        return [*free_weights, fixed_weight]