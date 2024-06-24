import logging
from mixture_optimization.datamodels.trial_tracking_config import Experiment
from mixture_optimization.datamodels.weight_selector_config import WeightSelectorConfig
from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from typing import List, Optional

logger = logging.getLogger("experiment_runner")

class BayesianWeightSelector(WeightSelectorInterface):
    def __init__(self, config: WeightSelectorConfig, experiment_history: Optional[List[Experiment]]= None):
        super().__init__(config)
        self.no_free_weights = self.no_weights - 1
        no_sobol, no_bayesian = config["no_sobol"], config["no_bayesian"]

        exp_counter = 0
        if run_history is not None:
            for run in run_history:
                generation_strategy = run["generation_strategy"]
                if generation_strategy == "sobol":
                    no_sobol = max(0, no_sobol - 1)
                elif generation_strategy == "bayesian":
                    no_bayesian  = max(0, no_bayesian - 1)
                else:
                    raise ValueError(f"Unknown generation strategy {generation_strategy}")
                
                # count finish experiments
                if run["status"] in ["parsed", "deleted"]:
                    exp_counter += 1

        generation_steps = []
        assert no_sobol + no_bayesian > 0, "No runs to be performed"

        if no_sobol > 0:
            generation_steps.append(GenerationStep(
                model= Models.SOBOL,
                num_trials= no_sobol
            ))
        if no_bayesian > 0:
            generation_steps.append(GenerationStep(
                model= Models.GPEI,
                num_trials= no_bayesian
            ))
        
        generation_strategy = GenerationStrategy(
            steps=generation_steps
        )

        # guarantee that of each stage enough runs are always completed

        self.no_runs = exp_counter + no_sobol + no_bayesian
        self.completed_runs = 0 # will late be increased when parsing-history
        self.client = AxClient(generation_strategy=generation_strategy)

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

        self.client.create_experiment(
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            objectives=objectives,
        )

        if run_history is not None:
            self._parse_history(run_history)

        logger.info(f"Setup new experiment. After parsing history, in addition to possibly resumed run, we have left sobol: {no_sobol} and bayesian: {no_bayesian} runs to perform.")
        logger.info(f"We already have {exp_counter} runs completed. In total there are {self.no_runs} runs to be performed.")

    def check_and_initialize_new_experiment(self):
        

    def _parse_history(self, run_history: List[dict]):
        for i, run in enumerate(run_history):
            mixing_weights = run["true_mixing_weights"]
            free_mixing_weights = mixing_weights[:-1]
            parameters = {f"w{i}": weight for i, weight in enumerate(free_mixing_weights)}
            param, trial_index = self.client.attach_trial(parameters=parameters)
            assert trial_index == i, "Trial index mismatch" 
            logger.info(f"PArsing trial {trial_index} with parameters {parameters} from history.")
            if "weighted_val_perplexity" in run:
                perplexity = run["weighted_val_perplexity"]
                self.add_evaluation(perplexity, i)
    
    def propose_next_weights(self):
        parameters, trial_index = self.client.get_next_trial()
        free_parameters_list = [parameters[f"w{i}"] for i in range(self.no_free_weights)]
        logger.info(f"Proposing next weights for trial {trial_index} with free parameters {free_parameters_list}")

        generation_strategy = "sobol" if trial_index < self.config["no_sobol"] else "bayesian"
        extra_keys = {"trial_index": trial_index, "generation_strategy": generation_strategy}
        return self._convert_free_weights_to_pdf(free_parameters_list), extra_keys
    
    def add_evaluation(self, perplexity, run_index):
        logger.info(f"Adding evaluation for trial {run_index} with perplexity {perplexity}")
        self.client.complete_trial(trial_index=run_index, raw_data={"perplexity": perplexity})
        self.completed_runs += 1

    def get_best_weights(self):
        best_parameters, values = self.client.get_best_parameters()
        best_weights = [best_parameters[f"w{i}"] for i in range(self.no_free_weights)]
        self._convert_free_weights_to_pdf(best_weights)

    def _convert_free_weights_to_pdf(self, free_weights):
        fixed_weight = 1 - sum(free_weights)
        return [*free_weights, fixed_weight]

    def experiment_done(self):
        assert self.completed_runs <= self.no_runs, "Too many runs completed"
        return self.completed_runs == self.no_runs