import sys
sys.path.append(".")

from mixture_optimization.weight_selector.weight_selector_interface import WeightSelectorInterface
import torch
from botorch.utils.sampling import sample_polytope

if __name__ == "__main__":
    # bounds = [(0.2, 0.7), (0,0.2), (0,0.1), (0.1, 0.4), (0.2, 0.5)]
    # no_weights = len(bounds)

    # num_test_samples = 10
    # weight_samples = WeightSelectorInterface._sample_uniform(num_test_samples, no_weights, bounds)
    # print(weight_samples)

    # A = torch.eye(3)
    # x = torch.tensor([0.1, 0.2, 0.7]).reshape(-1,1)
    # b = torch.ones(3,1)
    # print(f"A: {A}, x: {x}, b: {b}")

    # t = sample_polytope(A, b, x, 10, 10)
    # print(t)

    bounds = [(0.2, 0.7), (0,0.2), (0,0.1), (0.1, 0.4), (0.2, 0.5)]
    no_weights = len(bounds)

    weight_samples = WeightSelectorInterface._sample_uniform(10, no_weights, bounds)
    print(weight_samples)