import torch

def create_probability_constraint_free_weights(no_free_weights: int, dtype: torch.dtype, device: torch.device):
    """
    Generate constraint to guarantee weights are valid pdf, i.e. each in [0,1] and sum <=1
    We are using only free weights, last weight is fixed by the sum constraint
    """
    # generate inequality constrain, i.e. sum of free weights <= 1
    # inequality_constraints (List[Tuple[Tensor, Tensor, float]] | None) – A list of tuples (indices, coefficients, rhs), with each tuple encoding an inequality constraint of the form sum_i (X[indices[i]] * coefficients[i]) >= rhs. indices and coefficients should be torch tensors.
    # as fom sum(X) <= 1 follows -sum(X) >= -1, the constraint takes the form
    constraint = (torch.arange(no_free_weights, device=device), -1* torch.ones(no_free_weights, dtype=dtype, device=device), -1.0)
    return constraint


def get_bounds(no_weights):
    return torch.tensor([[0.0] * no_weights, [1.0] * no_weights])