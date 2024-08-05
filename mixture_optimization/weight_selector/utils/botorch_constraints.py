import torch

def create_probability_constraint_free_weights(no_free_weights: int, dtype, last_weight_constraint = None):
    """
    Generate constraint to guarantee weights are valid pdf, i.e. each in [0,1] and sum <=1
    We are using only free weights, last weight is fixed by the sum constraint
    """

    if last_weight_constraint is not None:
        lb , ub = last_weight_constraint[0], last_weight_constraint[1]
    else:
        lb, ub = 0.0, 1.0
    # generate inequality constrain, i.e. sum of free weights <= 1
    # inequality_constraints (List[Tuple[Tensor, Tensor, float]] | None) – A list of tuples (indices, coefficients, rhs), with each tuple encoding an inequality constraint of the form sum_i (X[indices[i]] * coefficients[i]) >= rhs. indices and coefficients should be torch tensors.
    # as fom sum(X) <= 1 follows -sum(X) >= -1, the constraint takes the form
    constraint1 = (torch.arange(no_free_weights), -1* torch.ones(no_free_weights, dtype=dtype), lb -1.0)
    constraint2 = (torch.arange(no_free_weights), torch.ones(no_free_weights, dtype=dtype), 1 - ub)

    return [constraint1, constraint2]


def get_unit_bounds(no_weights):
    return torch.tensor([[0.0] * no_weights, [1.0] * no_weights])


def get_bounds_from_config(bounds):
    lb = torch.tensor([0.0 if b is None else b[0] for b in bounds])
    ub = torch.tensor([1.0 if b is None else b[1] for b in bounds])

    return torch.stack([lb, ub], dim=0)