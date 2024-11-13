import torch
import numpy as np
from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def get_hv(objs : torch.Tensor, ref_point : torch.Tensor):
    pareto_Y = objs[is_non_dominated(objs)]
    bd = FastNondominatedPartitioning(ref_point=ref_point.to(device), 
                                      Y=pareto_Y.to(device))
    
    volume = bd.compute_hypervolume().item()
    return volume


# Latin Hypercube implementation from LA-MCTS

def from_unit_cube(point, lb, ub):
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point

def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points

# 2023 by Jzjerry
def nearest(x, lb, ub, dataset):
    assert x.shape[0] == dataset.shape[1]
    assert x.shape[0] == ub.shape[0]
    
    dist = np.sum(
        np.power(np.abs(x - dataset)/(ub-lb), 2), axis=1)
    return np.argmin(dist)