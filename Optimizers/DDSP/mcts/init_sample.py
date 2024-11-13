
import torch
import numpy as np
import random

from .utils import *


def rand_init_sample(n : int, space: torch.Tensor):

    idx = random.sample(range(0, space.shape[0]), k=n)

    return space[idx]

def lhs_init_sample(n : int, space: torch.Tensor):
    space = torch.tensor(space)
    dataset = space.cpu().numpy()

    lb = space.min(dim=0).values.cpu().numpy()
    ub = space.max(dim=0).values.cpu().numpy()
    cube = latin_hypercube(n, space.shape[1])
    samples = from_unit_cube(cube, lb, ub)
    
    return_samples = []
    for sample in samples:
        idx = nearest(sample, lb, ub, dataset)
        return_samples.append(dataset[idx].tolist())
    return torch.tensor(return_samples)