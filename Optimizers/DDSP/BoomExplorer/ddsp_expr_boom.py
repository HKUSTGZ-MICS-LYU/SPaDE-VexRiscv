'''

ICCAD Contest Problem C Experiments on BOOM-Explorer (Baseline)

'''

from typing import List
import matplotlib.pyplot as plt
import torch
import numpy as np

from botorch.test_functions.multi_objective import ZDT1, BraninCurrin, DTLZ2, Penicillin, VehicleSafety
from botorch.acquisition.multi_objective.utils import (
    sample_optimal_points,
    random_search_optimizer,
    compute_sample_box_decomposition
)

from .utils import get_hv
from .solver import BOOMExplorerSolver
from iccad_contest.functions.problem import PPAT, DesignSpaceExplorationProblem
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class ProblemWrapped(DesignSpaceExplorationProblem):

    def __init__(self):
        super().__init__()
        self.dim = self.microarchitecture_embedding_set.shape[1]
        self.num_objectives = self.dim_of_objective_values
        self.bounds = self.bounds = torch.tensor(
            np.array([np.min(self.microarchitecture_embedding_set, axis=0),
            np.max(self.microarchitecture_embedding_set, axis=0)]), dtype=torch.float64)
        self._ref_point = torch.tensor(self.reference_point).to(device)
        
        self.pareto = self.pareto_frontier.clone().to(dtype=torch.float64, device=device)
        for i in [1, 2]:
            self._ref_point[i] = -self._ref_point[i]
            self.pareto[:, i] = -self.pareto[:, i]
        
        # self.pareto = self.pareto[:, 0:2]
        # self._ref_point = self._ref_point[0:2]

        self.max_hv = get_hv(self.pareto, self._ref_point)

    def eval(self, X):   
        objs = []
        for x in X:
            ppat = self.evaluate(x.cpu())
            objs.append([ppat.performance, -ppat.power, -ppat.area])
            # objs.append([ppat.performance, -ppat.power])
        return torch.tensor(objs)
    
problem = ProblemWrapped()

n = 31
bounds = problem.bounds.to(device)
dims = problem.dim
nobjs = problem.num_objectives


configs = {
    # DKL-GP settings
    "dkl-gp":{
        # the MLP output dimension specification
        "mlp-output-dim": 6,
        # the learning rate for DKL-GP
        "learning-rate": 0.001,
        # the maximal training epochs
        "max-traininig-epoch": 1000
    },
    'microal': True
}

def expr(seed):
    hv_curve = []
    optimizer = BOOMExplorerSolver(configs=configs, 
                            ref_point=problem._ref_point,
                            space=torch.tensor(problem.microarchitecture_embedding_set, dtype=torch.float64),
                            dims=problem.dim,
                            ninits=20,
                            seed=seed)
    for _ in range(n):
        suggest_x = optimizer.suggest()
        # print(optimizer.inter_seed)
        print(f"[EXPERIMENT] suggest {len(suggest_x)} point(s)...")
        Y = problem.eval(suggest_x)
        _, hv = optimizer.observe(suggest_x, Y)
        hv_curve.append(hv)
        # print(f"[EXPERIMENT] Reference Hypervolume:", problem.max_hv)
    # plt.show()
    return np.array(hv_curve)

def plot_hv(hv: np.ndarray, color, label):
    hv_mean = hv.mean(axis=0)
    hv_ci = 1.96 * hv.std(axis=0) / hv.shape[0]
    x = range(len(hv_mean))
    plt.plot(x, hv_mean, color = color, label = label)
    plt.fill_between(x, y1=hv_mean+hv_ci, y2=hv_mean-hv_ci, 
                     color = color, alpha=0.2)
    
# hvs = np.array([expr() for _ in range(0, 10)])
# plot_hv(hvs, color='r', label = 'BOOM')
# plt.savefig('results_boom.pdf')
# print(hv_curve)