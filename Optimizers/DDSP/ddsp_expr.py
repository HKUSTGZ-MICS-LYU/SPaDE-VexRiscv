'''

General MOO Problem Test on DDSP

'''

import matplotlib.pyplot as plt
import torch

from botorch.test_functions.multi_objective import ZDT1, BraninCurrin, DTLZ2, Penicillin, VehicleSafety
from botorch.acquisition.multi_objective.utils import (
    sample_optimal_points,
    random_search_optimizer,
    compute_sample_box_decomposition
)

from utils import *
from mcts.mcts import mcts
from mcts.utils import *

import json

d = 10
M = 2
n = 61

tkwargs = {"dtype": torch.double, "device": "cpu"}
# problem = ZDT1(dim=d, num_objectives=M, noise_std=0, negate=True)
problem = BraninCurrin(negate=True) # d=2, M=2
# problem = DTLZ2(dim=d, num_objectives=M, negate=True)
# problem = Penicillin(negate=True)
# problem = VehicleSafety(negate=True)

bounds = problem.bounds.to(**tkwargs)
dims = problem.dim
nobjs = problem.num_objectives

def expr(method, weight_type="hv", cluster_type="dominant", seed=0):

    optimizer = mcts(
        dims=dims,
        nobjs=nobjs,
        bounds=bounds,
        ninits=40,
        Cp=0.2,
        leaf_size=20,
        seed=seed,
        ref_point= -torch.tensor(problem._ref_point, dtype=torch.float64),
        method=method,
        weight_method=weight_type,
        cluster_method=cluster_type
    )
    hv_curve = []
    for _ in range(n):
        suggest_x = optimizer.suggest()
        # print(optimizer.inter_seed)
        print(f"[EXPERIMENT] suggest {len(suggest_x)} point(s)...")
        Y = problem(suggest_x)
        _, hv = optimizer.observe(suggest_x, Y)
        hv_curve.append(problem.max_hv - hv)
        # print(f"[EXPERIMENT] Reference Hypervolume:", problem.max_hv)
    # optimizer.vis_nodes()
    return np.array(hv_curve)

def plot_hv(hv: np.ndarray, color, label):
    hv_mean = hv.mean(axis=0)
    hv_ci = 1.96 * hv.std(axis=0) / hv.shape[0]
    x = range(len(hv_mean))
    plt.plot(x, hv_mean, color = color, label = label)
    plt.fill_between(x, y1=hv_mean+hv_ci, y2=hv_mean-hv_ci, 
                     color = color, alpha=0.2)

num_of_exp = 10

mcts_g_hv = np.array([expr("mcts-ehvi", weight_type="global_hv", seed=s) for s in range(0, num_of_exp)])
plot_hv(mcts_g_hv, 'red', 'mcts-ehvi-ghv')

mcts_gc_hv = np.array([expr("mcts-ehvi", weight_type="global_hv", cluster_type="kmeans", seed=s) for s in range(0, num_of_exp)])
plot_hv(mcts_gc_hv, 'yellow', 'mcts-ehvi-ghv-kmeans')

mcts_hv = np.array([expr("mcts-ehvi", seed=s) for s in range(0, num_of_exp)])
plot_hv(mcts_hv, 'green', 'mcts-ehvi-hv')

ehvi_hv = np.array([expr("ehvi", seed=s) for s in range(0, num_of_exp)])
plot_hv(ehvi_hv, 'blue', 'ehvi')

plt.legend()
plt.savefig('results.pdf')
# plt.show()
