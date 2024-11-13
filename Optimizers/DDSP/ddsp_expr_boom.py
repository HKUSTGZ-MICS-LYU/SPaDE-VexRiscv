'''

ICCAD Contest Problem C Experiments on DDSP

'''

from typing import List
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

from iccad_contest.functions.problem import PPAT, DesignSpaceExplorationProblem


import json

class ProblemWrapped(DesignSpaceExplorationProblem):

    def __init__(self):
        super().__init__()
        self.dim = self.microarchitecture_embedding_set.shape[1]
        self.num_objectives = self.dim_of_objective_values
        self.bounds = self.bounds = torch.tensor(
            np.array([np.min(self.microarchitecture_embedding_set, axis=0),
            np.max(self.microarchitecture_embedding_set, axis=0)]), dtype=torch.float64)
        self._ref_point = torch.tensor(self.reference_point)
        
        self.pareto = self.pareto_frontier.clone().to(dtype=torch.float64)
        for i in [1, 2]:
            self._ref_point[i] = -self._ref_point[i]
            self.pareto[:, i] = -self.pareto[:, i]

        # self.pareto = self.pareto[:, 0:2]
        # self._ref_point = self._ref_point[0:2]

        self.max_hv = get_hv(self.pareto, self._ref_point)

    def eval(self, X):   
        objs = []
        for x in X:
            ppat = self.evaluate(x)
            objs.append([ppat.performance, -ppat.power, -ppat.area])
            # objs.append([ppat.performance, -ppat.power])
        return torch.tensor(objs)
    
problem = ProblemWrapped()

n = 11 # Num of Suggestions

tkwargs = {"dtype": torch.double, "device": "cpu"}

bounds = problem.bounds.to(**tkwargs)
dims = problem.dim
nobjs = problem.num_objectives

def expr(method, weight_type="hv", cluster_type="dominant", micro_al=False, seed=0):

    optimizer = mcts(
        dims=dims,
        nobjs=nobjs,
        bounds=bounds,
        ninits=10,
        Cp=0.1,
        leaf_size=10,
        seed=seed,
        ref_point=problem._ref_point,
        method=method,
        space=torch.tensor(problem.microarchitecture_embedding_set),
        weight_method=weight_type,
        cluster_method=cluster_type,
        micro_al=micro_al
    )
    hv_curve = []
    for _ in range(n):
        suggest_x = optimizer.suggest()
        # print(optimizer.inter_seed)
        print(f"[EXPERIMENT] suggest {len(suggest_x)} point(s)...")
        Y = problem.eval(suggest_x)
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

num_of_exp = 5

# mcts_g_hv = np.array([expr("mcts-ehvi", weight_type="global_hv", seed=s) for s in range(0, num_of_exp)])
# plot_hv(mcts_g_hv, 'pink', 'mcts-ehvi-ghv')

# mcts_gc_hv = np.array([expr("mcts-ehvi", weight_type="global_hv", cluster_type="kmeans", seed=s) for s in range(0, num_of_exp)])
# plot_hv(mcts_gc_hv, 'yellow', 'mcts-ehvi-ghv-kmeans')

# mcts_gh_hv = np.array([expr("mcts-ehvi", weight_type="hv", cluster_type="hybrid", seed=s) for s in range(0, num_of_exp)])
# plot_hv(mcts_gh_hv, 'red', 'mcts-ehvi-hv-hybrid')

mcts_gh_hv = np.array([expr("mcts-ehvi", weight_type="global_hv", cluster_type="hybrid", seed=s, micro_al=True) for s in range(0, num_of_exp)])
plot_hv(mcts_gh_hv, 'brown', 'mcts-ehvi-ghv-hybrid-microal')

mcts_gh_hv = np.array([expr("mcts-ehvi", weight_type="global_hv", cluster_type="hybrid", seed=s) for s in range(0, num_of_exp)])
plot_hv(mcts_gh_hv, 'red', 'mcts-ehvi-ghv-hybrid')

mcts_hv = np.array([expr("mcts-ehvi", seed=s, micro_al=True) for s in range(0, num_of_exp)])
plot_hv(mcts_hv, 'pink', 'mcts-ehvi-hv-microal')

mcts_hv = np.array([expr("mcts-ehvi", seed=s) for s in range(0, num_of_exp)])
plot_hv(mcts_hv, 'purple', 'mcts-ehvi-hv')

ehvi_hv = np.array([expr("ehvi", seed=s, micro_al=True) for s in range(0, num_of_exp)])
plot_hv(ehvi_hv, 'cyan', 'ehvi-microal')

ehvi_hv = np.array([expr("ehvi", seed=s) for s in range(0, num_of_exp)])
plot_hv(ehvi_hv, 'blue', 'ehvi')

plt.legend()
plt.savefig('results.pdf')
# plt.show()
