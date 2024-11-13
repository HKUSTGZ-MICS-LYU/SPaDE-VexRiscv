"""
    An example file
"""


import random
import torch
import sklearn
import numpy as np
import botorch
from iccad_contest.abstract_optimizer import AbstractOptimizer
from iccad_contest.design_space_exploration import experiment
from iccad_contest.functions.problem import get_pareto_frontier

from utils import *
from mcts.mcts import mcts

torch.manual_seed(0)
np.random.seed(0)

class DDSPOptimizer(AbstractOptimizer):
    primary_import = "iccad_contest"

    def __init__(self, design_space):
        """
            build a wrapper class for an optimizer.

            parameters
            ----------
            design_space: <class "MicroarchitectureDesignSpace">
        """
        AbstractOptimizer.__init__(self, design_space)
        self.x = []
        self.y = []
        self.n_suggest = 1
        self.microarchitecture_embedding_set = self.construct_microarchitecture_embedding_set()
        self.bounds = torch.tensor(
            np.array([np.min(self.microarchitecture_embedding_set, axis=0),
            np.max(self.microarchitecture_embedding_set, axis=0)]), dtype=torch.float64)
        self.lb = np.min(self.microarchitecture_embedding_set, axis=0)
        self.ub = np.max(self.microarchitecture_embedding_set, axis=0)
        self.dim = len(self.microarchitecture_embedding_set[0])

        self.model = mcts(
            dims= self.dim,
            bounds = self.bounds,
            nobjs=3,
            ninits=20,
            leaf_size=20,
            Cp=0.1,
            seed=42,
            weight_method="global-hv",
            cluster_method="hybrid",
            space = torch.tensor(self.microarchitecture_embedding_set, dtype=torch.float64),
            ref_point= torch.tensor([-1.0,-1.0,-1.0], dtype=torch.float64),
            method="ehvi"
        )


    def construct_microarchitecture_embedding_set(self):
        microarchitecture_embedding_set = []
        for i in range(1, self.design_space.size + 1):
            microarchitecture_embedding_set.append(
                self.design_space.vec_to_microarchitecture_embedding(
                    self.design_space.idx_to_vec(i)
                )
            )
        return np.array(microarchitecture_embedding_set)

    def suggest(self):
        """
            get a suggestion from the optimizer.

            returns
            -------
            next_guess: <list> of <list>
                list of `self.n_suggestions` suggestion(s).
                each suggestion is a microarchitecture embedding.
        """
        return self.model.suggest().tolist()
        
    
    def observe(self, x, y):
        """
            send an observation of a suggestion back to the optimizer.

            parameters
            ----------
            x: <list> of <list>
                the output of `suggest`.
            y: <list> of <list>
                corresponding values where each `x` is mapped to.
        """

        for _x in x:
            idx = np.argwhere(
                np.all(self.microarchitecture_embedding_set == _x, axis=1))
            self.microarchitecture_embedding_set = np.delete(
                self.microarchitecture_embedding_set, idx, axis=0)
        
        observe_y = []

        for _y in y:
            _y[1] = -_y[1]
            _y[2] = -_y[2]
            observe_y.append(_y)

        self.model.observe(torch.tensor(x, dtype=torch.float64), 
                           torch.tensor(observe_y))

if __name__ == "__main__":
    experiment(DDSPOptimizer)
