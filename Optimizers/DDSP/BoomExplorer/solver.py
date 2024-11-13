# Author: baichen318@gmail.com


import os
import tqdm
import torch
import gpytorch
import numpy as np
from typing import NoReturn
from .MicroAL import MicroAL
from .sample_utils import *
import random
# from dataset import rescale_dataset
from .model import initialize_dkl_gp
# from visualize import plot_pareto_set
# from problem import DesignSpaceProblem
# from metric import calc_adrs, get_pareto_frontier, get_pareto_optimal_solutions
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning



class BOOMExplorerSolver(object):
    def __init__(self, configs: dict, ref_point, space, dims, ninits, seed):

        super(BOOMExplorerSolver, self).__init__()
        self.ref_point = ref_point
        self.configs = configs
        self.space = space
        self.dims = dims
        self.ninits = ninits
        self.seed = seed

        self.X = torch.tensor([])
        self.Y = torch.tensor([])
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.init = True

    def set_optimizer(self) -> torch.optim.Adam:
        parameters = [
            {"params": self.model.mlp.parameters()},
            {"params": self.model.gp.covar_module.parameters()},
            {"params": self.model.gp.mean_module.parameters()},
            {"params": self.model.gp.likelihood.parameters()}
        ]
        return torch.optim.Adam(
            parameters, lr=self.configs["dkl-gp"]["learning-rate"]
        )

    def fit_dkl_gp(self) -> NoReturn:
        self.model = initialize_dkl_gp(
            self.X,
            self.Y,
            self.configs["dkl-gp"]["mlp-output-dim"]
        )
        self.model.set_train()
        optimizer = self.set_optimizer()

        # iterator = tqdm.trange(
        #     self.configs["dkl-gp"]["max-traininig-epoch"],
        #     desc="Training DKL-GP"
        # )
        iterator = range(self.configs["dkl-gp"]["max-traininig-epoch"])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.gp.likelihood,
            self.model.gp
        )
        y = self.model.transform_ylayout(self.Y).squeeze(1)
        for i in iterator:
            optimizer.zero_grad()
            _y = self.model.train(self.X)
            loss = -mll(_y, y)
            loss.backward()
            optimizer.step()
            # iterator.set_postfix(loss=loss.item())
        self.model.set_eval()

    def eipv_suggest(self, batch: int = 1):
        partitioning = NondominatedPartitioning(
            ref_point=self.ref_point.to(self.model.device),
            Y=self.Y.to(self.model.device)
        )

        acq_func = ExpectedHypervolumeImprovement(
            model=self.model.gp,
            ref_point=self.ref_point.tolist(),
            partitioning=partitioning
        ).to(self.model.device)

        post = self.model.forward_mlp(
            self.space.to(torch.float64).to(self.model.device))
        acq_val = acq_func(post.unsqueeze(1).to(self.model.device)).to(self.model.device)
        top_acq_val, indices = torch.topk(acq_val, k=batch)
        new_x = self.space[indices].to(torch.float64).reshape(-1, self.dims)

        return new_x

    def suggest(self):
        if self.init:
            initial_configs = {
                "Nrted": 59,
                "mu": 0.1,
                "sig": 0.1,
                # the total samples in a cluster
                "batch": self.ninits,
                "decoder-threshold": 35,
                # number for clusters
                "cluster": 5,
                # the iterations of the clustering
                "clustering-iteration": 1000,
                "vis-micro-al": False
            }
            if self.configs['microal']:
                print("[MicroAL] Start Clustering...")
                initializer = MicroAL(configs=initial_configs, n_dim=self.dims)
                suggest_x = torch.tensor(initializer.initialize(self.space.numpy()), 
                                         dtype = torch.float64)
            else:
                # suggest_x = rand_init_sample(self.ninits, self.space)
                suggest_x = lhs_init_sample(self.ninits, self.space)
            self.init = False
            return suggest_x
        else:
            return self.eipv_suggest()

    def observe(self, X, Y):

        self.X = torch.cat((self.X, X), 0)
        self.Y = torch.cat((self.Y, Y), 0)
        self.fit_dkl_gp()
        if self.space is not None:
            mask = ~torch.any(torch.stack([torch.all(self.space == x, dim=1) for x in X]), 
                              dim=0)
            self.space = self.space[mask]
            print("Unvisited Space:", self.space.shape[0])

        p_Y = self.Y
        # tmp = torch.concat((p_X, p_Y), dim=1)

        pareto_Y = p_Y[is_non_dominated(p_Y)]
        # pareto_mean = torch.mean(pareto_Y, dim=0)
            
        volume = get_hv(pareto_Y, self.ref_point)
        return pareto_Y, volume
