'''
Author: Jzjerry

A Monte Carlo Tree Search Class

'''

import torch
import traceback

from .node import node
from .utils import *

from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.pareto import is_non_dominated

from botorch import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list


# MC EHVI Aquisition
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.analytic import (
    ExpectedHypervolumeImprovement
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import(
    qHypervolumeKnowledgeGradient
)
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch
)
from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)

from botorch.acquisition.multi_objective.utils import (
    sample_optimal_points,
    random_search_optimizer,
    compute_sample_box_decomposition
)

from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

# Gaussian Process Regression
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.kernels.linear_kernel import LinearKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.hamming_kernel import HammingIMQKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.means.linear_mean import LinearMean
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import SingleTaskGP, MixedSingleTaskGP, GenericDeterministicModel
from botorch.models.kernels import CategoricalKernel

# Scikit Learn
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Numpy
import numpy as np
import random

# Visualize
import matplotlib.pyplot as plt

from .init_sample import rand_init_sample, lhs_init_sample

class mcts:
    def __init__(self, 
                 dims : int, 
                 bounds : torch.Tensor, 
                 nobjs: int,
                 ninits: int,
                 leaf_size: int,
                 method: str,
                 weight_method: str,
                 cluster_method: str,
                 ref_point = None,
                 space = None,
                 Cp: float = 0.1,
                 gp_kernel = None,
                 cat_dims : list[int] = None,
                 normalized: bool = False,
                 minimize: bool = False,
                 micro_al : bool = False,
                 seed: int = 42) -> None:
        
        self.dims       = dims
        self.bounds     = bounds
        self.nobjs      = nobjs
        self.ninits     = ninits
        self.leaf_size  = leaf_size
        self.Cp         = Cp
        self.normalized = normalized
        self.minimize   = minimize
        self.seed       = seed
        self.method     = method
        self.space      = space
        self.micro_al   = (self.space is not None) and micro_al

        self.weight_method = weight_method
        self.cluster_method = cluster_method
        self.gp_kernel = gp_kernel
        self.cat_dims = cat_dims
        if ref_point is None:
            self.given_ref_point = False
        else:
            self.given_ref_point = True
            self.ref_point = ref_point

        self.nodes      = []
        self.X          = torch.tensor([])
        self.Y          = torch.tensor([])

        self.ROOT = node(parent = None, nobjs = self.nobjs,
                         dims = self.dims, Cp=self.Cp, 
                         weight_method=self.weight_method, 
                         cluster_type=self.cluster_method)
        
        self.CURT = self.ROOT
        self.CURT_LABEL = None
        self.nodes.append(self.ROOT)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        self.init = True
        pass

    def populate(self):

        for n in self.nodes:
            n.clear_data()

        self.nodes.clear()
        new_root = node(parent=None, nobjs = self.nobjs, 
                        ref_point=self.ref_point,
                        dims=self.dims, Cp=self.Cp, 
                        weight_method=self.weight_method, 
                        cluster_type=self.cluster_method)
        
        self.nodes.append(new_root)

        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag(self.X, self.Y, self.curr_pareto)

    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() is True and node.X.shape[0] >= self.leaf_size \
                and node.is_splittable is True:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False

    def get_split_idx(self):
        split_by_samples = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return split_by_samples
    
    def treeify(self):
        self.populate()
        # self.CURT.split_data()
        while self.is_splitable():
            to_split = self.get_split_idx()
            print("==>to split:", to_split, " total:", len(self.nodes))
            for node_idx in to_split:
                parent = self.nodes[node_idx]
                assert parent.X.shape[0] >= self.leaf_size
                partition = parent.split_data()
                kids = []
                for data_X, data_Y in partition:
                    new_node = node(nobjs=self.nobjs, parent=parent, 
                                    ref_point=self.ref_point,
                                    dims=self.dims, Cp=self.Cp, 
                                    id=len(self.nodes), 
                                    weight_method=self.weight_method, 
                                    cluster_type=self.cluster_method)
                    
                    new_node.update_bag(data_X, data_Y, self.curr_pareto)
                    kids.append(new_node)
                    self.nodes.append(new_node)
                parent.update_kids(kids)
        print("==>total tree size:", len(self.nodes))
        return
    
    def select(self):
        self.CURT = self.ROOT
        while self.CURT.is_leaf() == False:
            self.CURT, self.CURT_LABEL = self.CURT.select_kid()
        return
    
    def backpropagation(self):

        pass
    
    def bo_get_ehvi_acq(self):
        with torch.no_grad():
            pred = self.gp.posterior(self.X).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.ref_point,
            Y=pred,
        )
        # MC EHVI
        acq_func = qExpectedHypervolumeImprovement(
            model=self.gp,
            ref_point=self.ref_point,
            partitioning = partitioning,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([100]), seed=self.inter_seed),
        )
        # Analytic EHVI
        # acq_func = ExpectedHypervolumeImprovement(
        #     model=self.gp,
        #     ref_point=self.ref_point,
        #     partitioning = partitioning
        # )
        return acq_func
    
    def bo_get_hvkg_acq(self):
        acq_func = qHypervolumeKnowledgeGradient(
            model=self.gp,
            ref_point=self.ref_point,
        )
        return acq_func
    
    def bo_get_entropy_search(self, method):

        num_pareto_samples = 20
        num_pareto_points = 5

        optimizer_kwargs = {
            "pop_size": 2000,
            "max_tries": 25,
        }

        ps, pf = sample_optimal_points(
            model=self.gp,
            bounds=self.bounds,
            num_samples=num_pareto_samples,
            num_points=num_pareto_points,
            optimizer=random_search_optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)
        if method=="mes":
        # Here we use the lower bound estimates for the MES
            return qLowerBoundMultiObjectiveMaxValueEntropySearch(
                model=self.gp,
                pareto_fronts=pf,
                hypercell_bounds=hypercell_bounds,
                estimation_type="LB",
            )
        elif method=="jes":
            return qLowerBoundMultiObjectiveJointEntropySearch(
                self.gp,
                ps,pf,hypercell_bounds,
                estimation_type="LB"
            )
        elif method=="pes":
            return qMultiObjectivePredictiveEntropySearch(
                self.gp,
                pareto_sets=ps
            )
        else:
            raise NotImplementedError
  
    def propose_sample(self, n=1, sampler='sobol', space = None):
        self.inter_seed = np.random.randint(1e6)
        total_samples = 2*1024  #TODO: Restart * Raw_sample style sampling

        if space is None:
            samples = draw_sobol_samples(self.bounds, n=total_samples, q=n, seed=self.inter_seed)
        else:
            samples = space.unsqueeze(-2)

        return samples
    
    def propose_mcts_sample(self, n=1, sampler='sobol', space = None):
        path = self.CURT.parent
        resample_times = 1
        resample_timeout = 100
        total_samples = 2*1024
        if self.CURT == self.ROOT:
            return self.propose_sample(n=1, sampler='sobol', space = space)

        if space is not None:
            # Fixed Space Samples
            total_samples = space.shape[0]
            samples = space
            selected_samples = samples[path.classifer.predict(samples.cpu().numpy()) == self.CURT_LABEL]

        elif sampler == 'sobol':
            # Sobol Samples
            samples = draw_sobol_samples(self.bounds, n=total_samples, q=n, seed=self.inter_seed).squeeze(-2)
            selected_samples = samples[path.classifer.predict(samples.cpu().numpy()) == self.CURT_LABEL]
            while selected_samples.shape[0] <= total_samples:
                self.inter_seed = np.random.randint(1e6)
                samples = draw_sobol_samples(self.bounds, n=total_samples, q=n, seed=self.inter_seed).squeeze(-2)
                samples_in_region = samples[path.classifer.predict(samples.cpu().numpy()) == self.CURT_LABEL]
                selected_samples = torch.vstack([selected_samples, samples_in_region])
                if resample_times > resample_timeout:
                    break
                resample_times += 1
        elif sampler == 'latin':
            # Latin Hypercube Samples
            cube = latin_hypercube(total_samples, self.dims)
            samples = from_unit_cube(cube, self.bounds[0].cpu().numpy(), self.bounds[1].cpu().numpy())
            samples = torch.tensor(samples)
            selected_samples = samples[path.classifer.predict(samples.cpu().numpy()) == self.CURT_LABEL]
            while selected_samples.shape[0] <= total_samples:
                samples = from_unit_cube(cube, self.bounds[0].cpu().numpy(), self.bounds[1].cpu().numpy())
                samples = torch.tensor(samples)
                samples_in_region = samples[path.classifer.predict(samples.cpu().numpy()) == self.CURT_LABEL]
                selected_samples = torch.vstack([selected_samples, samples_in_region])
                if (resample_times > resample_timeout) and (selected_samples.shape[0] > 0):
                    break
                resample_times += 1

        approx_precent = selected_samples.shape[0] / (resample_times*total_samples)
        print(f"[DDSP] Selected Region Approx. Size: {approx_precent*100:.2f}% "
              f"({selected_samples.shape[0]}/{(resample_times*total_samples)})")
        
        if len(selected_samples) == 0:
            print("[DDSP]Warning: Can't find valid sample in the region, back to global optimization.")
            X = samples
        elif space is not None:
            X = selected_samples
        else:
            X = selected_samples[torch.randint(len(selected_samples),(total_samples,))]

        X = X.to(dtype=torch.float64).unsqueeze(-2)
        return X
    
    def propose_bo(self, samples, acq_func, n=1):
        print("Proposing with BO")
        samples = samples.to(dtype=torch.float64)
        acq_value = acq_func(samples)
        X = samples.squeeze(-2)
        suggest_x = X[torch.argmax(acq_value)]
        return suggest_x.unsqueeze(0)
    
    def propose_bo_opt(self, acq_func, n=1):
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n,
            num_restarts=5,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return candidates
    
    def propose_adaboost(self, samples, n=1):
        self.ada = MultiOutputRegressor(AdaBoostRegressor())
        self.ada.fit(self.X.cpu().numpy(), self.Y.cpu().numpy())

        pred_Y = self.ada.predict(samples.cpu().numpy())
        pred_Y = torch.tensor(pred_Y)

        suggest_x = samples[is_non_dominated(pred_Y)]
        suggest_x = torch.stack(random.choices(suggest_x, k=n))

        return suggest_x
    
    def propose_xgboost(self, samples, n=1):
        self.xg = XGBRegressor(tree_method="hist", device="cuda")
        self.xg.fit(self.X.cpu().numpy(), self.Y.cpu().numpy())

        pred_Y = self.xg.predict(samples.cpu().numpy())
        pred_Y = torch.tensor(pred_Y)

        suggest_x = samples[is_non_dominated(pred_Y)]
        suggest_x = torch.stack(random.choices(suggest_x, k=n))

        return suggest_x

    def propose_rf(self, samples, n=1):
        self.rf = MultiOutputRegressor(RandomForestRegressor())
        self.rf.fit(self.X.cpu().numpy(), self.Y.cpu().numpy())

        pred_Y = self.rf.predict(samples.cpu().numpy())
        pred_Y = torch.tensor(pred_Y)

        suggest_x = samples[is_non_dominated(pred_Y)]
        suggest_x = torch.stack(random.choices(suggest_x, k=n))

        return suggest_x

    def propose_gaussian(self, samples, n=1):
        
        pred_Y = self.gp.posterior(samples).mean
        
        suggest_x = samples[is_non_dominated(pred_Y)]
        suggest_x = torch.stack(random.choices(suggest_x, k=n))
        return suggest_x

    def propose_random(self, n=1, space=None):
        if space is None:
            suggest_x = np.random.uniform(self.bounds[0], self.bounds[1], (n, self.dims))
            suggest_x = torch.tensor(suggest_x)
        else:
            suggest_x = rand_init_sample(n, space)
        return suggest_x

    def propose_init(self):

        if self.space is None:
            # Sobol from botorch
            # suggest_x = draw_sobol_samples(self.bounds, self.ninits, 
            #                            q=1, seed=self.inter_seed).squeeze(-2)

            # LHS from LA-MCTS
            cube = latin_hypercube(self.ninits, self.dims)
            samples = from_unit_cube(cube, self.bounds[0].cpu().numpy(), self.bounds[1].cpu().numpy())
            suggest_x = torch.tensor(samples)
        else:
            if self.micro_al:
                from BoomExplorerUtils.MicroAL import MicroAL
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
                print("[MicroAL] Start Clustering...")
                initializer = MicroAL(configs=initial_configs, n_dim=self.dims)
                suggest_x = torch.tensor(initializer.initialize(self.space.cpu().numpy()))
            else:
                # suggest_x = rand_init_sample(self.ninits, self.space)
                suggest_x = lhs_init_sample(self.ninits, self.space)

        return suggest_x

    def suggest(self):
        if self.init:
            self.inter_seed = 42 # Whatever, just don't do np rand here
            suggest_x = self.propose_init()
            self.init = False
        else:
            if self.method.startswith("mcts"):
                # MCTS Space Partition + Sample
                samples = self.propose_mcts_sample(space=self.space)
                method = self.method[5:]
            else:
                # Normal Sobol Sample
                samples = self.propose_sample(space=self.space)
                method = self.method

            try:
                if method == "ehvi":
                    acq = self.bo_get_ehvi_acq()
                    suggest_x = self.propose_bo(samples, acq)
                elif method == "hvkg":
                    acq = self.bo_get_hvkg_acq()
                    suggest_x = self.propose_bo(samples, acq)
                elif method in ["jes", "mes", "pes"]:
                    acq = self.bo_get_entropy_search(self.method)
                    suggest_x = self.propose_bo(samples, acq)
                elif method == "gaussian":
                    # Random Gaussian Progress Regression
                    suggest_x = self.propose_gaussian(samples)
                elif method == "adaboost":
                    # Random Adaboost Regression
                    samples = samples.squeeze(1)
                    suggest_x = self.propose_adaboost(samples)
                elif method == "xgboost":
                    # Random XGBoost Regression
                    samples = samples.squeeze(1)
                    suggest_x = self.propose_xgboost(samples)
                elif method == "rf":
                    # Random Forest Regression
                    samples = samples.squeeze(1)
                    suggest_x = self.propose_rf(samples)
                elif method == "rand":
                    # Random Walker Baseline
                    suggest_x = self.propose_random(space=self.space)
                else:
                    raise NotImplementedError
            except RuntimeError as e:
                traceback.print_exception(e)
                print("(suggest) Error encountered, back to random...")
                suggest_x = self.propose_random(space=self.space)
        return suggest_x

    def observe(self, X, Y):

        self.X = torch.cat((self.X, X))
        self.Y = torch.cat((self.Y, -Y)) if self.minimize else torch.cat((self.Y, Y))
        assert self.X.shape[0] == self.Y.shape[0]
        print(f"[DDSP] Total Collected Sample {self.X.shape[0]}")
        print("-" * 100)

        p_X = self.X
        p_Y = self.Y
        # tmp = torch.concat((p_X, p_Y), dim=1)

        pareto_Y = p_Y[is_non_dominated(p_Y)]
        # pareto_mean = torch.mean(pareto_Y, dim=0)

        if self.given_ref_point is False:
            self.ref_point = torch.min(p_Y, dim=0).values
            
        volume = get_hv(pareto_Y, self.ref_point)

        print("Pareto Set Size:", pareto_Y.shape[0])
        print("Ref Point:", self.ref_point)
        print("Hypervolume:", volume)

        self.curr_pareto = pareto_Y

        if self.gp_kernel is None:
            if "jes" in self.method:
                self.gp = SingleTaskGP(p_X, p_Y,
                                        outcome_transform=Standardize(self.nobjs))
            else:
                covar = ScaleKernel(MaternKernel())
                self.gp = SingleTaskGP(p_X, p_Y, covar_module=covar,
                                        outcome_transform=Standardize(self.nobjs))
        elif self.gp_kernel == "not_gp":
            # TODO: error when optimize
            xg = XGBRegressor(tree_method="hist", device="cuda")
            xg.fit(p_X.cpu().numpy(), p_Y.cpu().numpy())
            def xg_predict(X: torch.Tensor) -> torch.Tensor:
                pred_Y = xg.predict(X.squeeze(1).cpu().numpy())
                pred_Y = torch.tensor(pred_Y).unsqueeze(-2)
                return pred_Y
            self.gp = GenericDeterministicModel(xg_predict, self.nobjs)
        else:
            self.gp = SingleTaskGP(p_X, p_Y, covar_module=self.gp_kernel)


        if "rand" not in self.method and \
            "rf" not in self.method and \
            "adaboost" not in self.method and \
            "xgboost" not in self.method and \
            self.gp_kernel != "not_gp":

            self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(self.mll)

        if "mcts" in self.method:
            self.treeify()
            self.select()
            self.print_tree()

        if self.space is not None:
            mask = ~torch.any(torch.stack([torch.all(self.space == x, dim=1) for x in X]), 
                              dim=0)
            self.space = self.space[mask]
            print("Unvisited Space:", self.space.shape[0])
        return pareto_Y, volume

    def print_node(self, n):
        if n.id == self.CURT.id:
            print("\033[0;34;40m" + str(n) + "\033[0m")
        else:    
            print(n)
        for node in self.nodes:
            if(node.parent is n):
                self.print_node(node)
        return

    def in_selected_space(self, X):
        path = self.CURT.parent
        selected = path.classifer.predict(X.cpu().numpy()) == self.CURT_LABEL
        return selected

    def get_space_partition(self, X):
        node_label = {}
        for i in range(len(self.nodes)):
            if not self.nodes[i].is_leaf():
                node_label[f"node_{i}"] = self.nodes[i].classifer.predict(
                    X.cpu().numpy()).tolist()
        return node_label

    def print_tree(self):
        print('-'*100)
        self.print_node(self.ROOT)
