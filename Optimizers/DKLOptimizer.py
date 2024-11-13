from Problems import Problem
from Optimizers import Optimizer

from Optimizers.DDSP import BOOMExplorerSolver
import torch

class DKLOptimizer(Optimizer):

    def __init__(self, problem : Problem, 
                 ninits: int,
                 minimize: bool = False,
                 seed: int = 42) -> None:
        super().__init__(problem=problem)

        space = None if self.problem.dataset == {} else self.problem.input_space
        self.minimize = minimize


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
            'microal': False
        }
        assert space is not None, "Input space is required for DKLOptimizer (BOOM-Explorer)"
        self.model = BOOMExplorerSolver(
            configs         = configs,
            dims            = self.problem.dims,
            space           = None if space is None else torch.tensor(space),
            ninits          = ninits,
            seed            = seed,
            ref_point       = torch.tensor([-1, -1]) if minimize else torch.tensor([0, 0])
        )
    
    def suggest(self):
        return self.model.suggest().tolist()

    def observe(self, X, Y):

        self.sampled_X.append(X)
        self.sampled_Y.append(Y)

        if self.minimize: # minimize -> maximize
            self.model.observe(torch.tensor(X,dtype=torch.double), 
                               -torch.tensor(Y, dtype=torch.double))
        else:
            self.model.observe(torch.tensor(X,dtype=torch.double), 
                               torch.tensor(Y, dtype=torch.double))
        return
    
# Test Example
# if __name__ == "__main__":

#     import json
#     from Flows import VivadoFlow
#     from Design.riscv import VexRiscvGenerator, VexRiscvSimulator
#     from Problems import VexRiscvProblem

#     vexgen = VexRiscvGenerator()
#     vexsim = VexRiscvSimulator(benchmark="tinyml_AD")
#     vvdflow = VivadoFlow("xcau25p-sfvb784-1-i")

#     dataset_path = "PresampledDataset/VexRiscv/tinyml_AD_1963.json"

#     with open(dataset_path, "r") as dataset:
#         datadict = json.load(dataset)

#     problem = VexRiscvProblem(vexgen, vexsim, vvdflow, datadict, norm_ref=[0.5, 5000])

#     opt = SpadeOptimizer(problem=problem, 
#                          ninits = 10, 
#                          leaf_size = 10,
#                          method = 'mcts-ehvi',
#                          weight_method = 'global-hv',
#                          cluster_method = 'hybrid')
    
#     test_iter = 1

#     for _ in range(test_iter):
#         X = opt.suggest()
#         Y = problem.evaluate(X)
#         opt.observe(X, Y)