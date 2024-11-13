from Problems import Problem
from Optimizers import Optimizer

import torch
import numpy as np

class RandomWalker(Optimizer):
    def __init__(self, problem: Problem, 
                 ninits: int,
                 seed: int = 42) -> None:
        
        super().__init__(problem)
        self.ninits = ninits
        self.lb = np.array(self.problem.bounds[0])
        self.ub = np.array(self.problem.bounds[1])
        self.space = None if self.problem.dataset == {} else self.problem.input_space
        self.init = True

        np.random.seed(seed)

    def suggest(self):
        if self.init:
            self.init = False
            if self.space is None:
                X = np.random.random((self.ninits,self.problem.dims)) * \
                      (self.ub - self.lb) + self.lb
                return X.tolist()
            else:
                idx = np.random.permutation(len(self.space))[:self.ninits]
                space = np.array(self.space)
                return space[idx].tolist()
        else:
            if self.space is None:
                X = np.random.random((1, self.problem.dims)) * \
                    (self.ub - self.lb) + self.lb
                return X.tolist()
            else:
                idx = [np.random.permutation(len(self.space))[0]]
                space = np.array(self.space)
                return space[idx].tolist()

    
    def observe(self, X, Y):

        self.sampled_X.append(X)
        self.sampled_Y.append(Y)
        
        # We are random walking, so we don't need to observe anything
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

#     opt = RandomWalker(problem=problem, 
#                          ninits = 10)
    
#     test_iter = 21

#     for _ in range(test_iter):
#         X = opt.suggest()
#         Y = problem.evaluate(X)
#         opt.observe(X, Y)