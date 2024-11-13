from abc import ABC, abstractmethod

from Design import Generator, Simulator
from Flows import Flow

import json

class Problem:

    dims : int
    nobjs : int # To be given by the implementation
    bounds : list

    def __init__(self, 
                 generator: Generator,
                 simulator: Simulator,
                 flow: Flow,
                 dataset: dict = None) -> None:
        
        self.generator = generator
        self.simulator = simulator
        self.flow      = flow

        if self.generator is not None:
            self.bounds = [self.generator.lb, self.generator.ub]
            assert len(self.generator.lb) == len(self.generator.ub)
            self.dims = len(self.generator.lb)
        else:
            assert (self.bounds is not None) and (self.dims is not None)

        self.dataset = dataset

        self.input_space = []
        self.data_space = []

        if self.dataset is None:
            self.dataset = {}
        else:
            for key, data in dataset.items():
                assert "Embedding" in data, f"[SPaDE][Error] KeyError: Embedding not found in dataset, name={key}."
                self.input_space.append(data["Embedding"])
                self.data_space.append(data)

        return super().__init__()
    
    def space_size(self) -> int:
        return len(self.input_space)
    
    def evaluate(self, X, objective = None) -> list:
        # objective should be a function (dict -> list)
        # X and Y should be a list[list] (multiple designs)
        Y = []
        for _x in X:
            if _x in self.input_space:
                # if data is given in dataset, directly check the results
                idx = self.input_space.index(_x)
                data = self.data_space[idx]
            else:
                # if dataset is not given in dataset, invoke sampling flow
                data = {}
                data["Embedding"] = _x
                res = self.sample(_x)
                data.update(res)

                self.input_space.append(_x)
                self.data_space.append(data)
                self.dataset[f"{len(self.input_space)}"] = data
                
            if objective is not None:
                _y = objective(data)
            elif getattr(self.objective, '__isabstractmethod__', False):
                _y = data
            else:
                _y = self.objective(data)
            Y.append(_y)
        return Y

    def save(self, dataset_file):
        with open(dataset_file, "w") as dataset:
            json.dump(self.dataset, dataset, indent='\t')
        return

    @abstractmethod
    def sample(self, X) -> dict:
        # User-defined Evaluation/Sample Flow
        pass
    
    @abstractmethod
    def objective(self, data: dict) -> list:
        # User-defined Objective(s) Formulation
        pass


# Test Example
# if __name__ == "__main__":

#     import json
#     import random
#     from Flows import VivadoFlow
#     from Design.riscv import VexRiscvGenerator, VexRiscvSimulator

#     vexgen = VexRiscvGenerator()
#     vexsim = VexRiscvSimulator()
#     vvdflow = VivadoFlow("xcau25p-sfvb784-1-i")

#     dataset_path = "PresampledDataset/VexRiscv/tinyml_AD_1963.json"

#     with open(dataset_path, "r") as dataset:
#         datadict = json.load(dataset)

#     given_space_problem = Problem(vexgen, vexsim, vvdflow, datadict)
#     space_size = given_space_problem.space_size()
    
#     # Select one input from the given space 
#     input_X = [given_space_problem.input_space[random.randint(0, space_size - 1)]]
#     print(given_space_problem.evaluate(input_X))

#     def extract_PPA(data):
#         # Example of PPA Objective Extraction
#         return [data["Power"], 
#                 data["Cycles"]*data["Clock"], 
#                 data["LUT"]+data["REG"]]

#     print(given_space_problem.evaluate(input_X, extract_PPA))