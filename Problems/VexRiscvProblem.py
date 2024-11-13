from Design.riscv import VexRiscvGenerator, VexRiscvSimulator
from Flows import VivadoFlow

from Problems import Problem

import datetime

class VexRiscvProblem(Problem):

    def __init__(self, 
                 generator: VexRiscvGenerator, 
                 simulator: VexRiscvSimulator, 
                 flow: VivadoFlow,
                 dataset: dict = None,
                 norm_ref = None,
                 allow_invalid = False) -> None:
        
        super().__init__(generator, simulator, flow, dataset)
        
        self.design_path = self.generator.generator_output
        self.nobjs = 2
        self.allow_invalid = allow_invalid

        if norm_ref is not None:
            self.norm_ref = norm_ref
        else:
            self.norm_ref = [1, 1]
        assert len(self.norm_ref) == self.nobjs

    def sample(self, X):

        res = {}
        print(f"[SPaDE][INFO] VexRiscvProblem: Generating VexRiscv Design ({datetime.datetime.now()})")
        valid = self.generator.validate(X)
        if valid is False:
            print("[SPaDE][WARN] VexRiscvProblem: Invalid VexRiscv Design!")
            if self.allow_invalid:
                print("[SPaDE][INFO] VexRiscvProblem: Design corrected to solve the conflicts.")
                X = self.generator.validate(X, return_corrected=True)
            else:
                raise RuntimeError("Invalid Design is not allowed!")
                
        ret = self.generator.generate(X)
        
        print(f"[SPaDE][INFO] VexRiscvProblem: Simulating VexRiscv Design on {self.simulator.benchmark} ({datetime.datetime.now()})")
        res.update(self.simulator.simulate(X))

        # 2-step Fmax Synthesis
        print(f"[SPaDE][INFO] VexRiscvProblem: Synthsizing VexRiscv Design to get Fmax ({datetime.datetime.now()})")
        start_clk = 1000
        synth_res_1 = self.flow.evaluate(self.design_path, 'VexRiscv', clk = start_clk)
        assert synth_res_1, "[SPaDE][Error] Evaluation Failed!"
        clk_min = synth_res_1['MinClock']
        clk_relax_factor = 1.1
        second_clk = clk_min * clk_relax_factor

        print(f"[SPaDE][INFO] VexRiscvProblem: Synthsizing VexRiscv Design with Fmax of {1000/second_clk:.2f} MHz ({datetime.datetime.now()})")
        synth_res_2 = self.flow.evaluate(self.design_path, 'VexRiscv', clk = second_clk)
        res.update(synth_res_2)
        return res
    
    def objective(self, data: dict) -> list:

        energy = data["Power"]*data["Cycles"]*data["Clock"] / 10**9
        area = data["LUT"]+data["REG"]

        obj = [energy / self.norm_ref[0], 
                area / self.norm_ref[1]]
        
        return obj
    
# Test Example
# if __name__ == "__main__":

#     import json
#     import random
#     from Flows import VivadoFlow
#     from Design.riscv import VexRiscvGenerator, VexRiscvSimulator

#     vexgen = VexRiscvGenerator()
#     vexsim = VexRiscvSimulator(benchmark="tinyml_AD")
#     vvdflow = VivadoFlow("xcau25p-sfvb784-1-i")

#     dataset_path = "PresampledDataset/VexRiscv/tinyml_AD_1963.json"

#     with open(dataset_path, "r") as dataset:
#         datadict = json.load(dataset)

#     given_space_problem = VexRiscvProblem(vexgen, vexsim, vvdflow, datadict)
#     space_size = given_space_problem.space_size()
    
    # # Select one input from the given space 
    # input_X = given_space_problem.input_space[random.randint(0, space_size - 1)]
    # print(given_space_problem.sample(input_X))

    # # Select one input from the random space
    # import numpy as np
    # input_X = np.random.randint(given_space_problem.bounds[0], given_space_problem.bounds[1]).tolist()
    # print(given_space_problem.sample(input_X))

    # # Select one input from the random space and force evaluation
    # import numpy as np
    # input_X = np.random.randint(given_space_problem.bounds[0], given_space_problem.bounds[1]).tolist()
    # print(given_space_problem.sample(input_X, allow_invalid=True))