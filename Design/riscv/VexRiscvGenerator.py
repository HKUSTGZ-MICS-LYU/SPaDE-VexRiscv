from Design import Generator

from Design.riscv.VexRiscvUtils import (
    gen_config_from_vector, 
    gen_vector_from_config, 
    get_vector_bounds_from_config
)

import os
import subprocess
import sys
import time
import json
import copy

class VexRiscvGenerator(Generator):

    VEXRISCV_PATH = "Design/riscv/VexRiscv"

    VEXRISCV_TOP = "VexRiscv"

    VEXRISCV_RECIPE = [f"cd {VEXRISCV_PATH}",
                       f"rm -f VexRiscv.v", # this is quite dangerous
                       "make VexRiscv.v"]

    VEXRISCV_OUTPUT = f"{VEXRISCV_PATH}/VexRiscv.v"

    VEXRISCV_DESIGN_SPACE = f"Design/riscv/DesignSpace/VexRiscv.json"
    VEXRISCV_DESIGN_CONFIG = f"{VEXRISCV_PATH}/DSEConfig.json"

    def __init__(self, verbose = False) -> None:
        self.verbose = verbose
        with open(self.VEXRISCV_DESIGN_SPACE, "r") as ds:
            self.design_space = json.load(ds)
        self.ub = get_vector_bounds_from_config(self.design_space)
        self.lb = [0] * len(self.ub)
        super().__init__(
            generator_path = self.VEXRISCV_PATH,
            generator_output = self.VEXRISCV_OUTPUT)
        
    def generate(self, X : list):
        generator_time_start = time.time()
        # The function `gen_config_from_vector` has side effects, which will
        # destroy the list/array.
        # TODO: find a better way to implement it
        input_X = copy.deepcopy(X)
        design_space = copy.deepcopy(self.design_space)

        assert len(input_X) == len(self.ub)

        # Create Configuration File
        return_X, config = gen_config_from_vector(input_X, design_space)

        with open(self.VEXRISCV_DESIGN_CONFIG, "w") as ds_out:
            json.dump(config, ds_out, indent='\t')

        # Start generation
        cmd = " && ".join(self.VEXRISCV_RECIPE)
        
        print(cmd)
        ret = subprocess.run(cmd, shell=True, text=True,
                             capture_output=True)
        if self.verbose:
            print(ret.stdout)
            
        assert ret.returncode == 0, f"[SPaDE][ERROR] Generation failed!\n{ret.stderr}"
        self.generator_time = time.time() - generator_time_start
        # TODO: return something else
        return ret.returncode
    
    def validate(self, X, return_corrected: bool = False):

        input_X = copy.deepcopy(X)
        design_space = copy.deepcopy(self.design_space)
        return_X, config = gen_config_from_vector(input_X, design_space)

        is_valid = (X == return_X) # valid if input is equal

        if return_corrected:
            return return_X
        else:
            return is_valid
        
# Test Example
# if __name__ == "__main__":
#     import numpy as np
#     vexgen = VexRiscvGenerator()
#     ub = np.array(vexgen.ub)
#     lb = np.zeros_like(ub)
#     vexgen.generate(np.random.randint(lb,ub).tolist())