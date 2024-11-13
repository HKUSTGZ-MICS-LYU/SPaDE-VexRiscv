from Design import Simulator

from Design.riscv.VexRiscvUtils import (
    gen_config_from_vector, 
    gen_vector_from_config, 
    get_vector_bounds_from_config,
    parse_simulation_report
)

import os
import sys
import time
import subprocess
import json
import copy

class VexRiscvSimulator(Simulator):

    VEXRISCV_PATH = "Design/riscv/VexRiscv"

    VEXRISCV_DESIGN_SPACE = f"Design/riscv/DesignSpace/VexRiscv.json"
    VEXRISCV_DESIGN_CONFIG = f"{VEXRISCV_PATH}/DSEConfig.json"

    def __init__(self, output : str = None, verbose = False, benchmark : str = None) -> None:
        self.verbose = verbose
        self.benchmark = benchmark
        with open(self.VEXRISCV_DESIGN_SPACE, "r") as ds:
            self.design_space = json.load(ds)
        self.ub = get_vector_bounds_from_config(self.design_space)
        super().__init__(
            simulator_path = self.VEXRISCV_PATH,
            simulator_output = output)
        
    def simulate(self, X : list, benchmark : str = None):
        if benchmark is None:
            benchmark = self.benchmark

        simulate_time_start = time.time()
        # Clean previous report before the simulation starts
        self.report = None

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

        valid_tinyml_benchmarks = ["tinyml_AD", "tinyml_KWS", "tinyml_VWW", "tinyml_RESNET"]

        # Start generation
        if benchmark == "dhrystone":
            self.current_benchmark = "dhrystone"
            cmd = f"cd {self.VEXRISCV_PATH} && make"
        elif benchmark in valid_tinyml_benchmarks:
            self.current_benchmark = "tinyml"
            cmd = f"cd {self.VEXRISCV_PATH} && make all-tflite TFLITE_BENCH={benchmark[7:]}"
        else:
            raise NotImplementedError(f"[SPaDE][ERROR] Benchmark {benchmark} is not implemented!")

        print(cmd)        
        ret = subprocess.run(cmd, shell=True, text=True,
                            capture_output=True)

        self.report = ret.stdout

        if self.verbose:
            print(ret.stdout)

        if self.simulator_output is not None:
            with open(self.simulator_output, "w") as out:
                out.write(ret.stdout)

        assert ret.returncode == 0, f"[SPaDE][ERROR] Simulation failed!\n{ret.stderr}"
        self.simulate_time = time.time() - simulate_time_start

        return self.parse_report()

    def parse_report(self):
        assert self.report is not None, "[SPaDE][ERROR] Report not found, Please run simulation first!"
        cycle = parse_simulation_report(self.report, self.current_benchmark)
        return {'Cycles': cycle}

# Test Example
# if __name__ == "__main__":
#     import numpy as np
#     vexsim = VexRiscvSimulator(output="sim.log")
#     ub = np.array(vexsim.ub)
#     lb = np.zeros_like(ub)
#     vexsim.simulate(np.random.randint(lb,ub).tolist(), "dhrystone")
#     print(vexsim.parse_report())