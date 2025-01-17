from Flows import Flow
from Flows.DesignCompilerUtils import (read_power_report, read_qor_report)

import os
import time
import shutil

FLOW_BUILD_DIR = "build"
FLOW_REPORT_DIR = FLOW_BUILD_DIR + '/report'
FLOW_RTL_DIR = FLOW_BUILD_DIR + '/rtl'

FLOW_DIRS = " ".join([FLOW_BUILD_DIR, FLOW_REPORT_DIR, FLOW_RTL_DIR])

TCL_TEMPLATE = "Flows/templates/dc_synth_template.tcl"
TCL_GENERATE = FLOW_BUILD_DIR + '/dc_synth.tcl'

DEFAULT_PDK_LIB_PATH = "Flows/pdk/sky130_1v89.db"

class DesignCompilerFlow(Flow):
        
    def __init__(self, pdk_path : str = DEFAULT_PDK_LIB_PATH) -> None:
        self.evaluate_time = -1
        self.pdk_path = pdk_path
        super().__init__()
    
    def evaluate(self, generated_rtl : str, top_name : str, 
                 clk = 1000, # default clock setting
                 clk_name = 'clk',
                 multi_files = False):
        
        evalutate_time_start = time.time()

        assert os.path.exists(generated_rtl), f"[SpaDE][Error] RTL path not found! ({generated_rtl})"
        os.system(f'mkdir -p {FLOW_DIRS}')
        rtl_file = os.path.basename(generated_rtl)
        rtl_path = FLOW_RTL_DIR + '/' + rtl_file
        shutil.copy(generated_rtl, rtl_path)
        assert os.path.exists(rtl_path), f"[SpaDE][Error] failed to copy RTL! ({rtl_path})"

        if multi_files:
            # TODO: implement multi-file rtl flow
            raise NotImplementedError
        else:
            with open(TCL_TEMPLATE, 'r') as template:
                template_str = template.read()

            tcl_str = template_str.format(
                RTL = rtl_path,
                TOP_NAME = top_name,
                PDK = self.pdk_path,
                CLOCK_PERIOD=clk,
                CLOCK_NAME=clk_name,
                REPORT_PATH= FLOW_REPORT_DIR
            )

            with open(TCL_GENERATE, 'w') as tcl:
                tcl.write("# This file is automatically generated at ")
                tcl.write(time.asctime(time.localtime(time.time()))+"\n")
                tcl.write(tcl_str)
        cmd = f"dc_shell-t -f {TCL_GENERATE} > {FLOW_BUILD_DIR}/dc.log"
        print(cmd)
        ret = os.system(cmd)
        
        self.evaluate_time = time.time() - evalutate_time_start

        if ret==0:
            result = {}
            result.update(read_power_report(f"{FLOW_REPORT_DIR}/dc_power.rpt"))
            result.update(read_qor_report(f"{FLOW_REPORT_DIR}/dc_qor.rpt"))
            return result
        else:
            print("[SPaDE][ERROR] Design Compiler Synthesis failed!")
            return False
        
# Test Example
# if __name__ == "__main__":
#     dcflow = DesignCompilerFlow()
#     res = dcflow.evaluate("Design/riscv/VexiiRiscv/VexiiRiscv.v",
#                           "VexiiRiscv", clk = 10)
#     print(res)