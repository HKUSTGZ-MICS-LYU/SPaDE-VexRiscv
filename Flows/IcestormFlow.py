from Flows import Flow
from Flows.IcestormUtils import *

import os
import sys
import time
import shutil
import subprocess

FLOW_BUILD_DIR = "build"
FLOW_REPORT_DIR = FLOW_BUILD_DIR + '/report'
FLOW_RTL_DIR = FLOW_BUILD_DIR + '/rtl'
FLOW_IR_DIR = FLOW_BUILD_DIR + '/synth'

FLOW_DIRS = " ".join([FLOW_BUILD_DIR, FLOW_REPORT_DIR, FLOW_RTL_DIR, FLOW_IR_DIR])

SYNTHFLAGS = "-dsp -abc2"

class IcestormFlow(Flow):

    def __init__(self, part, pack) -> None:
        self.part = part # e.g. up5k
        self.pack = pack # e.g. sg48
        self.evaluate_time = -1
        super().__init__()

    def evaluate(self, generated_rtl : str, top_name : str, 
                 multi_files = False,
                 report_ratio = False):
        
        evalutate_time_start = time.time()

        assert os.path.exists(generated_rtl), f"[SpaDE][Error] RTL path not found! ({generated_rtl})"
        os.system(f'mkdir -p {FLOW_DIRS}')
        if multi_files:
            shutil.copytree(generated_rtl, FLOW_RTL_DIR, dirs_exist_ok=True)
            # TODO: need an assertion to check if copy successed here.
            rtl_file = os.listdir(FLOW_RTL_DIR)
            rtl_files = [FLOW_RTL_DIR+'/'+rtl for rtl in rtl_file if rtl.endswith(".v") or rtl.endswith(".sv")]
            assert len(rtl_files) != 0, f"[SpaDE][Error] No RTL found in path! ({generated_rtl})"
            rtl_file = " ".join(rtl_files)
        else:
            rtl_file = os.path.basename(generated_rtl)
            rtl_path = FLOW_RTL_DIR + '/' + rtl_file
            shutil.copy(generated_rtl, rtl_path)
            assert os.path.exists(rtl_path), f"[SpaDE][Error] failed to copy RTL! ({rtl_path})"

        PNRFLAGS = f"--top {top_name} --package {self.pack} --report {FLOW_REPORT_DIR}/report.json"
        # TODO: add PCF constr file if available
        cmd_list = [
            # Logic Synthesis
            f"yosys -p \"synth_ice40 {SYNTHFLAGS} -top {top_name} -json {FLOW_IR_DIR}/{top_name}.json \" {rtl_file} > {FLOW_BUILD_DIR}/yosys.log",
            # Place & Route
            f"nextpnr-ice40 {PNRFLAGS} --{self.part} --json {FLOW_IR_DIR}/{top_name}.json --asc {FLOW_IR_DIR}/{top_name}.asc -q --log {FLOW_BUILD_DIR}/next_pnr.log"
        ]
        for cmd in cmd_list:
            print(cmd)
            ret = os.system(cmd)
            if ret != 0:
                print(f"[SpaDE][Error] cmd failed! ({cmd})")
                return False
            # assert ret == 0, "[SpaDE][Error] cmd failed!"
        
        self.evaluate_time = time.time() - evalutate_time_start

        if ret==0:
            result = {}
            result.update(read_report(f"{FLOW_REPORT_DIR}/report.json", report_ratio))
            return result
        else:
            print("[SPaDE][Error] Icestorm Synthesis failed!")
            return False
    
# Test Example
# if __name__ == "__main__":
#     iceflow = IcestormFlow("up5k", "sg48")
#     res = iceflow.evaluate("Design/riscv/VexiiRiscv/hw/rtl", "IceSoc", True)
#     print(res)