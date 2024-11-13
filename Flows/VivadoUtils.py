import re

def read_util_report(filename):

    slice_lut_pat = r"\|\s+Slice LUTs\s+\|\s+(\d+)\s"
    slice_reg_pat = r"\|\s+Slice Registers\s+\|\s+(\d+)\s"
    clb_lut_pat = r"\|\s+CLB LUTs\s+\|\s+(\d+)\s"
    clb_reg_pat = r"\|\s+CLB Registers\s+\|\s+(\d+)\s"
    dsp_pat = r"\|\s+DSPs\s+\|\s+(\d+)\s"
    ram_pat = r"\|\s+Block RAM Tile\s+\|\s+([0-9\.]+)\s"

    result = {
        'LUT' : -1, 
        'REG' : -1,
        'DSP' : -1,
        'RAM' : -1}

    report = open(filename, "r")

    for line in report.readlines():
        
        slice_lut_match = re.match(slice_lut_pat, line)
        slice_reg_match = re.match(slice_reg_pat, line)
        clb_lut_match = re.match(clb_lut_pat, line)
        clb_reg_match = re.match(clb_reg_pat, line)
        dsp_match = re.match(dsp_pat, line)
        ram_match = re.match(ram_pat, line)

        if slice_lut_match:
            result['LUT'] = int(slice_lut_match.group(1))
        elif clb_lut_match:
            result['LUT'] = int(clb_lut_match.group(1))
        if slice_reg_match:
            result['REG'] = int(slice_reg_match.group(1))
        elif clb_reg_match:
            result['REG'] = int(clb_reg_match.group(1))
        if dsp_match:
            result['DSP'] = int(dsp_match.group(1))
        if ram_match:
            result['RAM'] = float(ram_match.group(1))

    report.close()

    return result

def read_clock_report(filename):
    # Take timing violation as synthesis failure
    clk_pat = r'\s*Period\(ns\):\s*([0-9\.]+)'

    report = open(filename, "r")

    result = {'Clock' : -1 }

    if "VIOLATED" in report.read(): 
        print("Timing Violation Detected! Abort Timing!")
        return result
    else:
        report.seek(0)
        for line in report.readlines():
            clk_match = re.match(clk_pat, line)
            if clk_match:
                result['Clock'] = float(clk_match.group(1))
    report.close()
    return result

def read_delay_report(filename):
    # Calculate timing from WNS and clock
    clk_pat = r'\s*Requirement:\s*([0-9\.]+)'
    slack_pat = r'\s*Slack\s*\([A-Z]+\)\s:\s*([\-0-9\.]+)'

    report = open(filename, "r")

    result = {}

    clk = 0
    report.seek(0)
    for line in report.readlines():
        clk_match = re.match(clk_pat, line)
        slack_match = re.match(slack_pat, line)
        if slack_match:
            slack = float(slack_match.group(1))
        if clk_match:
            clk = float(clk_match.group(1))
            result['MinClock'] = clk - slack
            break
    report.close()
    return result

def read_power_report(filename):

    report = open(filename, "r")

    power_pat = r"\|\s+Total On-Chip Power \(W\)\s+\|\s+([0-9\.]+)\s"

    result = {'Power' : -1}
    
    for line in report.readlines():
        
        power_match = re.match(power_pat, line)

        if power_match:
            result['Power'] = float(power_match.group(1))
    
    report.close()

    return result