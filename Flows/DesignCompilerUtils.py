import re

power_pat = r'Total(\s*[0-9]+\.[0-9e+\-]+\s*[munp]W\s*){3}([0-9]+\.[0-9e+\-]+)\s*([mnu]W)'
def read_power_report(report):
    
    power = 0
    unit = ""

    file = open(report, 'r')
    for line in file.readlines():
        match_power = re.match(power_pat, line)
        if match_power:
            power = (match_power.group(2))
            unit = (match_power.group(3))
    file.close()

    if power:
        ex = 1
        if(unit=="mW"): 
            ex = 1000
        elif(unit=="nW"):
            ex = 1/1000
        return {"Power": float(power) * ex}
    else:
        return {"Power": False}


area_pat = r'\s*Design Area:\s*([0-9]+\.[0-9]+)'
clock_pat = r'\s*Critical Path Clk Period:\s*([0-9]+\.[0-9]+)'
slack_pat = r'\s*Total Negative Slack:\s*(\-[0-9]+\.[0-9]+)'
cputime_pat = r'\s*Overall Compile Time:\s*([0-9]+\.[0-9]+)'

def read_qor_report(filename):

    area = -1
    clock = -1
    cputime = -1
    violation = False

    file = open(filename, 'r')
    for line in file.readlines():
        match_slack = re.match(slack_pat, line)
        match_area = re.match(area_pat, line)
        match_time = re.match(clock_pat, line)
        match_cputime = re.match(cputime_pat, line)
        if match_slack:
            slack = (float)(match_slack.group(1))
            if(slack != 0):
                print("Clock period violation(worst slack: %f)! Design cannot be used." % slack)
                violation = True
        if match_area:
            area = (float)(match_area.group(1))
        if match_time:
            clock = (float)(match_time.group(1))
        if match_cputime:
            cputime = (float)(match_cputime.group(1))
    file.close()
    if violation:
        clock = -1
    return {'Area': area, 'Clock': clock, 'CPUTime': cputime}