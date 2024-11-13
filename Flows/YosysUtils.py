import re
area_pat = r"\s+Chip area for top module .+:\s+(\d+.\d+)"

def read_report(filename):
    with open(filename, 'r') as file:
        for line in file.readlines():
            match_area = re.match(area_pat, line)
            if match_area:
                return {"Area": float(match_area.group(1))}
    return {"Area": False}