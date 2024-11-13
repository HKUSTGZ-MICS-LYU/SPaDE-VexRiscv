import json

def read_report(report, report_ratio = False):

    with open(report, 'r') as rep:
        results = json.load(rep)
    
    assert len(results['fmax']) == 1, "[SPaDE][Error] Multiple clocks found..."

    fmax = dict(results['fmax']).popitem()[1]['achieved']
    util = {}
    for key, item in results['utilization'].items():
        util[key] = item['used'] / item['available'] if report_ratio else item['used']
    res = util
    res['fmax'] = fmax
    return res

# if __name__ == "__main__":
#     print(read_report("build/report/report.json", True))