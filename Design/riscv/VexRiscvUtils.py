import re

def safe_remove(list, item):
    if item in list:
        list.remove(item)
    return

def conflict_resolve(config, rconf, key, skey):
    if key == 'Universe':
        if (skey == 'NoMemory') and rconf['Universe']['NoMemory']:
            safe_remove(config['Universe']['NoWriteBack'], False)
            safe_remove(config['Shift'], 'FL')
            safe_remove(config['Branch'], False)
            safe_remove(config['MulDiv']['MulType'], 'Iterative')
            safe_remove(config['MulDiv']['MulType'], 'Mul16')
            safe_remove(config['MulDiv']['MulType'], 'Buffer')
            safe_remove(config['DBus']['busType'], 'Cached')
        elif (skey == 'NoWriteBack') and rconf['Universe']['NoWriteBack']:
            safe_remove(config['MulDiv']['MulType'],'Mul16')
            safe_remove(config['MulDiv']['MulType'],'Buffer')
            safe_remove(config['DBus']['earlyWaysHits'], True)
        elif (skey == 'HasMulDiv') and (not rconf['Universe']['HasMulDiv']):
                for i in config['MulDiv']:
                    config['MulDiv'][i] = ['Undefined']
    if key == 'MulDiv':
        if (skey == 'MulType') and (rconf['MulDiv']['MulType'] != 'Buffer'):
            config['MulDiv']['BuffIn'] = ['Undefined']
            config['MulDiv']['BuffOut'] = ['Undefined']
        if (skey == 'MulType') and (rconf['MulDiv']['MulType'] != 'Iterative'):
            config['MulDiv']['MulUnroll'] = ['Undefined']
    if key == 'RegFileAsync':
        if (rconf['RegFileAsync'] == False) and (rconf['Universe']['ExecuteRegfile'] == False):
            config['IBus']['injectorStage'] = [True]
    if key == 'DBus':
        if (skey == 'cacheSize') and (rconf['DBus']['busType'] == 'Cached'):
            max_way = rconf['DBus']['cacheSize'] / 512
            config['DBus']['wayCount'] = [i for i in config['DBus']['wayCount'] if i <= max_way]
        elif (skey == 'memDataWidth'):
            max_width = rconf['DBus']['memDataWidth']
            config['DBus']['cpuDataWidth'] = [i for i in config['DBus']['cpuDataWidth'] if i <= max_width]
        elif (skey == 'busType') and (rconf['DBus']['busType'] == 'Simple'):
            config['DBus']['memDataWidth'] = ['Undefined']
            config['DBus']['cpuDataWidth'] = ['Undefined']
            config['DBus']['bytePerLine'] = ['Undefined']
            config['DBus']['relaxedRegister'] = ['Undefined']
            config['DBus']['earlyWaysHits'] = ['Undefined']
            config['DBus']['asyncTagMemory'] = ['Undefined']
            config['DBus']['cacheSize'] = ['Undefined']
            config['DBus']['wayCount'] = ['Undefined']
        elif (skey == 'busType') and (rconf['DBus']['busType'] == 'Cached'):
            config['DBus']['earlyInjection'] = ['Undefined']
    if key == 'IBus':
        if (skey == 'cacheSize') and (rconf['IBus']['busType'] == 'Cached'):
            max_way = rconf['IBus']['cacheSize'] / 512
            config['IBus']['wayCount'] = [i for i in config['IBus']['wayCount'] if i <= max_way]
        elif (skey == 'busType') and (rconf['IBus']['busType'] == 'Simple'):
            config['IBus']['asyncTagMemory'] = ['Undefined']
            config['IBus']['tighlyCoupled'] = ['Undefined']
            config['IBus']['reducedBankWidth'] = ['Undefined']
            config['IBus']['relaxedPcCalculation'] = ['Undefined']
            config['IBus']['twoCycleCache'] = ['Undefined']
            config['IBus']['twoCycleRamInnerMux'] = ['Undefined']
            config['IBus']['memDataWidth'] = ['Undefined']
            config['IBus']['bytePerLine'] = ['Undefined']
            config['IBus']['cacheSize'] = ['Undefined']
            config['IBus']['wayCount'] = ['Undefined']
        elif (skey == 'busType') and (rconf['IBus']['busType'] == 'Cached'):
            config['IBus']['latency'] = ['Undefined']
            config['IBus']['cmdForkOnSecondStage'] = ['Undefined']
            config['IBus']['cmdForkPersistence'] = ['Undefined']
        elif (skey == 'twoCycleRam'):
            InstructAntiOK = ((
                ((not rconf['IBus']['twoCycleRam']) or 
                (rconf['IBus']['wayCount'] == 1)) and 
                (not rconf['IBus']['compressed'])))
            if(not InstructAntiOK):
                config['IBus']['twoCycleCache'] = [False]
    return


def gen_config_from_vector(conf_vec, design_space):
    
    config = design_space

    real_vec = []

    rconfig = {}

    for key in config.keys():
        if type(config[key]) is dict:
            rconfig[key] = {}
            for skey in config[key]:
                # TODO: This may lead to confusion for the algorithm
                selection = conf_vec.pop(0) % len(config[key][skey])
                real_vec.append(selection)
                rconfig[key][skey] = config[key][skey][selection]
                conflict_resolve(config, rconfig, key, skey)
        else:
            # TODO: This may lead to confusion for the algorithm
            selection = conf_vec.pop(0) % len(config[key])
            real_vec.append(selection)
            rconfig[key] = config[key][selection]
            conflict_resolve(config, rconfig, key, '')

    return real_vec, rconfig

def gen_vector_from_config(conf, design_space):

    all_config = design_space
    vec = []
    for key in all_config:
        if type(all_config[key]) is dict:
             for skey in all_config[key]:
                print(key, skey,":", conf[key][skey])
                vec.append(list(all_config[key][skey]).index(conf[key][skey]))
        else:
            print(key,":", conf[key])
            vec.append(list(all_config[key]).index(conf[key]))
    return vec

def get_vector_bounds_from_config(design_space):

    json_dict = design_space
    constr_vec = []

    for key in json_dict:
        if type(json_dict[key]) == dict:
            for skey in json_dict[key]:
                constr_vec.append(len(json_dict[key][skey]))
        else:
            constr_vec.append(len(json_dict[key]))
    
    return constr_vec

def parse_simulation_report(report: str, benchmark: str):

    if benchmark == 'dhrystone':
        # Use dhrystone as benchmark
        benchmark_pat = r"Clock cycles=(\d+)"
    else:
        # Use TF Lite TinyML as benchmark
        benchmark_pat = r"TFLite Regression\(.*\) Passed in (\d+) cycles"

    report_lines = report.splitlines()

    cycle = 0
    num_bench = 0
    for line in report_lines:
        bmatch = re.match(benchmark_pat, line)
        if bmatch:
            cycle += int(bmatch.group(1))
            num_bench += 1

    cycle //= num_bench
    
    return cycle