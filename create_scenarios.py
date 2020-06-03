import os
import shutil
import json

with open("./scenarios.json") as json_file:
		scenario_params = json.load(json_file)

data_quantity_distributions = scenario_params['data_quantity_distributions']
data_label_distributions = scenario_params['data_label_distributions']
noise_level_settings = scenario_params['noise_level_settings']
x_level_noise_settings = scenario_params['x_level_noise_settings']

def generate_params():
    scenarios = []
    for i in data_quantity_distributions:
        for j in data_label_distributions:
            for k in noise_level_settings:
                for l in x_level_noise_settings:
                    scenarios.append({"data_portion": i,
                                    "label_distribution": j,
                                    "noise_level": k,
                                    "x_level_noise": l})
    # print(scenarios)
    for i in range(len(scenarios)):
        path = './data/' + str(i)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=False, onerror=errorRemoveReadonly)
        os.mkdir(path)
        current_scenario = scenarios[i]
        data = {"meta": {"n_parties": 5,
                        "n_classes": 2,
                        "n_features": 5,
                        "seed":1232323232,
                        "testset_size_per_party":30},
                "feature_distribution":{"x_mean":[],
                                        "x_sigma":[]},
                "sample_size":{"data_portion": current_scenario["data_portion"],
                                "total_size":3000},
                "label_distribution":current_scenario["label_distribution"],
                "noise":{"noise_level":current_scenario["noise_level"],
                        "x_level_noise": current_scenario["x_level_noise"]},
                "model_perturbation":{"mean":[],
                                    "std":[]}}
        with open(path + '/params.json', 'w') as outfile:
            json.dump(data, outfile)

def errorRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        # change the file to be readable,writable,executable: 0777
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # retry
        func(path)
    else:
        raise

generate_params()
