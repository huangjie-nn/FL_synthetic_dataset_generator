import argparse
import json
import numpy as np
import os
from scipy.special import softmax
import data_generator as generator
import math
import json
from collections import defaultdict
from helper_main import *
import sys
import time

start = time.time()

"""
The main file for generating sythetic datasets that takes in hyper-parameters and produce datasets according
to user specified characteristics. The datasets are being saved into folder named data/.
In total, four files will be generated.
1. data.json includes the data generated. The format are
	{
		"0": {
			"x": [[],[], ......],
			"y": []
		},
		"1": {
			"x": [[],[], ......],
			"y": []
		}
	}
2. test_data.json includes the generated testing data. This refers to the seeding datasets that TTP receives at the beginning. The format are
	{
		"x": [[],[], ......],
		"y": []
	}
3. x_stats.json includes the statistical information of all the features of every party. The format are
	{
		"party_0":{
			"feature_0":{
				"mean": 2 (example value, float),
				"std": 0.2 (example value, float)
			},
			"feature_1":{
				.....
			}
		}
		"party_1":{
			......
		}
	}
4. Q_dict.json includes the model weights of all clients and also global model. The format are
	{
		"0":[[],[],...],
		"1":[......],
		......,
		"global":[......]
	}
"""

base_path = "./data/" + str(sys.argv[1])
with open(base_path + "/params.json") as json_file:
		data = json.load(json_file)

print('Generating dataset')

############################
# getting params
############################

np.random.seed(data['meta']['seed'])


#############
# Obtain number of samples for each party as a list
#############
num_samples = get_num_samples(
	data['sample_size']['data_portion'],
	data['meta']['n_classes'],
	data['sample_size']['total_size'],
	data['meta']['n_parties'])

#############
# Obtain label portion for each classes for each party for training set
# shape: n_parties x n_classes
#############
weights = get_weights(
	data['label_distribution'],
	data['meta']['n_classes'],
	data['meta']['n_parties'])


#############
# Obtain label portion for each classes for each party for testing set
# shape: n_parties x n_classes.
# All label will have exact same weight for testing purpose
#############
test_weights = get_weights(
	None,
	data['meta']['n_classes'],
	data['meta']['n_parties']
)

#############
# Obtain noise for each party as a list
#############
noises = get_noises(
	data['noise']['noise_level'],
	data['meta']['n_parties']
)

loc_list = get_loc_list(
	data['feature_distribution']['x_mean'],
  	data['feature_distribution']['x_sigma'],
 	data['meta']['n_parties'],
	data['meta']['n_features']
)

perturb_list = get_perturbation(
	data['model_perturbation']['mean'],
	data['model_perturbation']['std'],
	data['meta']['n_features'],
	data['meta']['n_parties'],
	data['meta']['n_classes']
)

############################
# generating datasets
############################
g = generator.SyntheticDataset(
	num_classes=data['meta']['n_classes'],
	num_dim=data['meta']['n_features'],
	seed=data['meta']['seed'],
	n_parties = data['meta']['n_parties'],
	x_level_noise = data['noise']['x_level_noise']
	)

datasets, model_weight_dic = g.get_tasks(num_samples, weights, noises, loc_list, perturb_list)

testset, _ = g.get_tasks(
	([data['meta']['testset_size_per_party']]*data['meta']['n_parties']),
	test_weights,
	([0] * data['meta']['n_parties']),
	loc_list,
	perturb_list)

x_stats = get_x_stats(g, loc_list)

############################
# formatting
############################
user_data = to_format(datasets)
test_data = test_to_format(testset)

############################
# saving
############################
save_json(base_path, 'data.json', user_data)
save_json(base_path, 'test_data.json', test_data)
save_json(base_path,'x_stats.json',x_stats)
save_json(base_path,'Q_dict.json', model_weight_dic)

print('data generation done')

end = time.time()
print("TIME FOR CLIENT" + str(sys.argv[1]))
print(end - start)
