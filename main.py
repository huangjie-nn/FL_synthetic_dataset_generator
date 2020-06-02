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

base_path = "./data/" + str(sys.argv[1])
with open(base_path + "/params.json") as json_file:
		data = json.load(json_file)

print('Generating dataset')
np.random.seed(data['meta']['seed'])
num_samples = get_num_samples(
	data['sample_size']['data_portion'],
	data['meta']['n_classes'],
	data['sample_size']['total_size'],
	data['meta']['n_parties'])

weights = get_weights(
	data['label_distribution'],
	data['meta']['n_classes'],
	data['meta']['n_parties'])

test_weights = get_weights(
	None,
	data['meta']['n_classes'],
	data['meta']['n_parties']
)

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

g = generator.SyntheticDataset(
	num_classes=data['meta']['n_classes'],
	num_dim=data['meta']['n_features'],
	seed=data['meta']['seed'],
	n_parties = data['meta']['n_parties'],
	x_level_noise = data['noise']['x_level_noise']
	)

datasets, model_weight_dic = g.get_tasks(num_samples, weights, noises, loc_list, perturb_list)

print('test test %s' %([0] * data['meta']['n_parties']))
testset, _ = g.get_tasks(
	([data['meta']['testset_size_per_party']]*data['meta']['n_parties']),
	test_weights,
	([0] * data['meta']['n_parties']),
	loc_list,
	perturb_list)


user_data = to_format(datasets)
test_data = test_to_format(testset)
x_stats = get_x_stats(g, loc_list)

save_json(base_path, 'data.json', user_data)
save_json(base_path, 'test_data.json', test_data)
save_json(base_path,'x_stats.json',x_stats)
save_json(base_path,'Q_dict.json', model_weight_dic)
