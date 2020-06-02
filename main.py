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

data_quantity_distributions = [ [0.2, 0.2, 0.2, 0.2, 0.2] ]
data_label_distributions = [ [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]] ]
noise_level_settings = [ [0.1, 0.2, 0.4, 0.5, 0.6] ]
x_level_noise_settings = [0, 1]

def generate_params():



    data = {"meta": {"n_parties": 5,
                    "n_classes": 2,
                    "n_features": 5,
                    "seed":1232323232,
                    "testset_size_per_party":30},

            "feature_distribution":{"x_mean":[],
                                    "x_sigma":[]},

            "sample_size":{"data_portion": data_quantity_distributions[0],
                            "total_size":3000},

            "label_distribution":data_label_distributions[0],

            "noise":{"noise_level":noise_level_settings[0],
                    "x_level_noise": x_level_noise_settings[1] },

            "model_perturbation":{"mean":[],
                                "std":[]}
        }

    return data

data = generate_params()

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

save_json('data/all_data', 'data.json', user_data)
save_json('data/all_data', 'test_data.json', test_data)
save_json('data/all_data','x_stats.json',x_stats)
save_json('data/all_data','Q_dict.json', model_weight_dic)
