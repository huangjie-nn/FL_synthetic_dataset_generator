import argparse
import json
import numpy as np
import os
from scipy.special import softmax
import data_generator as generator
import math
import json
from collections import defaultdict




def main():

	#args = parse_args()


	with open('params.json') as json_file:
		data = json.load(json_file)
	print('Generating dataset')
	np.random.seed(data['meta']['seed'])
	num_samples = get_num_samples(
		data['sample_size']['data_potion'],
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

	return datasets, test_data, x_stats, model_weight_dic

def get_perturbation(perturb_mean, perturb_std, n_features, n_parties, n_classes):
	perturb_mean = get_stats_from_json(perturb_mean, n_parties, 0)
	perturb_std = get_stats_from_json(perturb_std, n_parties, 0.01)
	perturb_dict = defaultdict()
	for i, (mean, std) in enumerate(zip(perturb_mean, perturb_std)):

		epslon = np.random.normal(
        	loc=mean, scale=std, size=(n_features + 1, n_classes))
		perturb_dict[i] = epslon
	print('perturb dict is  %s' %perturb_dict)
	return perturb_dict

def test_to_format(testset):
	aggregated_test = [v for k,v in testset.items()]
	final_test_x = []
	final_test_y = []
	for i in range(len(aggregated_test)):
		final_test_x.extend(aggregated_test[i]['x'].tolist())
		final_test_y.extend(aggregated_test[i]['y'].tolist())

	testset = {'x': final_test_x, 'y': final_test_y}
	return testset

def get_x_stats(data_generator, loc_list):
	rlt_dict = defaultdict(dict)
	loc_list = np.array(loc_list)
	for i in range(loc_list.shape[0]):
		for j in range(loc_list.shape[1]):
			rlt_dict[str(i)]['feature_'+str(j)] = {"mean":loc_list[i][j], 
					"std":data_generator.Sigma[j][j]
					}
				
	print('obtained rlt_dict is: %s' %rlt_dict)
	return rlt_dict

def get_loc_list(x_mean, x_sigma, n_parties, n_features):
	loc = np.zeros((n_parties, n_features))
	x_mean = get_stats_from_json(x_mean, n_features, 0)
	x_sigma = get_stats_from_json(x_sigma, n_features, 1)
	for i, (mean, sigma) in enumerate(zip(x_mean, x_sigma)):
		loc[:,i] = np.random.normal(loc=mean, scale = sigma, size = n_parties)
	return loc

def get_stats_from_json(stats_list, length, default_value = 0):
	if stats_list is None:
		stats_list = np.array([default_value]*length)
	elif len(stats_list) < length:
		stats_list = stats_list + [default_value]*(length - len(stats_list))
	elif len(stats_list) > length:
		stats_list = stats_list[:length]
	return stats_list


def get_noises(noises, n_parties):
	if len(noises) < n_parties:
		for i in range(len(noises),n_parties):
			noises.append(0.0)
	elif len(noises) > n_parties:
		noises = noises[:n_parties]
	return noises


def get_weights(weights, n_classes, num_tasks):

	weight_list = np.zeros((num_tasks, n_classes))
	if weights is not None:
		count = 0
		for i, w in enumerate(weights):

			if len(w) not in [n_classes-1, n_classes]:
				raise ValueError("Weights specified but incompatible with number "
							"of classes.")

			if len(w) == n_classes - 1:
				if sum(w) >1 :
					raise ValueError("the weight specified for party %s does not add up to 1." %i)
				w = w + [1.0 - sum(w)]

			count += 1
			weight_list[i,:] = w

		if count < num_tasks:
			for idx in range(count, num_tasks):
				w = [(1/n_classes)]*n_classes
				weight_list[idx,:] = w

	else:
		for idx in range(num_tasks):
			w = [(1/n_classes)]*n_classes
			weight_list[idx,:] = w
	print('label distribution is: %s' %weights)
	print('generated weight list is: %s' %weight_list)
	return weight_list

def get_num_samples(data_potion, n_classes, size_ref, num_tasks):
	# set number of samples for each party
	if data_potion is not None:
		num_samples = []
		portion_list = [i for i in data_potion]

		if len(portion_list) not in [num_tasks, num_tasks - 1]:
			raise ValueError("Potion specified but incompatible with number "
							"of classes.")
		elif len(portion_list) == num_tasks - 1:

			portion_list = portion_list + [1.0 - sum(portion_list)]

		elif (len(portion_list) == num_tasks) and (sum(portion_list)!=1):
			raise ValueError("Potion specified does not add up to 1.")


		for portion in portion_list:
			num = int(size_ref * portion)
			num_samples.append(num)
	else:
		num_samples = [int(size_ref * (1/num_tasks))] * num_tasks
		
	left_over = size_ref - sum(num_samples)
	print('left over is: %s' %left_over)
	num_samples[-1] += left_over
	print('number of class: %s' %n_classes)
	print('number of parties: %s' %num_tasks)
	print('data potion is: %s' %data_potion)
	print('num_samples are: '+str(num_samples))
	return num_samples


def to_format(tasks):
	user_data = {}
	for k, v in tasks.items():
		x, y = v['x'].tolist(), v['y'].tolist()
		u_id = str(k)

		user_data[u_id] = {'x': x, 'y': y}

	return user_data



def save_json(json_dir, json_name, user_data):
	if not os.path.exists(json_dir):
		os.makedirs(json_dir)

	with open(os.path.join(json_dir, json_name), 'w') as outfile:
		json.dump(user_data, outfile)


if __name__ == '__main__':
	main()




