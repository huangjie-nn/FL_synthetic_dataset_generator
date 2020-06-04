import argparse
import json
import numpy as np
import os
from scipy.special import softmax
import data_generator as generator
import math
import json
from collections import defaultdict

def get_perturbation(perturb_mean, perturb_std, n_features, n_parties, n_classes):
	"""
	Obtain the perturbation hyper parameters from params.json.
	Args:
		perturb_mean (list(float)):
			The mean for perturbation matrix for every party
		perturb_std	(list(float)):
			The std for perturbation matrix for each party
		n_features (int):
			Number of features indicated in params.json
		n_classes (int):
			Number of classes indicated in params.json
	Returns:
		perturbation matrix list in the following format
		{
			"0":[[],[],.....],
		 	"1":[[],[],.....],
			 .....
		}
	"""
	perturb_mean = get_stats_from_json(perturb_mean, n_parties, 0.1)
	perturb_std = get_stats_from_json(perturb_std, n_parties, 1)
	perturb_dict = defaultdict()
	for i, (mean, std) in enumerate(zip(perturb_mean, perturb_std)):

		epslon = np.random.normal(
        	loc=mean, scale=std, size=(n_features + 1, n_classes))
		perturb_dict[i] = epslon
	print('perturb dict is  %s' %perturb_dict)
	return perturb_dict

def test_to_format(testset):
	"""
	Convert the generated testset into desired format.
	Args:
		testset (dict):
			The generated testsets from each party.
	Returns:
		The dictionary containing the final testset, without the party infomation
	"""
	aggregated_test = [v for k,v in testset.items()]
	final_test_x = []
	final_test_y = []
	for i in range(len(aggregated_test)):
		final_test_x.extend(aggregated_test[i]['x'].tolist())
		final_test_y.extend(aggregated_test[i]['y'].tolist())

	testset = {'x': final_test_x, 'y': final_test_y}
	return testset

def get_x_stats(data_generator, loc_list):
	"""
	Convert the generated testset into desired format.
	Args:
		testset (dict):
			The generated testsets from each party.
	Returns:
		The dictionary containing the final testset, without the party infomation
	"""
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
	"""
	Given the x_mean list and x_sigma list in params.json, sample n_parties amount of loc for each n_features.
	Args:
		x_mean (list(float)):
			The x_mean defined by user, for each feature. If undefined for feature i, will take 0.
		x_sigma (list(float)):
			The x_sigma defined by user, for each feature. If undefined for feature j, will take 1.
	Returns:
		The n_parties x n_features matrix of loc value.
		Row i corresponds to loc value for every feature for party i.
		Column j corresponds to loc value for every party for feature j.
	"""
	loc = np.zeros((n_parties, n_features))
	x_mean = get_stats_from_json(x_mean, n_features, 0.1)
	x_sigma = get_stats_from_json(x_sigma, n_features, 0.1)
	################
	# sample loc for every party of every feature (vertical sampling)
	# for example:
	# party_0: feature_0, feature_1
	# party_1: feature_0, feature_1
	# the sampling happened below for, for instance, feature_i,
	# is to sample loc for both party_0 and party_1 at the same time with mean x_mean[i], and std of x_sigma[i]
	################
	for i, (mean, sigma) in enumerate(zip(x_mean, x_sigma)):
		loc[:,i] = np.random.normal(loc=mean, scale = sigma, size = n_parties)
	return loc

def get_stats_from_json(stats_list, length, default_value = 0):
	"""
	Given the stats list, desired length of the list, and default_value, trim the list if longer than desired length,
	pad the list with default_value is shorter than desired length.
	Args:
		stats_list (list(float)):
			A list of statistics defined by user that might need to be either trim, or pad, or leave it as it is
		length (int):
			The desired length of the list
		default_value (float):
			Default at 0. The default value that user wanted to pad the list if the given one is shorter than the indicated length
	Returns:
		Stats_list that has been transformed into specific length, either by trimming, or padding with value, or as it is.
	"""
	if stats_list is None:
		stats_list = np.array([default_value]*length)
	elif len(stats_list) < length:
		stats_list = stats_list + [default_value]*(length - len(stats_list))
	elif len(stats_list) > length:
		stats_list = stats_list[:length]
	return stats_list


def get_noises(noises, n_parties):
	"""
	Get noises according to params.json
	Args:
		noises (list(float)):
			A list of noises defined by user
		n_parties (int):
			Number of parties
	Returns:
		Noises list for each party
	"""
	noises = get_stats_from_json(noises, n_parties, 0)
	return noises


def get_weights(weights, n_classes, num_tasks):
	"""
	Get label distribution for training set
	Args:
		weights (list(float)):
			A list of decimal values, with length equals to n_classes or n_classes -1.
		n_classes (int):
			Number of classes
		num_tasks (int):
			Number of parties
	Returns:
		Label distribution (list of decimal value) for each party with length of n_classes. A num_tasks x n_classes matrix.
		Row i indicating the label propotion for party i.
	"""
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

def get_num_samples(data_portion, n_classes, size_ref, num_tasks):
	"""
	Get number of samples for each party
	Args:
		data_portion (list(float)):
			A list of decimal values, with length equals to num_tasks or num_tasks -1.
		n_classes (int):
			Number of classes
		size_ref (int):
			Total number of data sample for all the parties as a whole. Only training set is included.
			Size of the testset is directly indicated as a number for each parties.
		num_tasks (int):
			Number of parties
	Returns:
		Number of samples for each party as a list. The round off will be given to the last party.
	"""
	if data_portion is not None:
		num_samples = []
		portion_list = [i for i in data_portion]

		if len(portion_list) not in [num_tasks, num_tasks - 1]:
			raise ValueError("portion specified but incompatible with number "
							"of classes.")
		elif len(portion_list) == num_tasks - 1:
			if sum(portion_list) >1 :
				raise ValueError("the portions specified does not add up to 1.")
			portion_list = portion_list + [1.0 - sum(portion_list)]

		elif (len(portion_list) == num_tasks) and (sum(portion_list)!=1):
			raise ValueError("portion specified does not add up to 1.")


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
	print('data portion is: %s' %data_portion)
	print('num_samples are: '+str(num_samples))
	return num_samples


def to_format(tasks):
	"""
	Convert the datasets into desired format for writing out into data/ directory
	Args:
		tasks (dict):
			The datasets of all parties.
	Returns:
		Datsets in the following format.
		{
			"0": {
				"x": [[],[],...],
				"y": []
			},
			....
		}
	"""

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
