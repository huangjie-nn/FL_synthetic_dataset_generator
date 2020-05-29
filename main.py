import argparse
import json
import numpy as np
import os
from scipy.special import softmax
from collections import defaultdict
import data_generator as generator
import math


def main():
	args = parse_args()

	print('Generating dataset')

	# set number of samples for each party
	num_samples = get_num_samples(args.num_tasks, args.size_factor, args.size_low, args.size_high)
	print('num_samples are: '+str(num_samples))

	weights = get_weights(args.label_sigma, args.n_classes, args.num_tasks, args.imbalance_factor)

	seps = get_class_sep(args.num_tasks, args.class_seps_sigma)

	Datasets_generator = generator.SyntheticDataGenerator(
		seed=args.seed,
		n_features=args.n_features, 
		n_classes=args.n_classes, 
		n_clusters_per_class=args.n_clusters_per_class
	)

	datasets = {}
	for idx, (s, w, sep) in enumerate(zip(num_samples, weights, seps)):
		dataset = Datasets_generator.make_dataset(s, w, sep) 
		datasets[str(idx)] = dataset

	user_data = to_format(datasets)

	save_json('data/all_data', 'data.json', user_data)
	print('Done :D')
	return datasets


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


def get_class_sep(num_tasks, class_seps_sigma):
	loc = np.random.normal(loc=0, scale=1, size=None)
	class_sep = np.random.normal(loc=loc, scale=class_seps_sigma, size = num_tasks)
	sep = [math.exp(i) for i in class_sep]
	return sep

# get weights by sample mu list as center for weight sampling at later stage 
# when label-sigma = 0, the distribution is not even.
def get_weights(label_sigma, n_classes, num_tasks, imbalance_factor):
	loc = np.random.normal(loc=0, scale=1., size=None)
	mu = np.random.normal(loc=loc, scale=label_sigma, size=num_tasks)
	weights = np.zeros((num_tasks, n_classes))

	if label_sigma == 0.0:
		print('label_sigma is zero')
		w = np.random.normal(loc = mu[0], scale = imbalance_factor, size = n_classes)
		for idx in range(weights.shape[0]):
			weights[idx, :] = w

	else: 
		for idx in range(weights.shape[0]):

			w = np.random.normal(loc = mu[idx], scale = imbalance_factor, size = n_classes)

			weights[idx, :] = w
	
	print('weights before softmax: '+str(weights))
	scaled = softmax(weights, axis = 1)
	print('weights after softmax: '+str(scaled))
	return scaled


def get_num_samples(num_tasks, size_diff, size_low, size_high):

	ref = (size_high - size_low)/2
	mu_ref = np.random.normal(loc = ref, scale = 1., size = None)

	mu_list =[]
	while len(mu_list) < num_tasks:
		mu = round(np.random.normal(loc = mu_ref, scale = size_diff, size = None))
		if size_low <= mu <=size_high:	
			mu_list.append(mu)
	return mu_list

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-num-tasks',
		help='number of parties',
		type=int,
		required=True)
	parser.add_argument(
		'-n-classes',
		help='number of classes;',
		type=int,
		required=True)
	parser.add_argument(
		'-n-features',
		help='number of dimensions;',
		type=int,
		required=True)
	parser.add_argument(
		'-seed',
		help='seed for the random processes;',
		type=int,
		default=931231,
		required=False)

	parser.add_argument(
		'-label-sigma',
		type=float,
		help='sigma for label distribution amongst parties',
		default=1.,
		required=False
	)

	parser.add_argument(
		'-imbalance-factor',
		type=float,
		help='imbalance factor for labels, apply to all parties',
		default=0,
		required=False)

	parser.add_argument(
		'-weights', 
		type=float,
		nargs='+', 
		help='weights for classes', 
		required=False)

	parser.add_argument(
		'-n-clusters-per-class', 
		type=int,
		default=2,
		help='probability for clusters, more values in the list, more diff the dataset', 
		required=False)

	parser.add_argument(
		'-size-factor',
		help='size-diff of the party sample size value',
		type=int,
		default=2,
		required=False)

	parser.add_argument(
		'-size-low',
		help='The lower bound of the party sample size. Default to be 10',
		type=int,
		default=10,
		required=False)

	parser.add_argument(
		'-class-seps-sigma',
		help='sigma for class seperations when generating datasets',
		type=float,
		default=1.,
		required=False
	)

	parser.add_argument(
		'-size-high',
		help='The upper bound of the party sample size. Default to be 10000',
		type=int,
		default=1000000,
		required=False)

	
	return parser.parse_args()


if __name__ == '__main__':
	main()
