import argparse
import json
import numpy as np
import os
from scipy.special import softmax
from collections import defaultdict
import data_generator as generator
import math
import json


PROB_CLUSTERS = [1.0]

def main():

	#args = parse_args()


	with open('params.json') as json_file:
		data = json.load(json_file)
	print('Generating dataset')

	num_samples = get_num_samples(
		data['sample_size']['data_potion'], 
		data['meta']['n_classes'], 
		data['sample_size']['total_size'], 
		data['meta']['n_parties'])

	weights = get_weights(
		data['label_distribution'], 
		data['meta']['n_classes'], 
		data['meta']['n_parties'])

	noises = get_noises(
		data['noise'],
		data['meta']['n_parties']
	)

	generator = generator.SyntheticDataset(
		num_classes=data['meta']['n_classes'], 
		prob_clusters=PROB_CLUSTERS, 
		num_dim=data['meta']['n_features'], 
		seed=data['meta']['seed'],
		num_parties = data['meta']['n_parties'],
		model_sigma = data['meta']['model_sigma']
		)
	
	datasets= generator.get_tasks(num_samples, weights)

	# user_data = to_format(datasets)

	# save_json('data/all_data', 'data.json', user_data)
	# print('Done :D')
	# return datasets

def get_noises(noises, n_parties):
	if len(noises) < n_parties:
		for i in range(len(noises),n_parties):
			noises.append[0.0]
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
				weight_list[i,:] = w
			count += 1 

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
		#num_samples = get_num_samples(args.num_tasks, args.size_factor, args.size_low, args.size_high)
	print('number of class: %s' %n_classes)
	print('number of parties: %s' %num_tasks)
	print('data potion is: %s' %data_potion)
	print('num_samples are: '+str(num_samples))


# def to_format(tasks):
# 	user_data = {}
# 	for k, v in tasks.items():
# 		x, y = v['x'].tolist(), v['y'].tolist()
# 		u_id = str(k)

# 		user_data[u_id] = {'x': x, 'y': y}

# 	return user_data



# def save_json(json_dir, json_name, user_data):
# 	if not os.path.exists(json_dir):
# 		os.makedirs(json_dir)
	
# 	with open(os.path.join(json_dir, json_name), 'w') as outfile:
# 		json.dump(user_data, outfile)


# def get_class_sep(num_tasks, class_seps_sigma):
# 	loc = np.random.normal(loc=0, scale=1, size=None)
# 	class_sep = np.random.normal(loc=loc, scale=class_seps_sigma, size = num_tasks)
# 	sep = [math.exp(i) for i in class_sep]
# 	return sep

# # get weights by sample mu list as center for weight sampling at later stage 
# # when label-sigma = 0, the distribution is not even.
# def get_weights(label_sigma, n_classes, num_tasks, imbalance_factor):
# 	loc = np.random.normal(loc=0, scale=1., size=None)
# 	mu = np.random.normal(loc=loc, scale=label_sigma, size=num_tasks)
# 	weights = np.zeros((num_tasks, n_classes))

# 	if label_sigma == 0.0:
# 		print('label_sigma is zero')
# 		w = np.random.normal(loc = mu[0], scale = imbalance_factor, size = n_classes)
# 		for idx in range(weights.shape[0]):
# 			weights[idx, :] = w

# 	else: 
# 		for idx in range(weights.shape[0]):

# 			w = np.random.normal(loc = mu[idx], scale = imbalance_factor, size = n_classes)

# 			weights[idx, :] = w
	
# 	print('weights before softmax: '+str(weights))
# 	scaled = softmax(weights, axis = 1)
# 	print('weights after softmax: '+str(scaled))
# 	return scaled


# def get_num_samples(num_tasks, size_diff, size_low, size_high):

# 	ref = (size_high - size_low)/2
# 	mu_ref = np.random.normal(loc = ref, scale = 1., size = None)

# 	mu_list =[]
# 	while len(mu_list) < num_tasks:
# 		mu = round(np.random.normal(loc = mu_ref, scale = size_diff, size = None))
# 		if size_low <= mu <=size_high:	
# 			mu_list.append(mu)
# 	return mu_list

# def parse_args():
# 	parser = argparse.ArgumentParser()

# 	parser.add_argument(
# 		'-num-tasks',
# 		help='number of parties',
# 		type=int,
# 		required=True)
# 	parser.add_argument(
# 		'-n-classes',
# 		help='number of classes;',
# 		type=int,
# 		required=True)
# 	parser.add_argument(
# 		'-n-features',
# 		help='number of dimensions;',
# 		type=int,
# 		required=True)
# 	parser.add_argument(
# 		'-seed',
# 		help='seed for the random processes;',
# 		type=int,
# 		default=931231,
# 		required=False)

# 	parser.add_argument(
# 		'-label-sigma',
# 		type=float,
# 		help='sigma for label distribution amongst parties',
# 		default=1.,
# 		required=False
# 	)

# 	parser.add_argument(
# 		'-imbalance-factor',
# 		type=float,
# 		help='imbalance factor for labels, apply to all parties',
# 		default=0,
# 		required=False)

# 	parser.add_argument(
# 		'-weights', 
# 		type=float,
# 		nargs='+', 
# 		help='weights for classes', 
# 		required=False)

# 	parser.add_argument(
# 		'-n-clusters-per-class', 
# 		type=int,
# 		default=2,
# 		help='probability for clusters, more values in the list, more diff the dataset', 
# 		required=False)

	# parser.add_argument(
	# 	'-size-factor',
	# 	help='size-diff of the party sample size value',
	# 	type=int,
	# 	default=2,
	# 	required=False)

	# parser.add_argument(
	# 	'-data-potion',
	# 	help='the data portion each client would have',
	# 	type=float,
	# 	nargs='+',
	# 	required=False
	# )

	# parser.add_argument(
	# 	'-size-ref',
	# 	help='the reference quantity of data the user would like to have',
	# 	type=int,
	# 	default=3000,
	# 	required=False
	# )

	# parser.add_argument(
	# 	'-size-low',
	# 	help='The lower bound of the party sample size. Default to be 10',
	# 	type=int,
	# 	default=10,
	# 	required=False)

	# parser.add_argument(
	# 	'-class-seps-sigma',
	# 	help='sigma for class seperations when generating datasets',
	# 	type=float,
	# 	default=1.,
	# 	required=False
	# )

	# parser.add_argument(
	# 	'-size-high',
	# 	help='The upper bound of the party sample size. Default to be 10000',
	# 	type=int,
	# 	default=1000000,
	# 	required=False)

	
	# return parser.parse_args()


if __name__ == '__main__':
	main()
