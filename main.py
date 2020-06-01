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

	g = generator.SyntheticDataset(
		num_classes=data['meta']['n_classes'], 
		prob_clusters=PROB_CLUSTERS, 
		num_dim=data['meta']['n_features'], 
		seed=data['meta']['seed'],
		x_sigma = data['feature_distribution']['x_sigma'],
		n_parties = data['meta']['n_parties']
		)
	
	datasets= g.get_tasks(num_samples, weights, noises)

	user_data = to_format(datasets)

	save_json('data/all_data', 'data.json', user_data)
	return datasets

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
		#num_samples = get_num_samples(args.num_tasks, args.size_factor, args.size_low, args.size_high)
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
