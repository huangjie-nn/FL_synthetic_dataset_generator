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
		data['noise']['noise_level'],
		data['meta']['n_parties']
	)

	B = get_B(
		data['feature_distribution']['B'],
		data['feature_distribution']['x_sigma'],
		data['meta']['n_parties']
	)

	g = generator.SyntheticDataset(
		num_classes=data['meta']['n_classes'], 
		prob_clusters=PROB_CLUSTERS, 
		num_dim=data['meta']['n_features'], 
		seed=data['meta']['seed'],
		n_parties = data['meta']['n_parties'],
		x_level_noise = data['noise']['x_level_noise']
		)
	

	datasets = g.get_tasks(num_samples, weights, noises, B)

	user_data = to_format(datasets)

	save_json('data/all_data', 'data.json', user_data)
	return datasets, B

def get_B(B, x_sigma, n_parties):
	if B is None:
		print("B is not provided, generating B")
		B = np.random.normal(loc=0.0, scale=x_sigma, size=n_parties)
	elif (B is not None) and (len(B) !=n_parties):
		raise ValueError("B is provided but the dimensionality "
			"is incompatible with number of parties") 
	print("B is %s" %B)
	return B
	

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
