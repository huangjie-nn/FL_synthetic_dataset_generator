import argparse
import json
import numpy as np
import os

import data_generator as generator


PROB_CLUSTERS = [1.0]


def main():
	args = parse_args()
	np.random.seed(args.seed)

	print('Generating dataset')

	# print("number of parties are: "+str(args.num_tasks))
	# print("number of classes are: "+str(args.num_classes))
	# print("number of dimensions are: "+str(args.num_dim))
	# print("number of partitions are: "+str(args.num_partition))
	# print("size lower bound is: "+str(args.size_low))
	# print("size higher bound is "+str(args.size_high))
	# set number of samples for each party
	num_samples = get_num_samples(args.num_tasks, args.sigma, args.size_low, args.size_high)
	
	# dataset = generator.SyntheticDataset(
	# 	num_classes=args.num_classes, prob_clusters=PROB_CLUSTERS, num_dim=args.num_dim, seed=args.seed)

	# # generate datasets for each party and render them as list
	# tasks = [dataset.get_task(s) for s in num_samples]

	# # transform the datasets into certain formats preferred
	# users, num_samples, user_data = to_leaf_format(tasks)

	# # save generated datasets to desired directory
	# save_json('data/all_data', 'data.json', users, num_samples, user_data)
	# print('Done :D')




def get_num_samples(num_tasks, sigma, size_low, size_high):

	ref = (size_high - size_low)/2
	mu_ref = np.random.normal(loc = ref, scale = 1., size = None)
	print('mu_ref is: '+str(mu_ref))


	mu_list =[]
	while len(mu_list) < num_tasks:
		mu = round(np.random.normal(loc = mu_ref, scale = sigma, size = None))
		if size_low <= mu <=size_high:	
			mu_list.append(mu)

	print('mu_list is: '+str(mu_list))




def to_leaf_format(tasks):
	users, num_samples, user_data = [], [], {}
	
	for i, t in enumerate(tasks):
		x, y = t['x'].tolist(), t['y'].tolist()
		u_id = str(i)

		users.append(u_id)
		num_samples.append(len(y))
		user_data[u_id] = {'x': x, 'y': y}

	return users, num_samples, user_data


def save_json(json_dir, json_name, users, num_samples, user_data):
	if not os.path.exists(json_dir):
		os.makedirs(json_dir)
	
	json_file = {
		'users': users,
		'num_samples': num_samples,
		'user_data': user_data,
	}
	
	with open(os.path.join(json_dir, json_name), 'w') as outfile:
		json.dump(json_file, outfile)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-num-tasks',
		help='number of parties',
		type=int,
		required=True)
	# parser.add_argument(
	# 	'-num-classes',
	# 	help='number of classes;',
	# 	type=int,
	# 	required=True)
	# parser.add_argument(
	# 	'-num-dim',
	# 	help='number of dimensions;',
	# 	type=int,
	# 	required=True)
	parser.add_argument(
		'-seed',
		help='seed for the random processes;',
		type=int,
		default=931231,
		required=False)



	parser.add_argument(
		'-sigma',
		help='Sigma of which the party sample size value',
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
		'-size-high',
		help='The upper bound of the party sample size. Default to be 10000',
		type=int,
		default=10000,
		required=False)

	
	return parser.parse_args()


if __name__ == '__main__':
	main()
