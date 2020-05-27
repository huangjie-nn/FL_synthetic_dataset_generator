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

	# set number of samples for each party
	num_samples = get_num_samples(args.num_tasks)

	dataset = generator.SyntheticDataset(
		num_classes=args.num_classes, prob_clusters=PROB_CLUSTERS, num_dim=args.num_dim, seed=args.seed)

	# generate datasets for each party and render them as list
	tasks = [dataset.get_task(s) for s in num_samples]

	# transform the datasets into certain formats preferred
	users, num_samples, user_data = to_leaf_format(tasks)

	# save generated datasets to desired directory
	save_json('data/all_data', 'data.json', users, num_samples, user_data)
	print('Done :D')


def get_num_samples(num_tasks, min_num_samples=5, max_num_samples=1000):
	num_samples = np.random.lognormal(3, 2, (num_tasks)).astype(int)
	num_samples = [min(s + min_num_samples, max_num_samples) for s in num_samples]
	return num_samples


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
		help='number of parties;',
		type=int,
		required=True)
	parser.add_argument(
		'-num-classes',
		help='number of classes;',
		type=int,
		required=True)
	parser.add_argument(
		'-num-dim',
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
		'-num-partition',
		help='The number of partitions the user want to divide the party sample size range. The greater, then more diverse.',
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
