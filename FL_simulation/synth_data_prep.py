import json
import syft as sy
import torch as th
import random

def prep_synth_data(path):
    training_datasets = {}
    with open(path) as json_file:
        data = json.load(json_file)
        for client_idx in data:
            num_data = len(data[client_idx]['x'])
            x_and_y = list(zip(data[client_idx]['x'],
                               data[client_idx]['y']))
            random.shuffle(x_and_y)
            x, y = zip(*x_and_y)
            training_datasets.update({int(client_idx) : (th.tensor(x),
                                                         th.tensor(y).view(-1, 1))})
    return training_datasets

def prep_synth_test_data(path):
    testing_dataset = None
    with open(path) as json_file:
        data = json.load(json_file)
        x_and_y = list(zip(data['x'],
                           data['y']))
        random.shuffle(x_and_y)
        x, y = zip(*x_and_y)
        return (th.tensor(x), th.tensor(y).view(-1, 1))
