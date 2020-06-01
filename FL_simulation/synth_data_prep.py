import json
import syft as sy
import torch as th
import random

def get_split_indices(length, proportions):
    train_end = round(length * proportions[0])
    val_end = train_end + round(length * proportions[1])
    test_end = val_end + round(length * proportions[2])
    return train_end, val_end, test_end

def prep_synth_data(path, split_proportions):
    training_datasets = {}
    validation_datasets = {}
    testing_datasets = {}
    with open(path) as json_file:
        data = json.load(json_file)
        for client_idx in data:
            num_data = len(data[client_idx]['x'])

            train_idx, val_idx, test_idx = get_split_indices(num_data, split_proportions)

            x_and_y = list(zip(data[client_idx]['x'],
                               data[client_idx]['y']))
            random.shuffle(x_and_y)
            x, y = zip(*x_and_y)
            training_datasets.update({int(client_idx) : (th.tensor(x[0:train_idx]),
                                                         th.tensor(y[0:train_idx]).view(-1, 1))})
            validation_datasets.update({int(client_idx) : (th.tensor(x[train_idx:val_idx]),
                                                         th.tensor(y[train_idx:val_idx]).view(-1, 1))})
            testing_datasets.update({int(client_idx) : (th.tensor(x[val_idx:test_idx]),
                                                         th.tensor(y[val_idx:test_idx]).view(-1, 1))})
    return training_datasets, validation_datasets, testing_datasets

def aggregate_testing_datasets(testing_datasets):
    output = None
    for client in testing_datasets:
        if output is None:
            output = (testing_datasets[client][0],
                      testing_datasets[client][1])
        else:
            output = (th.cat((output[0],
                             testing_datasets[client][0]),
                             0),
                      th.cat((output[1],
                             testing_datasets[client][1]),
                             0))

    return (output[0], output[1].view(-1, 1))
