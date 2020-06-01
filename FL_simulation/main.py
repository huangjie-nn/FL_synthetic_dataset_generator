import copy
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from collections import defaultdict
import json

# Libs
import sklearn as skl
from sklearn import preprocessing
import sklearn.datasets as skld
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sbn
import syft as sy
import torch as th
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, tnrange, tqdm_notebook
from tqdm.notebook import trange
from IPython.display import display
from scipy.special import softmax

from contrib_calc import *
from FL_env import *
from synth_data_prep import *

##################
# Configurations #
##################

training_datasets, validation_datasets, testing_datasets = prep_synth_data('../data/all_data/data.json', [0.8, 0.1, 0.1])
testing_dataset = aggregate_testing_datasets(testing_datasets)

# ============
# Hyperparams for FL model.

#============
# Binary Class hyperparams
binary_model_hyperparams = {
    "batch_size": 45,
    "lr": 0.1,
    "decay": 0.01,
    "rounds":2,
    "epochs": 2,
    "criterion": nn.BCELoss,
    "is_condensed": True
}

#============
# Multi Class hyperparams
multiclass_model_hyperparams = {
    "batch_size": 45,
    "lr": 0.01,
    "decay": 0.1,
    "rounds":1,
    "epochs": 1,
    "criterion": nn.CrossEntropyLoss,
    "is_condensed": False
}

binary_model_structure =  {
    '0':{'layer_size_mapping': {"in_features": 5,
                                "out_features": 64},
        "layer_type": 'linear',
        "activation": 'sigmoid'},
    '1':{'layer_size_mapping': {"in_features": 64,
                                "out_features": 1},
        "layer_type": 'linear',
        "activation": 'sigmoid'},
}

# multiclass_model_structure =  {
#     '0':{"in_features": 20,
#         "out_features": 200,
#         "layer_type": 'linear',
#         "activation": 'sigmoid'},
#     '1':{"in_features": 200,
#         "out_features": 4,
#         "layer_type": 'linear',
#         "activation": 'nil'},
# }

# {'in_channels': 10,
#      'out_channels': 10,
#      'kernel_size': 10,
# }

model_hyperparams = binary_model_hyperparams

#============
# Set up Federated Learning environment.
# Produce points to the datasets stored on workers
# and ttp. Also produce pointers to workers and ttp.

(training_pointers,
 validation_pointers,
 testing_pointer,
 workers,
 crypto_provider) = setup_FL_env(
    training_datasets,
    validation_datasets,
    testing_dataset
)

#=============
# Produce individual trainlaoders for each client
# which make use of the full data available to them.
trainloaders = {}
for worker_id in list(training_pointers.keys()):
    train = {worker_id:training_pointers[worker_id]}
    val = {worker_id:validation_pointers[worker_id]}
    #============
    # Convert training datasets into syft dataloaders.
    train_loader, validation_loader, test_loader = convert_to_FL_batches(
        binary_model_hyperparams,
        train,
        val,
        testing_pointer
    )
    trainloaders.update({worker_id : train_loader})

# for key in trainloaders:
#     print(key)
#     for batch_idx, batch in enumerate(trainloaders[key]):
#         print(batch[1].shape)

# #============
# # Commence FL training and return trained global model, as well as
# global and local model states over time.

trained_model, global_states, client_states, scale_coeffs, global_model_state_dicts = perform_FL_training(
    model_hyperparams,
    binary_model_structure,
    trainloaders,
    workers,
    crypto_provider
)

# # trainloaders
# print(scale_coeffs)

client_state_dict = fill_client_state_dict(client_states)
client_alignment_matrix, client_deletion_matrix = contribution_calculation(model_hyperparams, global_states, client_state_dict, testing_dataset, 'Singular', scale_coeffs)
contributions = aggregate_contribution_matrices(model_hyperparams, client_alignment_matrix, client_deletion_matrix)

print(contributions)
