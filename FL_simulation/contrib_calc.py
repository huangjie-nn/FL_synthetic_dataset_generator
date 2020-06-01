# Generic
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

def perform_FL_testing(dataset, model, model_hyperparams):
    """ Obtains predictions given a validation/test dataset upon
        a given PyTorch model.

    Args:
        model   (nn.Module): A PyTorch model
    Returns:
        accuracy score (float)
        roc_auc score  (float)
    """
    X_test = dataset[0]
    y_test = dataset[1]
#     print(X_test)
    model.eval()
    with th.no_grad():
        predicted_labels = model(X_test.float())
#         print(predicted_labels.shape)
#         print(y_test.shape)
        if model_hyperparams['is_condensed']:
            accuracy = accuracy_score(y_test.numpy(), predicted_labels.round().numpy())
            roc = roc_auc_score(y_test.numpy(), predicted_labels.numpy())
        else:
            accuracy = accuracy_score(y_test.numpy(), np.array([np.argmax(i) for i in predicted_labels.numpy()]))
            roc = roc_auc_score(y_test.numpy(), np.array([softmax(i) for i in predicted_labels.numpy()]), multi_class='ovr')
    return accuracy, roc

def prep_client_state_dict(client_states):
    """ Takes in the client states dictionary returned by training function.
        Returns dictionary initialized with empty dictionaries for each client
        at each timestep.

    Args:
        client_states (dict): Client states at each communication round.
    Returns:
        client_dict (dict): Dictionary initialized to be filled with actual model state
                            tensors.
    """
    client_dict = {}
    for timestep in client_states.keys():
        client_dict.update({timestep : {}})
        for param_type in client_states[timestep].keys():
            for client_idx in range(len(client_states[timestep][param_type])):
                client_dict[timestep].update({client_idx : {}})
            break
    return client_dict

def fill_client_state_dict(client_states):
    """ Populates prepared client state dictionary with model states
        for each client at each timestep.

    Args:
        client_states (dict): Client states at each communication round.
    Returns:
        client_state_dict (dict): Complete state dictionary
    """
    client_state_dict = prep_client_state_dict(client_states)
    for timestep in client_states.keys():
        for param_type in client_states[timestep].keys():
            for client_idx in range(len(client_states[timestep][param_type])):
                client_state_dict[timestep][client_idx].update({param_type : client_states[timestep][param_type][client_idx]})
    return client_state_dict

def index_scale_coeffs_by_integer(scale_coeffs):
    """ scale_coeffs are indexed by FL worker id. Since CC functions
        below rely on integer indexing of clients, need to reindex
        keys while keeping values constant.

    Args:
        scale_coeffs (OrderedDict): Scaling factors for each client based
                                    on number of datapoints, indexed by
                                    FL worker ID.
    Returns:
        scale_dictionary (dict): Identical to scale_coeffs, except indexed
                                 by integers 0, 1, 2, ... etc.
    """
    scale_dictionary = {}
    for i, key in enumerate(scale_coeffs.keys()):
        scale_dictionary.update({i : scale_coeffs[key]})
    return scale_dictionary

def produce_excluded_scaling_factor(indexed_scale_coeffs, idx):
    """ In aggregation of client subsets, scale_coeffs need to be recalculated.
        For example. Three clients. Each contributes 0.33 based on their dataset
        size. I exclude one. When I update global model, I should weight the
        remaining updates by 0.5 instead of 0.33. This function produces a scale
        factor which will divide the coefficients prior to aggregation.

    Args:
        indexed_scale_coeffs (OrderedDict): Output of above function.
    Returns:
        coeffs_scale_factor (float): The sum of scale_coeffs for non-excluded clients.
    """
    coeffs_scale_factor = 0
    for i in indexed_scale_coeffs:
        if i != idx:
            coeffs_scale_factor += indexed_scale_coeffs[i]
    return coeffs_scale_factor

def prep_dl_dw(client_state_dict):
    dl_dw = {}
    for layer_name in client_state_dict[1][0].keys():
        dl_dw.update({layer_name : th.zeros(client_state_dict[1][0][layer_name].shape, dtype=th.float64)})
    return dl_dw

def calculate_dl_dw(client_state_dict, global_state, rnd, idx):
    dl_dw = prep_dl_dw(client_state_dict)
    for layer_name in global_state.state_dict().keys():
        dl_dw[layer_name] = client_state_dict[rnd][idx][layer_name] - global_state.state_dict()[layer_name]
    return dl_dw

def calculate_GRV(final_state, current_state):
    diff = final_state
    for layer_name in final_state.state_dict().keys():
        diff.state_dict()[layer_name] -= current_state.state_dict()[layer_name]
    return diff

def calculate_alignment(dl_dw, GRV):
    alignment = 0
    num_layers = 0
    for layer_name in dl_dw:
        alignment += th.dot(th.flatten(GRV.state_dict()[layer_name]), th.flatten(dl_dw[layer_name]))
        num_layers += 1
    return alignment/num_layers

def singular(global_states,
             client_state_dict,
             testing_dataset,
             reference_eval,
             rnd,
             client_idx,
             scale_coeffs,
             model_hyperparams,
             client_subset_size = 1.0):
    """
    This function evaluates the client's contribution to performance metrics
    in isolation from those of other clients.

    Args:
        global_states (dict): Dictionary of global model states.
        client_state_dict (dict): Dictionary of client model states at each timestep.
        testing_dataset (th.tensor): The testing dataset
        reference_eval (float): The accuracy and ROC-AUC score of the baseline model
                                at timestep rnd.
        rnd (int): The current round
        client_idx (int): The index of the client being evaluated
        scale_coeffs (OrderedDict): Scaling factors for each client based
                                    on number of datapoints, indexed by
                                    FL worker ID.
    Returns:
        Difference between performance of client model and global model.
    """
    single_client_model = copy.deepcopy(global_states[rnd - 1])
    single_client_model.load_state_dict(client_state_dict[rnd][client_idx])

    client_eval = perform_FL_testing(testing_dataset,
                                     single_client_model,
                                     model_hyperparams)
    print('---------')
    print(f"Improvement of {client_idx}'s update in isolation when applied to global model at round {rnd - 1}")
    print((client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1]))
    return (client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1])

def aggregate(global_states,
              client_state_dict,
              testing_dataset,
              reference_eval,
              rnd,
              client_idx,
              scale_coeffs,
              model_hyperparams,
              client_subset_size = 1.0):
    """
    This function evaluates the client's contribution to performance metrics
    in isolation from those of other clients.

    Args:
        global_states (dict): Dictionary of global model states.
        client_state_dict (dict): Dictionary of client model states at each timestep.
        testing_dataset (th.tensor): The testing dataset
        reference_eval (float): The accuracy and ROC-AUC score of the baseline model
                                at timestep rnd.
        rnd (int): The current round
        client_idx (int): The index of the client being evaluated
        scale_coeffs (OrderedDict): Scaling factors for each client based
                                    on number of datapoints, indexed by
                                    FL worker ID.
    Returns:
        Difference between performance of client model and global model.
    """
    aggregate_exclusion_model = copy.deepcopy(global_states[rnd - 1])

    dl_dw = prep_dl_dw(client_state_dict)

    indexed_scale_coeffs = index_scale_coeffs_by_integer(scale_coeffs)
    scaling_soeff_factor = produce_excluded_scaling_factor(indexed_scale_coeffs, client_idx)

    num_clients = len(client_state_dict[1].keys())
    client_indices = np.random.choice(num_clients, int(np.floor(num_clients * client_subset_size)), replace=False)

    for client in client_indices:

        if client != client_idx:
            for layer_name in client_state_dict[rnd][client]:
                dl_dw[layer_name] += (indexed_scale_coeffs[client]/scaling_soeff_factor) * client_state_dict[rnd][client][layer_name]

    aggregate_states = aggregate_exclusion_model.state_dict()
    for param_type in dl_dw:
        aggregate_states[param_type] += dl_dw[param_type]
    aggregate_exclusion_model.load_state_dict(aggregate_states)

    client_eval = perform_FL_testing(testing_dataset,
                                     aggregate_exclusion_model,
                                     model_hyperparams)
    print('---------')
    print(f"Difference in performance of global model at round {rnd - 1} when client {client_idx} is excluded.")
    print((client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1]))

    return (client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1])

def contribution_calculation(model_hyperparams, global_states, client_state_dict, testing_dataset, del_method, scale_coeffs):
    """
    This function calculates the contribution of each client to model
    training using the deletion and alignment methods.

    Args:
        model_hyperparams (dict): Hyperparams used in FL model training.
        global_states (dict): Dictionary of global model states.
        client_state_dict (dict): Dictionary of client model states at each timestep.
        testing_dataset (th.tensor): The testing dataset.
        del_method (string): Choice of singular or aggregate deletion method.
        scale_coeffs (OrderedDict): Scaling factors for each client based
                                    on number of datapoints, indexed by
                                    FL worker ID.
    Returns:
        client_alignment_matrix (np.array): Client alignments at each timestep.
        client_deletion_matrix (np.array): Differences in client performance at each timestep.
    """
    total_num_rounds = model_hyperparams['rounds']

    final_global_state = global_states[total_num_rounds]
    client_alignment_matrix = np.zeros([len(client_state_dict[1].keys()), total_num_rounds], dtype=np.float64)
    client_deletion_matrix = np.zeros([len(client_state_dict[1].keys()), total_num_rounds], dtype=np.float64)

    del_dict = {'Singular' : singular,
               'Aggregate' : aggregate}

    for rnd in range(1, total_num_rounds + 1):
        current_global_state = global_states[rnd - 1]

        GRV = calculate_GRV(final_global_state, current_global_state)

        reference_eval = perform_FL_testing(testing_dataset, current_global_state, model_hyperparams)

        print('----------')
        print(f'Performance of global model at timestep {rnd-1}')
        print(reference_eval)

        for client_idx in client_state_dict[rnd].keys():

#             ============
#             Calculate alignment of dl/dw with GRV
#             print(client_state_dict[rnd][client_idx])
            dl_dw = calculate_dl_dw(client_state_dict, current_global_state, rnd, client_idx)
            alignment = calculate_alignment(dl_dw, GRV)
            print('---------')
            print(f'Alignment of client {client_idx} with GRV at round {rnd}')
            print(alignment.numpy())
            client_alignment_matrix[client_idx][rnd - 1] = alignment.numpy()

            #============
            # Calculate change in model performance metrics when client contribution is
            # selectively deleted.
            client_eval = del_dict[del_method](global_states,
                                               client_state_dict,
                                               testing_dataset,
                                               reference_eval,
                                               rnd,
                                               client_idx,
                                               scale_coeffs,
                                               model_hyperparams)
            client_deletion_matrix[client_idx][rnd - 1] = (client_eval[0] + client_eval[1])

    print('===============')
    print('Client alignment matrix')
    print(client_alignment_matrix)
    print('===============')
    print('Client deletion matrix')
    print(client_deletion_matrix)

    return client_alignment_matrix, client_deletion_matrix

def normalize_contribution_matrix(mat):
    return (mat - np.mean(mat)) / np.std(mat)

def aggregate_contribution_matrices(arguments, align_mat, del_mat):
#     align_mat = normalize_contribution_matrix(align_mat)
#     del_mat = normalize_contribution_matrix(del_mat)

    contributions = defaultdict()

    for i in range(align_mat.shape[0]):
        contributions[i] = np.sum(align_mat[i]) + np.sum(del_mat[i]) / arguments['rounds']
#         print(np.sum(align_mat[i]) + np.sum(del_mat[i]) / arguments['rounds'])
        print(np.sum(align_mat[i]) + np.sum(del_mat[i]) / arguments['rounds'])
    return contributions
