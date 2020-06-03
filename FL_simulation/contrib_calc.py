# Generic
import copy
import math
import os
import random
import sys
import time
from collections import OrderedDict
from collections import defaultdict

from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import numpy as np
import syft as sy
import torch as th
from scipy.special import softmax

class Contribution_Calculation:
    def __init__(self,
                global_states,
                model_hyperparams,
                client_states,
                testing_dataset,
                scale_coeffs):
        self.global_states = global_states
        self.model_hyperparams = model_hyperparams
        self.client_states = client_states
        self.testing_dataset = testing_dataset
        self.scale_coeffs = scale_coeffs
        self._fill_client_state_dict()

    def _prep_client_state_dict(self):
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
        for timestep in self.client_states.keys():
            client_dict.update({timestep : {}})
            for param_type in self.client_states[timestep].keys():
                for client_idx in range(len(self.client_states[timestep][param_type])):
                    client_dict[timestep].update({client_idx : {}})
                break
        self.client_state_dict = client_dict

    def _fill_client_state_dict(self):
        """ Populates prepared client state dictionary with model states
            for each client at each timestep.
        Args:
            client_states (dict): Client states at each communication round.
        Returns:
            client_state_dict (dict): Complete state dictionary
        """
        self._prep_client_state_dict()
        for timestep in self.client_states.keys():
            for param_type in self.client_states[timestep].keys():
                for client_idx in range(len(self.client_states[timestep][param_type])):
                    self.client_state_dict[timestep][client_idx].update({param_type : self.client_states[timestep][param_type][client_idx]})

    def singular(self,
                 reference_eval,
                 rnd,
                 client_idx):
        """
        This function evaluates the client's contribution to performance metrics
        in isolation from those of other clients.
        Args:
            global_states (dict): Dictionary of global model states.
            client_state_dict (dict): Dictionary of client model states at each timestep.
            reference_eval (float): The accuracy and ROC-AUC score of the baseline model
                                    at timestep rnd.
            rnd (int): The current round
            client_idx (int): The index of the client being evaluated
        Returns:
            Difference between performance of client model and global model.
        """
        single_client_model = copy.deepcopy(self.global_states[rnd - 1])
        single_client_model.load_state_dict(self.client_state_dict[rnd][client_idx])
        client_eval = self.perform_FL_testing(single_client_model)
        print('---------')
        print(f"Improvement of {client_idx}'s update in isolation when applied to global model at round {rnd - 1}")
        print((client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1]))
        return (client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1])

    def aggregate(self,
                  reference_eval,
                  rnd,
                  client_idx,
                  client_subset_size = 1.0):
        """
        This function evaluates the client's contribution to performance metrics
        in isolation from those of other clients.
        Args:
            global_states (dict): Dictionary of global model states.
            client_state_dict (dict): Dictionary of client model states at each timestep.
            reference_eval (float): The accuracy and ROC-AUC score of the baseline model
                                    at timestep rnd.
            rnd (int): The current round
            client_idx (int): The index of the client being evaluated
        Returns:
            Difference between performance of client model and global model.
        """
        aggregate_exclusion_model = copy.deepcopy(self.global_states[rnd - 1])

        dl_dw = self.prep_dl_dw()

        indexed_scale_coeffs = self.index_scale_coeffs_by_integer()
        scaling_soeff_factor = self.produce_excluded_scaling_factor(indexed_scale_coeffs, client_idx)

        num_clients = len(self.client_state_dict[1].keys())
        client_indices = np.random.choice(num_clients, int(np.floor(num_clients * client_subset_size)), replace=False)

        for client in client_indices:
            if client != client_idx:
                for layer_name in self.client_state_dict[rnd][client]:
                    dl_dw[layer_name] += (indexed_scale_coeffs[client]/scaling_soeff_factor) * self.client_state_dict[rnd][client][layer_name]

        aggregate_states = aggregate_exclusion_model.state_dict()
        for param_type in dl_dw:
            aggregate_states[param_type] += dl_dw[param_type]
        aggregate_exclusion_model.load_state_dict(aggregate_states)

        client_eval = self.perform_FL_testing(aggregate_exclusion_model)

        print('---------')
        print(f"Difference in performance of global model at round {rnd - 1} when client {client_idx} is excluded.")
        print((client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1]))

        return (client_eval[0] - reference_eval[0], client_eval[1] - reference_eval[1])

    def contribution_calculation(self, del_method):
        """
        This function calculates the contribution of each client to model
        training using the deletion and alignment methods.
        Args:
            model_hyperparams (dict): Hyperparams used in FL model training.
            global_states (dict): Dictionary of global model states.
            client_state_dict (dict): Dictionary of client model states at each timestep.
            del_method (string): Choice of singular or aggregate deletion method.
            scale_coeffs (OrderedDict): Scaling factors for each client based
                                        on number of datapoints, indexed by
                                        FL worker ID.
        Returns:
            client_alignment_matrix (np.array): Client alignments at each timestep.
            client_deletion_matrix (np.array): Differences in client performance at each timestep.
        """
        self.global_performance = []

        total_num_rounds = self.model_hyperparams['rounds']

        final_global_state = self.global_states[total_num_rounds]
        client_alignment_matrix = np.zeros([len(self.client_state_dict[1].keys()), total_num_rounds], dtype=np.float64)
        client_accuracy_matrix = np.zeros([len(self.client_state_dict[1].keys()), total_num_rounds], dtype=np.float64)
        client_ROC_matrix = np.zeros([len(self.client_state_dict[1].keys()), total_num_rounds], dtype=np.float64)

        del_dict = {'Singular' : self.singular,
                   'Aggregate' : self.aggregate}

        for rnd in range(1, total_num_rounds + 1):
            current_global_state = self.global_states[rnd - 1]

            GRV = self.calculate_GRV(final_global_state,
                                    current_global_state)

            reference_eval = self.perform_FL_testing(current_global_state)

            print('----------')
            print(f'Performance of global model at timestep {rnd-1}')
            print(reference_eval)
            self.global_performance.append(reference_eval)

            for client_idx in self.client_state_dict[rnd].keys():

                #============
                # Calculate alignment of dl/dw with GRV
                dl_dw = self.calculate_dl_dw(current_global_state, rnd, client_idx)
                alignment = self.calculate_alignment(dl_dw, GRV)
                print('---------')
                print(f'Alignment of client {client_idx} with GRV at round {rnd}')
                print(alignment.numpy())
                client_alignment_matrix[client_idx][rnd - 1] = alignment.numpy()
                #============
                # Calculate change in model performance metrics when client contribution is
                # selectively deleted.
                client_eval = del_dict[del_method](reference_eval,
                                                   rnd,
                                                   client_idx)
                client_accuracy_matrix[client_idx][rnd - 1] = (client_eval[0])
                client_ROC_matrix[client_idx][rnd - 1] = (client_eval[1])
        print('===============')
        print('Client alignment matrix')
        print(client_alignment_matrix)
        print('===============')
        print('Client accuracy matrix')
        print(client_accuracy_matrix)
        print('===============')
        print('Client ROC matrix')
        print(client_ROC_matrix)

        self.client_alignment_matrix = client_alignment_matrix
        self.client_accuracy_matrix = client_accuracy_matrix
        self.client_ROC_matrix = client_ROC_matrix
    # def normalize_contribution_matrix(self, mat):
    #     return (mat - np.mean(mat)) / np.std(mat)


    #==================================================
    # EDIT CONTRIB CALC FUNCTIONS HERE
    def _find_min_max(self, matrix):
        min = None
        max = None
        for arr in matrix:
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if min is None:
                min = arr_min
            else:
                if arr_min < min:
                    min = arr_min
            if max is None:
                max = arr_max
            else:
                if arr_max > max:
                    max = arr_max
        return min, max

    def _scale_arr(self, arr, min, max):
        print("=======")
        print(min, max)
        print(arr)
        print((arr - min) / (max - min))
        return (arr - min) / (max - min)

    def aggregate_contribution_matrices(self):
        contributions = defaultdict()

        alignment_min, alignment_max = self._find_min_max(self.client_alignment_matrix)
        accuracy_min, accuracy_max = self._find_min_max(self.client_accuracy_matrix)
        ROC_min, ROC_max = self._find_min_max(self.client_ROC_matrix)
        # print(self.client_alignment_matrix)

        for i in range(self.client_alignment_matrix.shape[0]):
            alignment = self._scale_arr(self.client_alignment_matrix[i],
                                        alignment_min,
                                        alignment_max)

            accuracy = self._scale_arr(self.client_accuracy_matrix[i],
                                       accuracy_min,
                                       accuracy_max)

            ROC = self._scale_arr(self.client_ROC_matrix[i],
                                  ROC_min,
                                  ROC_max)

            alignment_component = np.sum(alignment)
            accuracy_component = np.sum(accuracy)
            ROC_component = np.sum(ROC)
            num_rounds = self.model_hyperparams['rounds']

            aggregate = alignment_component + accuracy_component + ROC_component
            contributions[i] = {"aggregate" : aggregate,
                                "alignment_arr": self.client_alignment_matrix[i],
                                "accuracy_arr": self.client_accuracy_matrix[i],
                                "ROC_arr": self.client_ROC_matrix[i],
                                "alignment_component": alignment_component,
                                "accuracy_component": accuracy_component,
                                "ROC_component": ROC_component,
                                "num_rounds": num_rounds}

        return contributions, self.global_performance

    def perform_FL_testing(self, model):
        """ Obtains predictions given a validation/test dataset upon
            a given PyTorch model.

        Args:
            model   (nn.Module): A PyTorch model
        Returns:
            accuracy score (float)
            roc_auc score  (float)
        """
        X_test = self.testing_dataset[0]
        y_test = self.testing_dataset[1]
        model.eval()
        with th.no_grad():
            predicted_labels = model(X_test.float())
            if self.model_hyperparams['is_condensed']:
                accuracy = accuracy_score(y_test.numpy(),
                                        predicted_labels.round().numpy())
                roc = roc_auc_score(y_test.numpy(),
                                        predicted_labels.numpy())
            else:
                accuracy = accuracy_score(y_test.numpy(),
                                        np.array([np.argmax(i) for i in predicted_labels.numpy()]))
                roc = roc_auc_score(y_test.numpy(),
                                        np.array([softmax(i) for i in predicted_labels.numpy()]), multi_class='ovr')
        return accuracy, roc

    def index_scale_coeffs_by_integer(self):
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
        for i, key in enumerate(self.scale_coeffs.keys()):
            scale_dictionary.update({i : self.scale_coeffs[key]})
        return scale_dictionary

    def produce_excluded_scaling_factor(self, indexed_scale_coeffs, idx):
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

    def prep_dl_dw(self):
        dl_dw = {}
        for layer_name in self.client_state_dict[1][0].keys():
            dl_dw.update({layer_name : th.zeros(self.client_state_dict[1][0][layer_name].shape, dtype=th.float64)})
        return dl_dw

    def calculate_dl_dw(self, global_state, rnd, idx):
        dl_dw = self.prep_dl_dw()
        for layer_name in global_state.state_dict().keys():
            dl_dw[layer_name] = self.client_state_dict[rnd][idx][layer_name] - global_state.state_dict()[layer_name]
        return dl_dw

    def calculate_GRV(self, final_state, current_state):
        diff = final_state
        for layer_name in final_state.state_dict().keys():
            diff.state_dict()[layer_name] -= current_state.state_dict()[layer_name]
        return diff

    def calculate_alignment(self, dl_dw, GRV):
        alignment = 0
        num_layers = 0
        for layer_name in dl_dw:
            alignment += th.dot(th.flatten(GRV.state_dict()[layer_name]), th.flatten(dl_dw[layer_name]))
            num_layers += 1
        return alignment/num_layers
