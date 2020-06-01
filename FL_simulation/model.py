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

class Model(nn.Module):
    """
    The Model class declares a PySyft neural network based on the specifications contained
    inside the structural_definition dictionary.

    Args:
        structural_definition (Dict): For each layer of the network, specify name, input
                                      and output size, activation function, layer_type.

    Attributes:
        layers (list): This is a list of tuples, each containing the layer name and
                       activation function.
    """
    def __init__(self, structural_definition):
        super(Model, self).__init__()

        self.layers = []

        for layer in structural_definition:
            layer_params = structural_definition[layer]
            layer_type = self.__parse_layer_type(layer_params['layer_type'])
            layer_size_mapping = layer_params['layer_size_mapping']
            activation = self.__parse_activation_type(layer_params['activation'])

            setattr(self,
                    layer,
                    layer_type(**layer_size_mapping))

            self.layers.append((layer, activation))

    ###########
    # Helpers #
    ###########

    @staticmethod
    def __parse_layer_type(layer_type):
        """ Detects layer type of a specified layer from configuration

        Args:
            layer_type (str): Layer type to initialise
        Returns:
            Layer definition (Function)
        """
        if layer_type == "linear":
            return nn.Linear
        elif layer_type == 'conv2d':
            return nn.Conv2d
        else:
            raise ValueError("Specified layer type is currently not supported!")


    @staticmethod
    def __parse_activation_type(activation_type):
        """ Detects activation function specified from configuration

        Args:
            activation_type (str): Activation function to use
        Returns:
            Activation definition (Function)
        """
        if activation_type == "sigmoid":
            return th.sigmoid
        elif activation_type == "relu":
            return th.relu
        elif activation_type == "nil":
            return None
        else:
            raise ValueError("Specified activation is currently not supported!")

    ##################
    # Core Functions #
    ##################

    def forward(self, x):
        for layer_activation_tuple in self.layers:
            current_layer =  getattr(self, layer_activation_tuple[0])
            if layer_activation_tuple[1] is None:
                x = current_layer(x)
            else:
                x = layer_activation_tuple[1](current_layer(x))

        return x
