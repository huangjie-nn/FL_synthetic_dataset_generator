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

from model import *

pt_hook = sy.TorchHook(th)

########################
# Simulation Functions #
########################

def connect_to_workers(n_workers):
    """ Simulates the existence of N workers

    Args:
        n_workers (int): No. of virtual workers to simulate
    Returns:
        N virtual workers (list(sy.VirtualWorker))
    """
    return [
        sy.VirtualWorker(
            pt_hook, id=f"worker{i+1}"
        ).clear_objects(
        ) for i in range(n_workers)
    ]

def connect_to_crypto_provider():
    """ Simulates the existence of an arbitor to facilitate
        model generation & client-side utilisation

    Returns:
        Arbiter (i.e. TTP) (sy.VirtualWorker)
    """
    return sy.VirtualWorker(
        pt_hook,
        id="crypto_provider"
    ).clear_objects()

def secret_share(tensor, workers, crypto_provider, precision_fractional):
    """ Transform to fixed precision and secret share a tensor

    Args:
        tensor             (PointerTensor): Pointer to be shared
        workers   (list(sy.VirtualWorker)): Involved workers of the grid
        crypto_provider (sy.VirtualWorker): Arbiter (i.e. TTP) of the grid
        precision_fractional (int): Precision for casting integer ring arithimetic
    """
    return (
        tensor
        .fix_precision(precision_fractional=precision_fractional)
        .share(
            *workers,
            crypto_provider=crypto_provider,
            requires_grad=True
        )
    )

def setup_FL_env(training_datasets, validation_datasets,
                 testing_dataset, is_shared=False):
    """ Sets up a basic federated learning environment using virtual workers,
        with a allocated arbiter (i.e. TTP) to faciliate in model development
        & utilisation, and deploys datasets to their respective workers

    Args:

        training_datasets   (dict(tuple(th.Tensor))): Datasets to be used for training
        validation_datasets (dict(tuple(th.Tensor))): Datasets to be used for validation
        testing_dataset           (tuple(th.Tensor)): Datasets to be used for testing
        is_shared (bool): Toggles if SMPC encryption protocols are active
    Returns:
        training_pointers  (dict(sy.BaseDataset))
        validation_pointer (dict(sy.BaseDataset))
        testing_pointer    (sy.BaseDataset)
        workers            (list(sy.VirtualWorker))
        crypto_provider    (sy.VirtualWorker)
    """
    # Simulate FL computation amongst K worker nodes,
    # where K is the no. of datasets to be federated
    workers = connect_to_workers(n_workers=len(training_datasets))

    # Allow for 1 exchanger/Arbiter (i.e. TTP)
    crypto_provider = connect_to_crypto_provider()
    crypto_provider.clear_objects()

    assert (len(crypto_provider._objects) == 0)

    # Send training & validation datasets to their respective workers
    training_pointers = {}
    validation_pointers = {}
    for w_idx in range(len(workers)):

        # Retrieve & prepare worker for receiving dataset
        curr_worker = workers[w_idx]
        curr_worker.clear_objects()

        assert (len(curr_worker._objects) == 0)

        train_data = training_datasets[w_idx]
        validation_data = validation_datasets[w_idx]

        # Cast dataset into a Tensor & send it to the relevant worker
        train_pointer = sy.BaseDataset(*train_data).send(curr_worker)
        validation_pointer = sy.BaseDataset(*validation_data).send(curr_worker)

        # Store data pointers for subsequent reference
        training_pointers[curr_worker] = train_pointer
        validation_pointers[curr_worker] = validation_pointer

    # 'Me' serves as the client -> test pointer stays with me, but is shared via SMPC
    testing_pointer = sy.BaseDataset(*testing_dataset).send(crypto_provider)

    return training_pointers, validation_pointers, testing_pointer, workers, crypto_provider

def convert_to_FL_batches(model_hyperparams, train_pointers, validation_pointers, test_pointer):
    """ Supplementary function to convert initialised datasets into their
        SGD compatible dataloaders in the context of PySyft's federated learning
        (NOTE: This is based on the assumption that querying database size does
               not break FL abstraction (i.e. not willing to share quantity))
    Args:
        model_hyperparams                      (model_hyperparams): Parameters defining current experiment
        train_pointers      (dict(sy.BaseDataset)): Distributed datasets for training
        validation_pointers (dict(sy.BaseDataset)): Distributed datasets for model calibration
        test_pointer              (sy.BaseDataset): Distributed dataset for verifying performance
    Returns:
        train_loaders     (sy.FederatedDataLoader)
        validation_loader (sy.FederatedDataLoader)
        test_loader       (sy.FederatedDataLoader)
    """

    def construct_FL_loader(data_pointer, **kwargs):
        """ Cast paired data & labels into configured tensor dataloaders
        Args:
            dataset (list(sy.BaseDataset)): A tuple of X features & y labels
            kwargs: Additional parameters to configure PyTorch's Dataloader
        Returns:
            Configured dataloader (th.utils.data.DataLoader)
        """
        federated_dataset = sy.FederatedDataset(data_pointer)

#         print(federated_dataset)

        federated_data_loader = sy.FederatedDataLoader(
            federated_dataset,
            batch_size=(
                model_hyperparams['batch_size']
                if model_hyperparams['batch_size']
                else len(federated_dataset)
            ),
            shuffle=True,
            iter_per_worker=True, # for subsequent parallelization
            **kwargs
        )

        return federated_data_loader


    # Load training pointers into a configured federated dataloader
    train_loader = construct_FL_loader(train_pointers.values())

    # Load validation pointer into a configured federated dataloader
    validation_loader = construct_FL_loader(validation_pointers.values())

    # Load testing dataset into a configured federated dataloader
    test_loader = construct_FL_loader([test_pointer])

    return train_loader, validation_loader, test_loader

def perform_FL_training(model_hyperparams,
                        model_structure,
                        datasets,
                        workers,
                        crypto_provider,
                        optimizer=th.optim.SGD):
    """
    Simulates a PySyft federated learning cycle using PyTorch, in order
    to prove that it can be done conceptually using the PyTorch interface

    Args:
        model_hyperparams (model_hyperparams):
                                            Parameters defining current experiment
        datasets  (sy.FederatedDataLoader):
                                        Distributed training datasets
        workers   (list(sy.VirtualWorker)):
                                        Workers involved in training
        crypto_provider (sy.VirtualWorker):
                                        Arbiter supervising training
        model     (nn.Module):
                            Current PyTorch model to train
        optimizer (th.optim):
                            Optimizer to use
    Returns:
        global_model (nn.Module) : The trained model
        global_states (dict)
                        {timestep: nn.Module}
                        : The record of trained global models at each timestep.
        client_states (dict)
                        {timestep {worker_id: nn.Module}}
                        : The record of trained models for each worker at each
                          timestep.
        scale_coeffs (dict)
                        {worker_id: float}
                        : A dictionary of the update weightings for each worker
                          based on individual dataset size.
    """

    criterion = model_hyperparams['criterion']

    def perform_parallel_training(datasets,
                                  models,
                                  optimizers,
                                  criterions,
                                  epochs):
        """
        Parallelizes training across each distributed dataset (i.e. simulated worker)
        Parallelization here refers to the training of all distributed models per
        epoch.
        NOTE: Current approach does not have early stopping implemented

        Args:
            datasets   (dict(th.utils.data.DataLoader)):
                                                       Distributed training datasets
            models     (list(nn.Module)):
                                        Simulated local models (after distribution)
            optimizers (list(th.optim)):
                                       Simulated local optimizers (after distribution)
            criterions (list(th.nn)):
                                    Simulated local objective function (after distribution)
            epochs (int):
                        No. of epochs to train each local model
        Returns:
            trained local models
        """
        for e in range(epochs):
            for worker in datasets:
#                 print("========================")
#                 print(worker)
                for batch_idx, batch in enumerate(datasets[worker]):
#                     print(batch_idx)
                    data = batch[0]
                    labels = batch[1]
                    '''
                    ========================
                    Each worker trains its own model individually.
                    ========================
                    '''
                    curr_model = models[worker]
                    curr_optimizer = optimizers[worker]
                    curr_criterion = criterions[worker]

                    # Zero gradients to prevent accumulation
                    curr_model.train()
                    curr_optimizer.zero_grad()

                    # Forward Propagation
                    predictions = curr_model(data.float())
#                     print(predictions.shape)
#                     print(labels.shape)

                    if model_hyperparams['is_condensed']:
                        loss = curr_criterion(predictions, labels.float())
                    else:
                        loss = curr_criterion(predictions, labels.long())

                    # Backward propagation
                    loss.backward()
                    curr_optimizer.step()

                    # Update models, optimisers & losses
                    models[worker] = curr_model
                    optimizers[worker] = curr_optimizer
                    criterions[worker] = curr_criterion

                    assert (models[worker] == curr_model and
                            optimizers[worker] == curr_optimizer and
                            criterions[worker] == curr_criterion)

        trained_models = {w: m.send(crypto_provider) for w,m in models.items()}

        return trained_models

    def calculate_global_params(global_model, models, datasets):
        """ Aggregates weights from trained locally trained models after a round.

        Args:
            global_model   (nn.Module): Global model to be trained federatedly
            models   (dict(nn.Module)): Simulated local models (after distribution)
            datasets (dict(th.utils.data.DataLoader)): Distributed training datasets
        Returns:
            Aggregated parameters (OrderedDict)
        """
        param_types = global_model.state_dict().keys()
        model_states = {w: m.state_dict() for w,m in models.items()}

        # Calculate scaling factors for each worker
        scale_coeffs = {w: 1/len(list(datasets.keys())) for w in list(datasets.keys())}

        # PyTorch models can only swap weights of the same structure
        # Hence, aggregate weights while maintaining original layering structure
        aggregated_params = OrderedDict()

        '''
        ======================
        Grab the param_states
        ======================
        '''
        params = {}

        for p_type in param_types:
            #param_states = [th.mul(ms[p_type], sc)
            #                for ms,sc in zip(model_states, scale_coeffs)]
            param_states = [
                th.mul(
                    model_states[w][p_type],
                    scale_coeffs[w]
                ).get().get() for w in workers
            ]

            '''
            ======================
            Grab the param_states
            ======================
            '''
            params.update({p_type : param_states})

            layer_shape = tuple(global_model.state_dict()[p_type].shape)

            '''
            ======================
            Modification made here to allow multiple layers.
            ======================
            '''
            aggregated_params[p_type] = th.zeros(param_states[0].shape, dtype=th.float64)
            for param_state in param_states:
                aggregated_params[p_type] += param_state
            aggregated_params[p_type] = aggregated_params[p_type].view(*layer_shape)

        return aggregated_params, params, scale_coeffs

    # Generate a global model & send it to the TTP

    template_model = Model(model_structure)

    global_model = copy.deepcopy(template_model).send(crypto_provider)

    print("Global model parameters:\n", [p.location for p in list(global_model.parameters())],
          "\nID:\n", [p.id_at_location for p in list(global_model.parameters())],
          "\n Cloning effect on global model:\n", [p.clone() for p in list(global_model.parameters())])

    rounds = 0
    pbar = tqdm(total=model_hyperparams['rounds'], desc='Rounds', leave=True)

    '''
    * Dicts for model and client states

    '''
    global_states = {}
    client_states = {}
    global_model_state_dicts = {}

    client_template = copy.deepcopy(template_model)

    while rounds < model_hyperparams['rounds']:

        local_models = {w: copy.deepcopy(client_template).send(w) for w in workers}

        optimizers = {
            w: optimizer(
                params=model.parameters(),
                lr=model_hyperparams['lr'],
                weight_decay=model_hyperparams['decay']
            ) for w, model in local_models.items()
        }

        criterions = {w: criterion(reduction='mean')
                      for w,m in local_models.items()}

        trained_models = perform_parallel_training(
            datasets,
            local_models,
            optimizers,
            criterions,
            model_hyperparams['epochs']
        )

        aggregated_params, params, scale_coeffs = calculate_global_params(
            global_model,
            trained_models,
            datasets
        )

        '''
        ============================
        * Save states to dictionary

        '''
        global_model_transfer_out = global_model.get()
        global_states.update({rounds : copy.deepcopy(global_model_transfer_out)})
        global_model_state_dicts.update({rounds : global_model_transfer_out.state_dict()})

        client_states.update({rounds + 1 : params})

        # Update weights with aggregated parameters
        global_model_transfer_out.load_state_dict(aggregated_params)
#         model = copy.deepcopy(global_model_transfer_out)
        client_template = copy.deepcopy(global_model_transfer_out)
        global_model = global_model_transfer_out.send(crypto_provider)

        rounds += 1
        pbar.update(1)

    '''
    ============================
    * Save final global state

    '''
    global_model_transfer_out = global_model.get()
    global_states.update({rounds : copy.deepcopy(global_model_transfer_out)})
    global_model = global_model_transfer_out.send(crypto_provider)
    global_model_state_dicts.update({rounds : global_model_transfer_out.state_dict()})
    pbar.close()

    return global_model, global_states, client_states, scale_coeffs, global_model_state_dicts
