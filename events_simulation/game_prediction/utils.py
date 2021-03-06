#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Utilitary functions.
"""

import time
import numpy as np

import torch
from torch.autograd import Variable

from parameters import *

def build_k_indices(data_and_targets, k_fold, seed=1):
    """Builds k indices for k-fold."""

    num_row = len(data_and_targets)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def train_valid_split_k_fold(train_loader, k_fold, seed):
    """
    Splits the data into training and validation sets based on
    the ratio passed in argument.
    """

    data = []
    targets = []

    # Wrap tensors
    for (d, t) in train_loader:
        data.append(Variable(d))
        targets.append(Variable(t))

    # Create list of k indices
    k_indices = build_k_indices(data, k_fold, seed)

    all_train_data = []
    all_train_targets = []
    all_valid_data = []
    all_valid_targets = []
    for k in range(k_fold):

        # Create the validation fold
        valid_data = [data[i] for i in k_indices[k]]
        valid_targets = [targets[i] for i in k_indices[k]]

        # Create the training folds
        k_indices_train = np.delete(k_indices, k, 0)
        k_indices_train = k_indices_train.flatten()

        train_data = [data[i] for i in k_indices_train]
        train_targets = [targets[i] for i in k_indices_train]

        all_train_data.append(train_data)
        all_train_targets.append(train_targets)
        all_valid_data.append(valid_data)
        all_valid_targets.append(valid_targets)

    return all_train_data, all_train_targets, all_valid_data, all_valid_targets

def train_valid_split(train_loader, ratio, seed):
    """
    Splits the data into training and validation sets based on
    the ratio passed in argument.
    """

    data = []
    targets = []

    # Wrap tensors
    for (d, t) in train_loader:
        data.append(Variable(d))
        targets.append(Variable(t))

    # Create list of k indices
    k_indices = build_k_indices(data, int(round(1/ratio)), seed)

    k = 0

    # Create the validation fold
    valid_data = [data[i] for i in k_indices[k]]
    valid_targets = [targets[i] for i in k_indices[k]]

    # Create the training folds
    k_indices_train = np.delete(k_indices, k, 0)
    k_indices_train = k_indices_train.flatten()

    train_data = [data[i] for i in k_indices_train]
    train_targets = [targets[i] for i in k_indices_train]

    return train_data, train_targets, valid_data, valid_targets

def get_hyperparams_filename(filename, batch_size=None, learning_rate=None, hidden_layer_size1=None, hidden_layer_size2=None, dropout_rate=None):
    if batch_size is None:
        batch_size = CHOSEN_BATCH_SIZE

    if learning_rate is None:
        learning_rate = CHOSEN_LEARNING_RATE

    if hidden_layer_size1 is None:
        hidden_layer_size1 = CHOSEN_HIDDEN_LAYER_SIZES[0]

    if hidden_layer_size2 is None:
        hidden_layer_size2 = CHOSEN_HIDDEN_LAYER_SIZES[1]

    if dropout_rate is None:
        dropout_rate = CHOSEN_DROPOUT_RATE

    tab = filename.split('.')
    name = '.'.join(tab[:-1])
    extension = tab[-1]
    return "%s_%d_%.6f_(%d_%d)_%.1f.%s" % (name, batch_size, learning_rate, hidden_layer_size1, hidden_layer_size2, dropout_rate, extension)

def get_dated_filename(filename):
    tab = filename.split('.')
    name = '.'.join(tab[:-1])
    extension = tab[-1]
    return "%s_%s.%s" % (name, time.strftime("%Y%m%d-%H%M"), extension)

