#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Utilitary functions.
"""

import numpy as np

import torch
from torch.autograd import Variable


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

