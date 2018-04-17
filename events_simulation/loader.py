#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Data loader for the PyTorch framework.
"""

import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.utils.data as data

from parameters import *
from utils import SOG_TOKEN


class TrainingSet(data.Dataset):
    def __init__(self, batch_size):
        """Initialize the training set"""

        print("*** Loading events file. ***")

        events_df = pd.read_csv('../data/football-events/new_events.csv')

        ids_to_df = {key: events_df.loc[value] for key, value in events_df.groupby("id_odsp").groups.items()}
        
        nb_games_training = len(ids_to_df) // 10
        if nb_games_training % batch_size != 0:
            nb_games_training -= nb_games_training % batch_size

        tensors = []
        y_tensors = []
        self.teams = []
        for idd in tqdm(sorted(ids_to_df)[:nb_games_training]):
            df = ids_to_df[idd]

            tensor_df = df.loc[:, '0':]
            home_team = df['home_team'].iloc[0]
            away_team = df['away_team'].iloc[0]
            self.teams.append((home_team, away_team))
            game_tensors = [torch.FloatTensor(SOG_TOKEN)]
            game_y_tensors = []
            for idx, row in tensor_df.iterrows():
                game_tensors.append(torch.FloatTensor(row.values))

                # Find index of 1's
                indices = [i for i, x in enumerate(row.values) if x == 1]
                assert(len(indices) == 2)
                indices[1] -= NB_ALL_EVENTS
                game_y_tensors.append(torch.LongTensor(indices))

            tensors.append(torch.stack(game_tensors, 0))
            y_tensors.append(torch.stack(game_y_tensors))

        # Get sizes of tensors
        max_events = tensor_df.shape[0]
        tensor_size = tensor_df.shape[1]

        stacked_tensors = torch.stack(tensors, 0)
        y_stacked_tensors = torch.stack(y_tensors, 0)

        # First event is only 0's, and we don't have last one
        # TODO: Last one should be "last REAL event", not counting padding
        self.X = stacked_tensors[:, :-1, :]
        #print("X:", self.X)

        self.Y = y_stacked_tensors
        #print("Y:", self.Y)

        # TODO: Cast in LongTensor needed?
        self.Y = self.Y.type(torch.LongTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.teams[index]


class TestSet(data.Dataset):
    def __init__(self, batch_size):
        """Initialize the test set"""

        print("*** Loading events file. ***")

        events_df = pd.read_csv('../data/football-events/new_events.csv')

        ids_to_df = {key: events_df.loc[value] for key, value in events_df.groupby("id_odsp").groups.items()}
        
        nb_games_training = len(ids_to_df) // 10
        if nb_games_training % batch_size != 0:
            nb_games_training -= nb_games_training % batch_size

        nb_games_test = len(ids_to_df) // 10
        if nb_games_test % batch_size != 0:
            nb_games_test -= nb_games_test % batch_size

        tensors = []
        y_tensors = []
        self.teams = []
        for idd in tqdm(sorted(ids_to_df)[nb_games_training:nb_games_training + nb_games_test]):
            df = ids_to_df[idd]

            tensor_df = df.loc[:, '0':]
            home_team = df['home_team'].iloc[0]
            away_team = df['away_team'].iloc[0]
            self.teams.append((home_team, away_team))
            game_tensors = [torch.FloatTensor(SOG_TOKEN)]
            game_y_tensors = []
            for idx, row in tensor_df.iterrows():
                game_tensors.append(torch.FloatTensor(row.values))

                # Find index of 1's
                indices = [i for i, x in enumerate(row.values) if x == 1]
                assert(len(indices) == 2)
                indices[1] -= NB_ALL_EVENTS
                game_y_tensors.append(torch.LongTensor(indices))

            tensors.append(torch.stack(game_tensors, 0))
            y_tensors.append(torch.stack(game_y_tensors))

        # Get sizes of tensors
        max_events = tensor_df.shape[0]
        tensor_size = tensor_df.shape[1]

        stacked_tensors = torch.stack(tensors, 0)
        y_stacked_tensors = torch.stack(y_tensors, 0)

        # First event is only 0's, and we don't have last one
        # TODO: Last one should be "last REAL event", not counting padding
        self.X = stacked_tensors[:, :-1, :]
        #print("X:", self.X)

        self.Y = y_stacked_tensors
        #print("Y:", self.Y)

        # TODO: Cast in LongTensor needed?
        self.Y = self.Y.type(torch.LongTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.teams[index]

