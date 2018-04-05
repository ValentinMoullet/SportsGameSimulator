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


class TrainingSet(data.Dataset):
    def __init__(self):
        """Initialize the training set"""

        print("*** Loading events file. ***")

        events_df = pd.read_csv('../data/football-events/new_events.csv')

        ids_to_df = {key: events_df.loc[value] for key, value in events_df.groupby("id_odsp").groups.items()}
        
        nb_games_training = len(ids_to_df) // 100

        tensors = []
        y_tensors = []
        game = 0
        for idd, df in tqdm(ids_to_df.items()):
            if game >= nb_games_training:
                break

            tensor_df = df.loc[:, '0':]
            game_tensors = []
            game_y_tensors = []
            for idx, row in tensor_df.iterrows():
                game_tensors.append(torch.FloatTensor(row.values))

                # Find index of 1's
                indices = [i for i, x in enumerate(row.values) if x == 1]
                assert(len(indices) == 2)
                indices[1] -= (2 * NB_EVENT_TYPES + 2)
                game_y_tensors.append(torch.LongTensor(indices))

            tensors.append(torch.stack(game_tensors, 0))
            y_tensors.append(torch.stack(game_y_tensors))

            game += 1

        # Get sizes of tensors
        max_events = tensor_df.shape[0]
        tensor_size = tensor_df.shape[1]

        stacked_tensors = torch.stack(tensors, 0)
        y_stacked_tensors = torch.stack(y_tensors, 0)

        # First event is only 0's, and we don't have last one
        # TODO: Last one should be "last REAL event", not counting padding
        self.X = torch.cat([torch.zeros(nb_games_training, 1, tensor_size), stacked_tensors[:, :max_events-1, :]], 1)
        #print("X:", self.X)

        self.Y = y_stacked_tensors
        #print("Y:", self.Y)

        # TODO: Cast in LongTensor needed?
        self.Y = self.Y.type(torch.LongTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class TestSet(data.Dataset):
    def __init__(self, teams_to_idx, game_info_df, msk):
        """Initialize the test set"""

        print("*** Loading test set. ***")

        # Take test set only
        game_info_df = game_info_df[~msk].reset_index(drop=True)
        nb_games_test = game_info_df.shape[0]

        # Create one-hot vectors as X, and -1, 0 or 1 as Y
        self.X_home = torch.zeros(nb_games_test, len(teams_to_idx))
        self.X_away = torch.zeros(nb_games_test, len(teams_to_idx))
        self.X = torch.zeros(nb_games_test, 2*len(teams_to_idx))
        self.Y = torch.zeros(nb_games_test)
        for idx, row in game_info_df.iterrows():
            home_team = row['ht']
            away_team = row['at']
            home_score = int(row['fthg'])
            away_score = int(row['ftag'])

            self.X_home[idx, teams_to_idx[home_team]] = 1
            self.X_away[idx, teams_to_idx[away_team]] = 1
            self.X[idx, :] = torch.cat([self.X_home[idx, :], self.X_away[idx, :]])

            if home_score > away_score:
                self.Y[idx] = 0
            elif away_score > home_score:
                self.Y[idx] = 1
            else:
                self.Y[idx] = 2

        # Cast to LongTensor for CrossEntropyLoss later
        self.Y = self.Y.type(torch.LongTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

