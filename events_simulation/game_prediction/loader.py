#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Data loader for the PyTorch framework.
"""

import numpy as np
import pandas as pd
import glob, random

import torch
import torch.utils.data as data

from preprocessing import *
from utils import *
from parameters import *


class TrainingSet(data.Dataset):
    def __init__(self, teams_to_idx, game_info_df, msk):
        """Initialize the training set"""

        print("*** Loading training set. ***")

        game_info_df = game_info_df[msk].reset_index(drop=True)
        nb_games_training = game_info_df.shape[0]

        # Create one-hot vectors as X, and 0, 1 or 2 as Y
        self.X_home = torch.zeros(nb_games_training, len(teams_to_idx))
        self.X_away = torch.zeros(nb_games_training, len(teams_to_idx))
        self.X = torch.zeros(nb_games_training, 2*len(teams_to_idx))
        self.Y = torch.zeros(nb_games_training)
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


class TestSet(data.Dataset):
    def __init__(self, teams_to_idx, game_info_df, msk):
        """Initialize the test set"""

        print("*** Loading test set. ***")

        # Take test set only
        game_info_df = game_info_df[~msk].reset_index(drop=True)
        nb_games_test = game_info_df.shape[0]

        # Create one-hot vectors as X, and 0, 1 or 2 as Y
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

class BookmakersPred(data.Dataset):
    def __init__(self, league='F1'):
        """Initialize the test set"""

        print("*** Loading bookmakers predictions. ***")

        game_info_df = pd.read_csv('../../data/football-events/ginf.csv')
        game_info_df = game_info_df[game_info_df['league'] == league]
        game_info_df = game_info_df.reset_index(drop=True)
        nb_games = game_info_df.shape[0]

        # Create one-hot vectors as X, and -1, 0 or 1 as Y
        self.pred = torch.zeros(nb_games, 3)
        self.target = torch.zeros(nb_games)
        for idx, row in game_info_df.iterrows():
            home_team = row['ht']
            away_team = row['at']
            home_score = int(row['fthg'])
            away_score = int(row['ftag'])
            odd_home = float(row['odd_h'])
            odd_away = float(row['odd_a'])
            odd_draw = float(row['odd_d'])

            self.pred[idx, :] = torch.Tensor([1/odd_home, 1/odd_away, 1/odd_draw])
            if home_score > away_score:
                self.target[idx] = 0
            elif away_score > home_score:
                self.target[idx] = 1
            else:
                self.target[idx] = 2

        # Cast to LongTensor for CrossEntropyLoss later
        self.target = self.target.type(torch.LongTensor)

    def __len__(self):
        return len(self.pred)

    def __getitem__(self, index):
        return self.pred[index], self.target[index]
