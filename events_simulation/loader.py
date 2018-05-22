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

        training_ids_df = pd.read_csv('../data/football-events/training_ids.csv')
        training_df = events_df[events_df['id_odsp'].isin(training_ids_df['training_id'].values)]

        ids_to_df = {key: training_df.loc[value] for key, value in training_df.groupby("id_odsp").groups.items()}
        
        nb_games_training = len(ids_to_df)
        if nb_games_training % batch_size != 0:
            nb_games_training -= nb_games_training % batch_size

        ids_to_df = {k: ids_to_df[k] for k in list(ids_to_df)[:nb_games_training]}
        
        tensors = []
        y_tensors = []
        self.teams = []
        for idd in tqdm(sorted(ids_to_df)):
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

        test_ids_df = pd.read_csv('../data/football-events/test_ids.csv')
        test_df = events_df[events_df['id_odsp'].isin(test_ids_df['test_id'].values)]

        ids_to_df = {key: test_df.loc[value] for key, value in test_df.groupby("id_odsp").groups.items()}
        
        '''
        nb_games_training = len(ids_to_df) // 10
        if nb_games_training % batch_size != 0:
            nb_games_training -= nb_games_training % batch_size
        '''

        nb_games_test = len(ids_to_df)
        if nb_games_test % batch_size != 0:
            nb_games_test -= nb_games_test % batch_size

        ids_to_df = {k: ids_to_df[k] for k in list(ids_to_df)[:nb_games_test]}

        tensors = []
        y_tensors = []
        self.teams = []
        for idd in tqdm(sorted(ids_to_df)):
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


class TrainingSetFirstHalf(data.Dataset):
    def __init__(self, batch_size):
        """Initialize the training set"""

        print("*** Loading events file. ***")

        events_df = pd.read_csv('../data/football-events/new_events.csv')

        training_ids_df = pd.read_csv('../data/football-events/training_ids.csv')
        training_df = events_df[events_df['id_odsp'].isin(training_ids_df['training_id'].values)]

        ids_to_df = {key: training_df.loc[value] for key, value in training_df.groupby("id_odsp").groups.items()}
        
        nb_games_training = len(ids_to_df)
        if nb_games_training % batch_size != 0:
            nb_games_training -= nb_games_training % batch_size

        ids_to_df = {k: ids_to_df[k] for k in list(ids_to_df)[:nb_games_training]}
        
        tensors = []
        y_values = []
        self.teams = []
        for idd in tqdm(sorted(ids_to_df)):
            df = ids_to_df[idd]

            tensor_df = df.loc[:, '0':]
            home_team = df['home_team'].iloc[0]
            away_team = df['away_team'].iloc[0]
            self.teams.append((home_team, away_team))
            game_tensors = [torch.FloatTensor(SOG_TOKEN)]
            game_y_tensors = []
            current_time = 1
            home_goals = [0, 0]
            away_goals = [0, 0]
            current_half = 0
            for idx, row in tensor_df.iterrows():
                if current_time > 45 and current_half == 0:
                    current_half = 1

                # Find index of 1's
                event, time = [i for i, x in enumerate(row.values) if x == 1]
                time -= NB_ALL_EVENTS
                if event == GOAL_HOME:
                    home_goals[current_half] += 1
                elif event == GOAL_AWAY:
                    away_goals[current_half] += 1
 
                if time == DIFF_TIME_THAN_PREV:
                    current_time += 1

            if home_goals[0] > away_goals[0]:
                res = [1, 0, 0]
            elif home_goals[0] < away_goals[0]:
                res = [0, 1, 0]
            else:
                res = [0, 0, 1]

            tensors.append(torch.FloatTensor(res))

            if home_goals[1] > away_goals[1]:
                res = 0
            elif home_goals[1] < away_goals[1]:
                res = 1
            else:
                res = 2

            y_values.append(res)

        stacked_tensors = torch.stack(tensors, 0)

        self.X = stacked_tensors
        self.Y = torch.LongTensor(y_values)       

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.teams[index]

class TestSetFirstHalf(data.Dataset):
    def __init__(self, batch_size):
        """Initialize the training set"""

        print("*** Loading events file. ***")

        events_df = pd.read_csv('../data/football-events/new_events.csv')

        test_ids_df = pd.read_csv('../data/football-events/test_ids.csv')
        test_df = events_df[events_df['id_odsp'].isin(test_ids_df['test_id'].values)]

        ids_to_df = {key: test_df.loc[value] for key, value in test_df.groupby("id_odsp").groups.items()}
        
        nb_games_test = len(ids_to_df)
        if nb_games_test % batch_size != 0:
            nb_games_test -= nb_games_test % batch_size

        ids_to_df = {k: ids_to_df[k] for k in list(ids_to_df)[:nb_games_test]}
        
        tensors = []
        y_values = []
        self.teams = []
        for idd in tqdm(sorted(ids_to_df)):
            df = ids_to_df[idd]

            tensor_df = df.loc[:, '0':]
            home_team = df['home_team'].iloc[0]
            away_team = df['away_team'].iloc[0]
            self.teams.append((home_team, away_team))
            game_tensors = [torch.FloatTensor(SOG_TOKEN)]
            game_y_tensors = []
            current_time = 1
            home_goals = [0, 0]
            away_goals = [0, 0]
            current_half = 0
            for idx, row in tensor_df.iterrows():
                if current_time > 45 and current_half == 0:
                    current_half = 1

                # Find index of 1's
                event, time = [i for i, x in enumerate(row.values) if x == 1]
                time -= NB_ALL_EVENTS
                if event == GOAL_HOME:
                    home_goals[current_half] += 1
                elif event == GOAL_AWAY:
                    away_goals[current_half] += 1
 
                if time == DIFF_TIME_THAN_PREV:
                    current_time += 1

            if home_goals[0] > away_goals[0]:
                res = [1, 0, 0]
            elif home_goals[0] < away_goals[0]:
                res = [0, 1, 0]
            else:
                res = [0, 0, 1]

            tensors.append(torch.FloatTensor(res))

            if home_goals[1] > away_goals[1]:
                res = 0
            elif home_goals[1] < away_goals[1]:
                res = 1
            else:
                res = 2

            y_values.append(res)

        stacked_tensors = torch.stack(tensors, 0)

        self.X = stacked_tensors
        self.Y = torch.LongTensor(y_values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.teams[index]

