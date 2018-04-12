#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Recurrent Neural Network implementation (LSTM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from loss import *
from team import *

class LSTMEvents(nn.Module):

    def __init__(self, hidden_dim, event_types_size, time_types_size, num_layers=1, batch_size=1, learning_rate=0.01):
        super(LSTMEvents, self).__init__()

        events_weight = torch.FloatTensor([1]*NB_ALL_EVENTS)
        events_weight[GOAL_HOME] = 1
        events_weight[GOAL_AWAY] = 1
        self.loss_function_events = nn.CrossEntropyLoss(weight=events_weight)
        self.loss_function_time = nn.CrossEntropyLoss()

        self.loss_function_goals_home = nn.MSELoss()
        self.loss_function_goals_away = nn.MSELoss()
        self.loss_function_goals_diff = nn.MSELoss()

        self.hidden_dim = hidden_dim
        self.event_types_size = event_types_size
        self.time_types_size = time_types_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(event_types_size + time_types_size, hidden_dim, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to event space
        self.hidden2event = nn.Linear(hidden_dim, event_types_size)

        # The linear layer that maps from hidden state space to time space
        self.hidden2time = nn.Linear(hidden_dim, time_types_size)

        self.hidden = self.init_hidden()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def init_hidden(self):
        # Before we've done anything, we don't have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def get_hidden_from_teams(self, teams):
        assert(len(teams) == self.batch_size)

        teams_tensor = get_teams_caracteristics(teams)

        #print("Teams tensor:", teams_tensor)

        return (teams_tensor, teams_tensor)

    def forward(self, input, teams):
        #print("Input:", input.size())
        lstm_out, _ = self.lstm(input, self.get_hidden_from_teams(teams))
        #print("LSTM out:", lstm_out.size())

        event_space = self.hidden2event(lstm_out)
        time_space = self.hidden2time(lstm_out)

        #print("Event space:", event_space.size())

        #event_scores = F.softmax(event_space, dim=1)
        #time_scores = F.softmax(time_space, dim=1)

        return event_space, time_space

    def step(self, input, target, teams):
        """Do one training step and return the loss."""

        self.train()
        self.zero_grad()
        #self.hidden = self.get_hidden_from_teams(teams)
        event_scores, time_scores = self.forward(input, teams)

        event_proba = F.softmax(event_scores, 2)
        time_proba = F.softmax(time_scores, 2)

        # Only get events during the games
        events_during_game, target_events_during_game, time_during_game, target_time_during_game = get_during_game_tensors(event_scores, time_scores, target)
       
        # Only get goals during the games
        goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(event_proba, time_proba, target)
        goals_diff_tensor = goals_home_tensor - goals_away_tensor
        goals_diff_target_tensor = goals_home_target_tensor - goals_away_target_tensor

        # Events and time loss functions
        loss_time_during_game = self.loss_function_time(time_during_game, target_time_during_game)
        loss_events_during_game = self.loss_function_events(events_during_game, target_events_during_game)

        # Goals loss functions
        loss_goals_home = self.loss_function_goals_home(goals_home_tensor, goals_home_target_tensor)
        loss_goals_away = self.loss_function_goals_away(goals_away_tensor, goals_away_target_tensor)
        loss_goals_diff = self.loss_function_goals_diff(goals_diff_tensor, goals_diff_target_tensor)

        total_loss = (loss_time_during_game + loss_events_during_game + loss_goals_home + loss_goals_away + loss_goals_diff) / 5

        total_loss.backward()

        self.optimizer.step()

        return event_proba, time_proba, total_loss.data[0]

    def predict(self, input, teams):
        """
        Predict an input using the trained network.
        """

        return self.forward(input, teams)

    def predict_proba(self, input, teams):
        pred_event, pred_time = self.predict(input, teams)
        return F.softmax(pred_event, dim=2), F.softmax(pred_time, dim=2)

    def predict_proba_and_get_loss(self, input, target, teams):
        event_scores, time_scores = self.forward(input, teams)

        # Get probabilities
        event_proba = F.softmax(event_scores, 2)
        time_proba = F.softmax(time_scores, 2)

        # Separate events from time
        target_events = target[:, :, 0]
        target_time = target[:, :, 1]

        # Only get events during the games
        events_during_game, target_events_during_game, time_during_game, target_time_during_game = get_during_game_tensors(event_scores, time_scores, target)
       
        # Only get goals during the games
        goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(event_proba, time_proba, target)
        goals_diff_tensor = goals_home_tensor - goals_away_tensor
        goals_diff_target_tensor = goals_home_target_tensor - goals_away_target_tensor

        # Events and time loss functions
        loss_time_during_game = self.loss_function_time(time_during_game, target_time_during_game)
        loss_events_during_game = self.loss_function_events(events_during_game, target_events_during_game)

        # Goals loss functions
        loss_goals_home = self.loss_function_goals_home(goals_home_tensor, goals_home_target_tensor)
        loss_goals_away = self.loss_function_goals_away(goals_away_tensor, goals_away_target_tensor)
        loss_goals_diff = self.loss_function_goals_diff(goals_diff_tensor, goals_diff_target_tensor)

        total_loss = (loss_time_during_game + loss_events_during_game + loss_goals_home + loss_goals_away + loss_goals_diff) / 5

        return event_proba, time_proba, total_loss.data[0]
