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

        self.loss_function_result = nn.CrossEntropyLoss()

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

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def init_hidden(self, teams):
        #assert(len(teams) == self.batch_size)

        teams_tensor = get_teams_caracteristics(teams)

        return (teams_tensor, teams_tensor)

    def forward(self, input, teams):
        #print("Input:", input)
        lstm_out, _ = self.lstm(input, self.init_hidden(teams))
        #print("LSTM out:", lstm_out)

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
        goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(event_proba, target)
        goals_diff_tensor = goals_home_tensor - goals_away_tensor
        goals_diff_target_tensor = goals_home_target_tensor - goals_away_target_tensor

        goals_tensor = torch.stack([goals_home_tensor, goals_away_tensor], 1)
        goals_target_tensor = torch.stack([goals_home_target_tensor, goals_away_target_tensor], 1)

        games_proba = get_games_proba_from_goals_proba(goals_tensor)
        games_results = get_games_results_from_goals(goals_target_tensor)

        #loss_result_game = self.loss_function_result(games_proba, games_results)

        # Events and time loss functions
        loss_events_during_game = self.loss_function_events(events_during_game, target_events_during_game)
        loss_time_during_game = self.loss_function_time(time_during_game, target_time_during_game)

        # Goals loss functions
        loss_goals_home = self.loss_function_goals_home(goals_home_tensor, goals_home_target_tensor)
        loss_goals_away = self.loss_function_goals_away(goals_away_tensor, goals_away_target_tensor)
        loss_goals_diff = self.loss_function_goals_diff(goals_diff_tensor, goals_diff_target_tensor)

        #total_loss = (loss_result_game + loss_events_during_game + loss_time_during_game) / 3
        total_loss = (loss_events_during_game + loss_time_during_game + 1/3 * (loss_goals_home + loss_goals_away + loss_goals_diff)) / 3

        total_loss.backward()

        self.optimizer.step()

        #return event_proba, time_proba, total_loss.data[0], loss_result_game.data[0], loss_events_during_game.data[0], loss_time_during_game.data[0]
        return event_proba, time_proba, total_loss.data[0], loss_events_during_game.data[0], loss_time_during_game.data[0], loss_goals_home.data[0], loss_goals_away.data[0], loss_goals_diff.data[0]

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
        goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(event_proba, target)
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

    def sample_and_get_loss(self, target, teams, return_goal_proba=False):
        total_event_loss = Variable(torch.zeros(1))
        total_time_loss = Variable(torch.zeros(1))
        total_result_loss = Variable(torch.zeros(1))
        total_goals_home_loss = Variable(torch.zeros(1))
        total_goals_away_loss = Variable(torch.zeros(1))
        total_goals_diff_loss = Variable(torch.zeros(1))

        sampled_events = []
        sampled_times = []
        target_events = []
        target_times = []
        all_goal_home_proba = []
        all_goal_away_proba = []
        for batch_idx in range(target.size(0)):
            current_input = Variable(torch.FloatTensor(SOG_TOKEN)).unsqueeze(0).unsqueeze(0)
            self.hidden = self.init_hidden([teams[batch_idx]])

            sampled_events_in_game = []
            sampled_times_in_game = []
            target_events_in_game = []
            target_times_in_game = []

            goal_home_proba = []
            goal_away_proba = []

            end_of_game_idx = get_end_of_game_idx(target[batch_idx, :, 0])

            game_event_proba = Variable(torch.zeros(end_of_game_idx, NB_ALL_EVENTS))

            event_loss_game = Variable(torch.zeros(1))
            time_loss_game = Variable(torch.zeros(1))
            for event_idx in range(end_of_game_idx):
                #print("Input sample:", current_input)

                output, self.hidden = self.lstm(current_input, self.hidden)

                event_scores = self.hidden2event(output)
                time_scores = self.hidden2time(output)

                event_loss = self.loss_function_events(event_scores.view(1, -1), target[batch_idx, event_idx, 0])
                time_loss = self.loss_function_time(time_scores.view(1, -1), target[batch_idx, event_idx, 1])

                event_loss_game += event_loss
                time_loss_game += time_loss

                event_proba = F.softmax(event_scores, 2)
                time_proba = F.softmax(time_scores, 2)

                game_event_proba[event_idx, :] = event_proba

                generated_event = int(torch.multinomial(event_proba[0, 0], 1)[0])
                generated_time = int(torch.multinomial(time_proba[0, 0], 1)[0])

                sampled_events_in_game.append(generated_event)
                sampled_times_in_game.append(generated_time)
                target_events_in_game.append(target[batch_idx, event_idx, 0].data[0])
                target_times_in_game.append(target[batch_idx, event_idx, 1].data[0])

                goal_home_proba.append(event_proba[0, 0, GOAL_HOME])
                goal_away_proba.append(event_proba[0, 0, GOAL_AWAY])

                #print("event_space:", event_space)

                current_input = Variable(torch.zeros(1, 1, NB_ALL_EVENTS + NB_ALL_TIMES))
                current_input[0, 0, generated_event] = 1
                current_input[0, 0, NB_ALL_EVENTS + generated_time] = 1

            goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(game_event_proba.unsqueeze(0), target[batch_idx, :].unsqueeze(0))
            goals_diff_tensor = goals_home_tensor - goals_away_tensor
            goals_diff_target_tensor = goals_home_target_tensor - goals_away_target_tensor

            goals_tensor = torch.stack([goals_home_tensor, goals_away_tensor], 1)
            goals_target_tensor = torch.stack([goals_home_target_tensor, goals_away_target_tensor], 1)

            games_proba = get_games_proba_from_goals_proba(goals_tensor)
            games_results = get_games_results_from_goals(goals_target_tensor)

            # Goals loss functions
            loss_goals_home = self.loss_function_goals_home(goals_home_tensor, goals_home_target_tensor)
            loss_goals_away = self.loss_function_goals_away(goals_away_tensor, goals_away_target_tensor)
            loss_goals_diff = self.loss_function_goals_diff(goals_diff_tensor, goals_diff_target_tensor)

            #loss_result_game = self.loss_function_result(games_proba, games_results)

            #total_result_loss += loss_result_game
            total_event_loss += event_loss_game / end_of_game_idx
            total_time_loss += time_loss_game / end_of_game_idx
            total_goals_home_loss += loss_goals_home
            total_goals_away_loss += loss_goals_away
            total_goals_diff_loss += loss_goals_diff

            sampled_events.append(sampled_events_in_game)
            sampled_times.append(sampled_times_in_game)
            target_events.append(target_events_in_game)
            target_times.append(target_times_in_game)

            all_goal_home_proba.append(goal_home_proba)
            all_goal_away_proba.append(goal_away_proba)

        total_result_loss /= target.size(0)
        total_event_loss /= target.size(0)
        total_time_loss /= target.size(0)
        total_goals_home_loss /= target.size(0)
        total_goals_away_loss /= target.size(0)
        total_goals_diff_loss /= target.size(0)

        #loss = (total_result_loss + total_event_loss + total_time_loss) / 3
        loss = (total_event_loss + total_time_loss + 1/3 * (total_goals_home_loss + total_goals_away_loss + total_goals_diff_loss)) / 3

        if return_goal_proba:
            #return sampled_events, sampled_times, target_events, target_times, all_goal_home_proba, all_goal_away_proba, loss.data[0], total_result_loss.data[0], total_event_loss.data[0], total_time_loss.data[0]
            return sampled_events, sampled_times, target_events, target_times, all_goal_home_proba, all_goal_away_proba, loss.data[0], total_event_loss.data[0], total_time_loss.data[0], total_goals_home_loss.data[0], total_goals_away_loss.data[0], total_goals_diff_loss.data[0]
        else:
            #return sampled_events, sampled_times, target_events, target_times, loss.data[0], total_result_loss.data[0], total_event_loss.data[0], total_time_loss.data[0]
            return sampled_events, sampled_times, target_events, target_times, loss.data[0], total_event_loss.data[0], total_time_loss.data[0], total_goals_home_loss.data[0], total_goals_away_loss.data[0], total_goals_diff_loss.data[0]
