#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Recurrent Neural Network implementation (LSTM)
"""

import glob, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.beta import Beta
from torch.autograd import Variable

from utils import *
from loss import *
from team import *


def load_latest_model():
    model = LSTMEvents(hidden_dim=40, event_types_size=NB_ALL_EVENTS, time_types_size=NB_ALL_TIMES, num_layers=1, batch_size=BATCH_SIZE, learning_rate=0.01)
    all_saved_models = glob.glob("%s/*.pt" % MODELS_DIR)
    latest_model_file = max(all_saved_models, key=os.path.getmtime)
    model.load_state_dict(torch.load(latest_model_file))

    return model

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

        self.lstm = nn.LSTM(event_types_size + time_types_size + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

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

        teams_tensor = get_teams_caracteristics(teams)

        teams_input = teams_tensor.squeeze(0).unsqueeze(1).repeat(1, 208, 1)

        input_with_prior = torch.cat([input, teams_input], 2)
        #print(input_with_prior)

        lstm_out, _ = self.lstm(input_with_prior, self.init_hidden(teams))
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
        events_during_game, target_events_during_game, time_during_game, target_time_during_game, end_game_indices = get_during_game_tensors(event_scores, time_scores, target, return_end_game_idx=True)

        # Only get goals during the games
        goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(event_proba, target)

        goals_tensor = torch.stack([goals_home_tensor, goals_away_tensor], 1)
        goals_target_tensor = torch.stack([goals_home_target_tensor, goals_away_target_tensor], 1)

        '''
        games_proba = get_games_proba_from_goals_proba(goals_tensor)
        # Compute accuracy
        accuracy = 0
        for batch_idx in range(target.size(0)):
            accuracy += games_proba[batch_idx, games_results[batch_idx]]

        accuracy /= target.size(0)

        # Cross entropy loss for result, but don't use it in backwards
        loss_result_game = self.loss_function_result(games_proba, games_results)
        '''

        accuracy = torch.tensor(0)
        loss_result_game = torch.tensor(0)

        '''
        predicted_results = get_games_results_from_goals(goals_tensor)
        games_results = get_games_results_from_goals(goals_target_tensor)

        diff = predicted_results - games_results
        accuracy = (diff.numel() - diff.nonzero().size(0)) / target.size(0)
        accuracy = torch.tensor(accuracy)
        '''

        # Events and time loss functions
        loss_events_during_game = self.loss_function_events(events_during_game, target_events_during_game)
        loss_time_during_game = self.loss_function_time(time_during_game, target_time_during_game)

        #print(time_during_game)
        #print(time_during_game.size())
        #print(target.size(0))

        time_proba_during_game = F.softmax(time_during_game, 1)

        # Compute loss for forcing not having too much events at the same minute
        alphas = 4.0
        betas = 6.53242321
        beta_distr = Beta(alphas, betas)
        log_prob = beta_distr.log_prob(time_proba_during_game[:, SAME_TIME_THAN_PREV])
        same_minute_event_loss = -torch.mean(log_prob)

        '''
        same_minute_event_loss = 0
        current_idx = 0
        for end_game_idx in end_game_indices:
            log_prob = beta_distr.log_prob(torch.mean(time_proba_during_game[current_idx:current_idx+end_game_idx, SAME_TIME_THAN_PREV]))
            game_loss = -log_prob
            same_minute_event_loss += game_loss

            current_idx = end_game_idx

        same_minute_event_loss /= target.size(0)
        '''

        #print("time_proba_during_game:", time_proba_during_game)

        #print("same_minute_event_loss:", same_minute_event_loss)

        '''
        # Goals loss functions
        loss_goals_home = self.loss_function_goals_home(goals_home_tensor, goals_home_target_tensor)
        loss_goals_away = self.loss_function_goals_away(goals_away_tensor, goals_away_target_tensor)
        loss_goals_diff = self.loss_function_goals_diff(goals_diff_tensor, goals_diff_target_tensor)
        '''

        total_loss = (loss_events_during_game + loss_time_during_game + 0.25 * same_minute_event_loss) / 2.25
        #total_loss = (loss_events_during_game + loss_time_during_game + 1/3 * (loss_goals_home + loss_goals_away + loss_goals_diff)) / 3

        total_loss.backward()

        self.optimizer.step()

        return event_proba, time_proba, total_loss.data.item(), loss_events_during_game.data.item(), loss_time_during_game.data.item(), same_minute_event_loss.item(), loss_result_game.data.item(), accuracy.item()
        #return event_proba, time_proba, total_loss.data.item(), loss_events_during_game.data.item(), loss_time_during_game.data.item(), loss_result_game.data.item(), accuracy.item()
        #return event_proba, time_proba, total_loss.data[0], loss_events_during_game.data[0], loss_time_during_game.data[0], loss_goals_home.data[0], loss_goals_away.data[0], loss_goals_diff.data[0]

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

        goals_tensor = torch.stack([goals_home_tensor, goals_away_tensor], 1)
        goals_target_tensor = torch.stack([goals_home_target_tensor, goals_away_target_tensor], 1)

        games_proba = get_games_proba_from_goals_proba(goals_tensor)
        games_results = get_games_results_from_goals(goals_target_tensor)
    
        # Cross entropy loss for result, but don't use it in backwards
        loss_result_game = self.loss_function_result(games_proba, games_results)

        # Events and time loss functions
        loss_time_during_game = self.loss_function_time(time_during_game, target_time_during_game)
        loss_events_during_game = self.loss_function_events(events_during_game, target_events_during_game)

        # Goals loss functions
        #loss_goals_home = self.loss_function_goals_home(goals_home_tensor, goals_home_target_tensor)
        #loss_goals_away = self.loss_function_goals_away(goals_away_tensor, goals_away_target_tensor)
        #loss_goals_diff = self.loss_function_goals_diff(goals_diff_tensor, goals_diff_target_tensor)

        #total_loss = (loss_time_during_game + loss_events_during_game + loss_goals_home + loss_goals_away + loss_goals_diff) / 5
        total_loss = (loss_time_during_game + loss_events_during_game) / 2

        return event_proba, time_proba, total_loss.data[0], loss_events_during_game.data[0], loss_time_during_game.data[0], loss_result_game.data[0]

    def sample_and_get_loss(self, target, teams, return_proba=False):
        total_event_loss = Variable(torch.zeros(1))
        total_time_loss = Variable(torch.zeros(1))
        total_result_loss = Variable(torch.zeros(1))
        total_same_minute_event_loss = Variable(torch.zeros(1))
        total_same_minute_proba_game = Variable(torch.zeros(1))
        total_accuracy = 0

        total_goals_home_loss = Variable(torch.zeros(1))
        total_goals_away_loss = Variable(torch.zeros(1))
        total_goals_diff_loss = Variable(torch.zeros(1))

        sampled_events = []
        sampled_times = []
        target_events = []
        target_times = []
        
        all_proba = []

        for batch_idx in range(target.size(0)):

            end_of_game_idx = get_end_of_game_idx(target[batch_idx, :, 0])

            accuracies = []
            results_losses = []
            results = torch.FloatTensor([0, 0, 0])
            # Sample multiple times
            for _ in range(NB_GAMES_TO_SAMPLE):
                #current_input = Variable(torch.FloatTensor(SOG_TOKEN)).unsqueeze(0).unsqueeze(0)
                current_input = Variable(torch.zeros(1, 1, NB_ALL_EVENTS + NB_ALL_TIMES))
                current_input[0, 0, SOG_TOKEN] = 1
                current_input[0, 0, NB_ALL_EVENTS + GAME_NOT_RUNNING_TIME] = 1

                self.hidden = self.init_hidden([teams[batch_idx]])

                teams_tensor = get_teams_caracteristics([teams[batch_idx]])
                teams_input = teams_tensor.squeeze(0).unsqueeze(1)

                sampled_events_in_game = []
                sampled_times_in_game = []
                target_events_in_game = []
                target_times_in_game = []

                proba = []

                game_event_proba = Variable(torch.zeros((end_of_game_idx, NB_ALL_EVENTS)))

                event_loss_game = Variable(torch.zeros(1))
                time_loss_game = Variable(torch.zeros(1))
                same_minute_event_loss_game = Variable(torch.zeros(1))
                same_minute_proba_game = Variable(torch.zeros(1))
                for event_idx in range(end_of_game_idx):
                    #print("Input sample:", current_input)

                    #print("Teams input:", teams_input)
                    #print("current input:", current_input)

                    input_with_prior = torch.cat([current_input, teams_input], 2)
                    output, self.hidden = self.lstm(input_with_prior, self.hidden)

                    event_scores = self.hidden2event(output)
                    time_scores = self.hidden2time(output)

                    event_loss = self.loss_function_events(event_scores.view(1, -1), target[batch_idx, event_idx, 0].view(1))
                    time_loss = self.loss_function_time(time_scores.view(1, -1), target[batch_idx, event_idx, 1].view(1))

                    event_loss_game += event_loss
                    time_loss_game += time_loss

                    event_proba = F.softmax(event_scores, 2)
                    time_proba = F.softmax(time_scores, 2)

                    # Increase total proba
                    #same_minute_proba_game += time_proba[0, 0, SAME_TIME_THAN_PREV]

                    alphas = 4.0
                    betas = 6.53242321
                    beta_distr = Beta(alphas, betas)
                    log_prob = beta_distr.log_prob(time_proba[0, 0, SAME_TIME_THAN_PREV])
                    same_minute_event_loss_game += -log_prob

                    game_event_proba[event_idx, :] = event_proba

                    generated_event = int(torch.multinomial(event_proba[0, 0], 1)[0])
                    generated_time = int(torch.multinomial(time_proba[0, 0], 1)[0])

                    # Force different time if generating NO_EVENT
                    if generated_event == NO_EVENT:
                        generated_time = DIFF_TIME_THAN_PREV

                    sampled_events_in_game.append(generated_event)
                    sampled_times_in_game.append(generated_time)
                    target_events_in_game.append(target[batch_idx, event_idx, 0].data.item())
                    target_times_in_game.append(target[batch_idx, event_idx, 1].data.item())

                    # Store probabilities of event to happen
                    proba.append([])
                    for event_nb in range(NB_ALL_EVENTS):
                        proba[-1].append(event_proba[0, 0, event_nb])

                    #print("event_space:", event_space)

                    current_input = Variable(torch.zeros(1, 1, NB_ALL_EVENTS + NB_ALL_TIMES))
                    current_input[0, 0, generated_event] = 1
                    current_input[0, 0, NB_ALL_EVENTS + generated_time] = 1

                goals_home_tensor, goals_home_target_tensor, goals_away_tensor, goals_away_target_tensor = get_during_game_goals(game_event_proba.unsqueeze(0), target[batch_idx, :].unsqueeze(0))

                goals_tensor = torch.stack([goals_home_tensor, goals_away_tensor], 1)
                goals_target_tensor = torch.stack([goals_home_target_tensor, goals_away_target_tensor], 1)

                #games_proba = get_games_proba_from_goals_proba(goals_tensor)
                predicted_results = get_games_results_from_goals(goals_tensor)
                games_results = get_games_results_from_goals(goals_target_tensor)

                # Count sampled goals for both teams
                goal_home = sampled_events_in_game.count(GOAL_HOME)
                goal_away = sampled_events_in_game.count(GOAL_AWAY)
                if goal_home > goal_away:
                    sampled_res = 0
                elif goal_home < goal_away:
                    sampled_res = 1
                else:
                    sampled_res = 2

                results[sampled_res] += 1

                #loss_result_game = self.loss_function_result(games_proba, games_results)
                #accuracy = games_proba[0][games_results.item()]

                #results_losses.append(loss_result_game.item())
                #accuracies.append(accuracy.item())

            total_event_loss += event_loss_game.item() / end_of_game_idx
            total_time_loss += time_loss_game.item() / end_of_game_idx
            #total_same_minute_event_loss += same_minute_event_loss_game / end_of_game_idx
            #total_result_loss += np.mean(results_losses)
            #total_accuracy += np.mean(accuracies)

            results /= NB_GAMES_TO_SAMPLE
            total_accuracy += results[games_results.item()]
            total_result_loss += self.loss_function_result(results.unsqueeze(0), games_results)

            same_minute_proba_game /= end_of_game_idx

            # Compute same minute event loss
            '''
            alphas = 4.0
            betas = 6.53242321
            beta_distr = Beta(alphas, betas)
            log_prob = beta_distr.log_prob(same_minute_proba_game)
            same_minute_loss_game = -log_prob
            total_same_minute_event_loss += same_minute_loss_game
            '''

            total_same_minute_event_loss += same_minute_event_loss_game / end_of_game_idx

            #total_same_minute_proba_game += same_minute_proba_game

            '''
            total_goals_home_loss += loss_goals_home
            total_goals_away_loss += loss_goals_away
            total_goals_diff_loss += loss_goals_diff
            '''

            sampled_events.append(sampled_events_in_game)
            sampled_times.append(sampled_times_in_game)
            target_events.append(target_events_in_game)
            target_times.append(target_times_in_game)

            all_proba.append(proba)

        '''
        total_same_minute_proba_game /= target.size(0)
        alphas = 4.0
        betas = 6.53242321
        beta_distr = Beta(alphas, betas)
        log_prob = beta_distr.log_prob(total_same_minute_proba_game)
        '''

        total_result_loss /= target.size(0)
        total_event_loss /= target.size(0)
        total_time_loss /= target.size(0)
        total_same_minute_event_loss /= target.size(0)
        #total_same_minute_event_loss = -log_prob
        total_accuracy /= target.size(0)
        total_goals_home_loss /= target.size(0)
        total_goals_away_loss /= target.size(0)
        total_goals_diff_loss /= target.size(0)

        #print("total_same_minute_event_loss:", total_same_minute_event_loss)

        loss = (total_event_loss + total_time_loss + 0.25 * total_same_minute_event_loss) / 2.25
        #loss = (total_event_loss + total_time_loss + 1/3 * (total_goals_home_loss + total_goals_away_loss + total_goals_diff_loss)) / 3

        if return_proba:
            return sampled_events, sampled_times, target_events, target_times, all_proba, loss.data[0], total_event_loss.data[0], total_time_loss.data[0], total_same_minute_event_loss.item(), total_result_loss.data[0], total_accuracy.item()
            #return sampled_events, sampled_times, target_events, target_times, all_goal_home_proba, all_goal_away_proba, loss.data[0], total_event_loss.data[0], total_time_loss.data[0], total_goals_home_loss.data[0], total_goals_away_loss.data[0], total_goals_diff_loss.data[0]
        else:
            return sampled_events, sampled_times, target_events, target_times, loss.data[0], total_event_loss.data[0], total_time_loss.data[0], total_same_minute_event_loss.item(), total_result_loss.data[0], total_accuracy.item()
            #return sampled_events, sampled_times, target_events, target_times, loss.data[0], total_event_loss.data[0], total_time_loss.data[0], total_goals_home_loss.data[0], total_goals_away_loss.data[0], total_goals_diff_loss.data[0]


    def sample(self, teams, events=None, times=None, return_proba=False):
        if events is None:
            events = []

        if times is None:
            times = []

        assert(len(events) == len(times))

        sampled_events = []
        sampled_times = []

        event_probas = []
        time_probas = []

        current_input = Variable(torch.zeros(1, 1, NB_ALL_EVENTS + NB_ALL_TIMES))
        current_input[0, 0, SOG_TOKEN] = 1
        current_input[0, 0, NB_ALL_EVENTS + GAME_NOT_RUNNING_TIME] = 1

        self.hidden = self.init_hidden(teams)

        teams_tensor = get_teams_caracteristics(teams)
        teams_input = teams_tensor.squeeze(0).unsqueeze(1)

        current_time = 1
        concurrent_same_minute = 0
        i = 0
        while current_time <= 90:
            #print("Input sample:", current_input)

            #print("Teams input:", teams_input)
            #print("current input:", current_input)

            input_with_prior = torch.cat([current_input, teams_input], 2)
            output, self.hidden = self.lstm(input_with_prior, self.hidden)

            event_scores = self.hidden2event(output)
            time_scores = self.hidden2time(output)

            event_proba = F.softmax(event_scores, 2)
            time_proba = F.softmax(time_scores, 2)

            generated_event = int(torch.multinomial(event_proba[0, 0], 1)[0])
            generated_time = int(torch.multinomial(time_proba[0, 0], 1)[0])

            # Force given events and times from the beginning of the game
            if i + 1 < len(events):
                generated_event = events[i+1]
                generated_time = times[i+1]

            # Force different time if generating NO_EVENT
            if generated_event == NO_EVENT:
                generated_time = DIFF_TIME_THAN_PREV
            
            if generated_time == DIFF_TIME_THAN_PREV:
                current_time += 1
                concurrent_same_minute = 1
            else:
                concurrent_same_minute += 1

            if concurrent_same_minute > 10:
                print("Oups:", concurrent_same_minute)
                #return self.sample(teams, return_proba=return_proba)

            sampled_events.append(generated_event)
            sampled_times.append(generated_time)
        
            # Store probabilities of event to happen
            event_probas.append([])
            for event_nb in range(NB_ALL_EVENTS):
                event_probas[-1].append(event_proba[0, 0, event_nb])

            # Store probabilities of time
            time_probas.append([])
            for time_nb in range(NB_ALL_TIMES):
                time_probas[-1].append(time_proba[0, 0, time_nb])

            current_input = Variable(torch.zeros(1, 1, NB_ALL_EVENTS + NB_ALL_TIMES))
            current_input[0, 0, generated_event] = 1
            current_input[0, 0, NB_ALL_EVENTS + generated_time] = 1

            i += 1

        if return_proba:
            return sampled_events, sampled_times, event_probas, time_probas
        else:
            return sampled_events, sampled_times

