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

class LSTMEvents(nn.Module):

    def __init__(self, hidden_dim, event_types_size, time_types_size, num_layers=1, batch_size=1, learning_rate=0.01):
        super(LSTMEvents, self).__init__()

        self.loss_function = nn.CrossEntropyLoss()

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
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))

    def forward(self, input):
        #print("Input:", input.size())
        reshaped_input = input.permute(1, 0, 2)
        #print("Reshaped input:", reshaped_input.size())
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        #print("LSTM out:", lstm_out.size())

        #reshaped_output = lstm_out.view(input.size(1), -1)
        reshaped_output = lstm_out.permute(1, 0, 2)
        #print("Reshaped LSTM out:", reshaped_output.size())
        event_space = self.hidden2event(lstm_out)
        time_space = self.hidden2time(repackage(lstm_out))

        #print("Event space:", event_space.size())

        #event_scores = F.softmax(event_space, dim=1)
        #time_scores = F.softmax(time_space, dim=1)

        return event_space, time_space

    def step(self, input, target):
        """Do one training step and return the loss."""

        self.train()
        self.zero_grad()
        self.hidden = repackage(self.hidden)
        event_scores, time_scores = self.forward(input)
        #print('Scores')
        #print(event_scores.size())
        #print(time_scores.size())
        #print('-----')
        target_events = target[:, :, 0]
        target_time = target[:, :, 1]
        target_events = target_events.contiguous().view(-1)
        target_time = target_time.contiguous().view(-1)
        #print(target_events)
        #print(target_time)
        reshaped_event_scores = event_scores.view(-1, self.event_types_size)
        reshaped_time_scores = time_scores.view(-1, self.time_types_size)
        #target_events = target_events.view(-1, self.event_types_size)
        #print("Event scores:", reshaped_event_scores.size())
        #print("Target events:", target_events.size())
        loss_events = self.loss_function(reshaped_event_scores, target_events)
        loss_time = self.loss_function(reshaped_time_scores, target_time)

        loss_events.backward()
        loss_time.backward()
        self.optimizer.step()

        return loss_events.data[0], F.softmax(event_scores, dim=2), F.softmax(time_scores, dim=2)
