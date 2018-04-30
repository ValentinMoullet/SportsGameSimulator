#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Neural Network implemenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from parameters import *


class NN(nn.Module):
    def __init__(self, nb_teams, learning_rate=1e-4, hidden_layer_size1=50, hidden_layer_size2=50, d_ratio=0.2):
        super(NN, self).__init__()

        self.nb_teams = nb_teams

        self.loss_function = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=d_ratio)
        self.activation = nn.ReLU()

        # Layers
        self.input_layer = nn.Sequential(
            nn.Linear(nb_teams, hidden_layer_size1),
            self.activation,
            self.dropout)

        self.home_layer = nn.Sequential(
            nn.Linear(hidden_layer_size1, hidden_layer_size2),
            self.activation,
            self.dropout)
        self.away_layer = nn.Sequential(
            nn.Linear(hidden_layer_size1, hidden_layer_size2),
            self.activation,
            self.dropout)

        self.output_layer = nn.Linear(hidden_layer_size2*2, NB_CLASSES)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, input):
        """Feed the model with input and return output."""

        inputs = torch.split(input, self.nb_teams, 1)

        home_teams1 = self.input_layer(inputs[0])
        away_teams1 = self.input_layer(inputs[1])

        home_teams2 = self.home_layer(home_teams1)
        away_teams2 = self.away_layer(away_teams1)

        last_hidden_layer = torch.cat([home_teams2, away_teams2], 1)

        #print(last_hidden_layer)

        output = self.output_layer(last_hidden_layer)

        #print("Output:", output)

        return output

    def step(self, input, target):
        """Do one training step and return the loss."""

        self.train()
        self.zero_grad()
        out = self.forward(input)
        loss = self.loss_function(out, target)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def get_last_layer(self, input):
        inputs = torch.split(input, self.nb_teams, 1)

        home_teams1 = self.input_layer(inputs[0])
        away_teams1 = self.input_layer(inputs[1])

        home_teams2 = self.home_layer(home_teams1)
        away_teams2 = self.away_layer(away_teams1)

        last_hidden_layer = torch.cat([home_teams2, away_teams2], 1)

        return last_hidden_layer

    def predict(self, input):
        """
        Predict an input using the trained network.
        """

        return self.forward(input)

    def predict_proba(self, input):
        return F.softmax(self.predict(input), dim=1)

    def predict_proba_and_get_loss(self, input, target):
        prediction = self.forward(input)
        loss = self.loss_function(prediction, target)
        return self.predict_proba(input), loss.data[0]

    def predict_and_get_loss(self, input, target):
        out = self.forward(input)
        loss = self.loss_function(out, target)
        return out, loss.data[0]
