#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
 
"""
    Neural Network implemenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, nb_teams, learning_rate=1e-3, hidden_layer_size=50):
        super(NN, self).__init__()

        self.nb_teams = nb_teams

        self.loss_function = nn.CrossEntropyLoss()

        # Layers
        self.input_layer = nn.Linear(nb_teams, hidden_layer_size)
        self.home_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.away_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size*2, 3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, input):
        """Feed the model with input and return output."""

        inputs = torch.split(input, self.nb_teams, 1)

        home_teams1 = self.input_layer(inputs[0])
        away_teams1 = self.input_layer(inputs[1])

        home_teams2 = self.home_layer(home_teams1)
        away_teams2 = self.away_layer(away_teams1)

        output = self.output_layer(torch.cat([home_teams2, away_teams2], 1))

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

    def predict(self, input):
        """
        Predict an input using the trained network + do a exp to
        get proba of classes.
        """

        return self.forward(input)
