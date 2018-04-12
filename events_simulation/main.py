import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from tqdm import tqdm
tqdm.monitor_interval = 0
from sklearn.metrics import accuracy_score
from random import shuffle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from loader import TrainingSet, TestSet
from parameters import *
from model import *
from plot import *
from utils import *


training_set = TrainingSet(batch_size=BATCH_SIZE)
train_loader = DataLoader(training_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

test_set = TestSet(batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

# Create training and validation split
random_number = np.random.randint(1000000000)
all_train_data, all_train_targets, all_train_teams, all_valid_data, all_valid_targets, all_valid_teams = train_valid_split_k_fold(train_loader, K_FOLD, random_number)

for k in range(K_FOLD):
    model = LSTMEvents(40, NB_EVENT_TYPES * 2 + 2, 3, batch_size=BATCH_SIZE)

    train_data = all_train_data[k]
    train_targets = all_train_targets[k]
    train_teams = all_train_teams[k]
    valid_data = all_valid_data[k]
    valid_targets = all_valid_targets[k]
    valid_teams = all_valid_teams[k]

    # Combine train/validation data and targets as tuples
    train_info = list(zip(train_data, train_targets, train_teams))
    valid_info = list(zip(valid_data, valid_targets, valid_teams))

    training_loss_history = []
    validation_accuracy_history = []
    validation_loss_history = []
    test_accuracy_history = []
    test_loss_history = []
    for epoch in tqdm(range(MAX_EPOCH)):

        ########## Train the model ##########

        model.hidden = model.init_hidden()

        losses_training = []
        for data, target, teams in train_info:
            teams = list(zip(teams[0], teams[1]))
            event_pred_proba, time_pred_proba, loss = model.step(data, target, teams)
            losses_training.append(loss)

            '''
            for batch_idx in range(event_pred_proba.size(0)):
                goal_home = 0
                goal_away = 0
                for event_idx in range(event_pred_proba.size(1)):
                    goal_home += event_pred_proba[batch_idx, event_idx, GOAL_HOME].data[0]
                    goal_away += event_pred_proba[batch_idx, event_idx, GOAL_AWAY].data[0]

                #print("%.4f -VS- %.4f" % (goal_home, goal_away))
                if goal_away > goal_home:
                    print("%.4f - %.4f" % (goal_home, goal_away))
                    print("%s -VS- %s" % (teams[batch_idx][0], teams[batch_idx][1]))
            '''

        training_loss = np.mean(losses_training)
        #print("Loss at epoch %d:" % epoch, training_loss)
        training_loss_history.append(training_loss)

        if epoch + 1 == MAX_EPOCH and k + 1 == 1:
            output_events_file(event_pred_proba, time_pred_proba, target, teams, get_dated_filename('training.txt'))

            generated_events = generate_events(event_pred_proba, time_pred_proba)

            # Only get events during the games
            event_gen = generated_events[:, :, 0]
            time_gen = generated_events[:, :, 1]

            events_during_game, target_events_during_game, _, _ = get_during_game_tensors(event_gen, time_gen, target, proba=False)

            events_count_dict = count_events(events_during_game)
            events_target_count_dict = count_events(target_events_during_game.data)
            plot_events_count(events_count_dict, events_target_count_dict, get_dated_filename('training.pdf'))

        ########## Validate the model ##########

        model.eval()

        losses_validation = []
        accuracy = 0
        validation_set_size = 0
        class_correct = list(0 for i in range(3))
        class_total = list(0 for i in range(3))
        for data, target, teams in valid_info:
            teams = list(zip(teams[0], teams[1]))
            event_pred_proba, time_pred_proba, loss = model.predict_proba_and_get_loss(data, target, teams)

            losses_validation.append(loss)

            for batch in range(event_pred_proba.size(0)):
                for event in range(event_pred_proba.size(1)):
                    validation_set_size += 1
                    accuracy += event_pred_proba[batch, event, target[batch, event, 0].data[0]].data[0]

            '''
            for batch in range(len(target[:, :, 0].data)):
                for event in range(len(target[batch, :, 0].data)):
                    validation_set_size += 1
                    #print("Event:", event_pred_proba)
                    #print("Target nb:", target[batch, event, 0].data[0])
                    #print("Event proba:", event_pred_proba[batch, event, target[batch, event, 0].data[0]].data[0])
                    accuracy += event_pred_proba[batch, event, target[batch, event, 0].data[0]].data[0]
                    
            '''

        accuracy_validation = accuracy / validation_set_size
        loss_validation = np.mean(losses_validation)

        validation_accuracy_history.append(accuracy_validation)
        validation_loss_history.append(loss_validation)

        ########## Test the model at this epoch ##########

        accuracy = 0
        test_set_size = 0
        losses_test = []
        for data, target, teams in test_loader:
            teams = list(zip(teams[0], teams[1]))
            data = Variable(data)
            target = Variable(target)
            event_pred_proba, time_pred_proba, loss = model.predict_proba_and_get_loss(data, target, teams)

            losses_test.append(loss)

            for batch in range(event_pred_proba.size(0)):
                for event in range(event_pred_proba.size(1)):
                    test_set_size += 1
                    accuracy += event_pred_proba[batch, event, target[batch, event, 0].data[0]].data[0]

        loss_test = np.mean(losses_test)
        accuracy_test = accuracy / test_set_size

        test_accuracy_history.append(accuracy_test)
        test_loss_history.append(loss_test)

        if epoch + 1 == MAX_EPOCH and k + 1 == 1:
            output_events_file(event_pred_proba, time_pred_proba, target, teams, get_dated_filename('test.txt'))

            generated_events = generate_events(event_pred_proba, time_pred_proba)

            # Only get events during the games
            event_gen = generated_events[:, :, 0]
            time_gen = generated_events[:, :, 1]
            events_during_game, target_events_during_game, _, _ = get_during_game_tensors(event_gen, time_gen, target, proba=False)

            events_count_dict = count_events(events_during_game)
            events_target_count_dict = count_events(target_events_during_game.data)
            plot_events_count(events_count_dict, events_target_count_dict, get_dated_filename('test.pdf'))


loss_histories = (
    ('Training loss', training_loss_history),
    ('Validation loss', validation_loss_history),
    ('Test loss', test_loss_history),)
plot_history(loss_histories, get_dated_filename('loss.pdf'), "Training, validation and test loss")

accuracy_histories = (
    ('Validation accuracy', validation_accuracy_history),
    ('Test accuracy', test_accuracy_history),)
plot_history(accuracy_histories, get_dated_filename('accuracy.pdf'), "Validation and test accuracy")

'''
training_loss_history = []
for epoch in tqdm(range(MAX_EPOCH)):

    ########## Train the model ##########

    model.hidden = model.init_hidden()

    losses_training = []
    for data, target, teams in train_loader:
        teams = list(zip(teams[0], teams[1]))
        data = Variable(data)
        target = Variable(target)
        #print(data.size())
        #print(target.size())
        loss, event_scores, time_scores = model.step(data, target)
        #print("Loss:", loss)
        losses_training.append(loss)

    training_loss = np.mean(losses_training)
    #print("Loss at epoch %d:" % epoch, training_loss)
    training_loss_history.append(training_loss)

    if epoch + 1 == MAX_EPOCH:
        output_events_file(event_scores, time_scores, target, 'test.txt')

        generated_events = generate_events(event_scores, time_scores)
        events_count_dict = count_events(generated_events)
        plot_events_count(events_count_dict, 'test.pdf')

        events_count_dict = count_events(target.data)
        plot_events_count(events_count_dict, 'test_target.pdf')

loss_histories = (
    ('Training loss', training_loss_history),)
plot_history(loss_histories, 'loss.pdf', "Training loss")
'''

