import sys, math
import numpy as np
import pandas as pd
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
    best_model = None
    best_test_accuracy = 0

    model = LSTMEvents(40, NB_ALL_EVENTS, 3, batch_size=BATCH_SIZE)
    if CUDA:
        model.cuda()

    train_data = all_train_data[k]
    train_targets = all_train_targets[k]
    train_teams = all_train_teams[k]
    valid_data = all_valid_data[k]
    valid_targets = all_valid_targets[k]
    valid_teams = all_valid_teams[k]

    # Combine train/validation data and targets as tuples
    train_info = list(zip(train_data, train_targets, train_teams))
    valid_info = list(zip(valid_data, valid_targets, valid_teams))

    # Little hack to make it work even if we don't want K-fold validation
    if K_FOLD == 1:
        temp = train_info
        train_info = valid_info
        valid_info = temp

    training_loss_history = []
    training_result_loss_history = []
    training_events_loss_history = []
    training_time_loss_history = []
    training_same_minute_event_loss_history = []
    training_accuracy_history = []
    training_hg_loss_history = []
    training_ag_loss_history = []
    training_diff_loss_history = []

    validation_accuracy_history = []

    validation_loss_history = []
    validation_result_loss_history = []
    validation_events_loss_history = []
    validation_time_loss_history = []
    validation_same_minute_event_loss_history = []
    validation_accuracy_history = []
    validation_hg_loss_history = []
    validation_ag_loss_history = []
    validation_diff_loss_history = []

    test_accuracy_history = []

    test_loss_history = []
    test_result_loss_history = []
    test_events_loss_history = []
    test_time_loss_history = []
    test_same_minute_event_loss_history = []
    test_accuracy_history = []
    test_hg_loss_history = []
    test_ag_loss_history = []
    test_diff_loss_history = []
    for epoch in tqdm(range(MAX_EPOCH)):

        ########## Train the model ##########

        losses_training = []
        result_losses_training = []
        events_losses_training = []
        time_losses_training = []
        same_minute_event_losses_training = []
        accuracies_training = []
        hg_losses_training = []
        ag_losses_training = []
        diff_losses_training = []
        for data, target, teams in train_info:
            if CUDA:
                data = data.cuda()
                target = target.cuda()

            teams = list(zip(teams[0], teams[1]))
            event_pred_proba, time_pred_proba, loss, events_loss, time_loss, same_minute_event_loss, result_loss, accuracy = model.step(data, target, teams)

            losses_training.append(loss)
            result_losses_training.append(result_loss)
            events_losses_training.append(events_loss)
            time_losses_training.append(time_loss)
            same_minute_event_losses_training.append(same_minute_event_loss)
            accuracies_training.append(accuracy)

        training_loss = np.mean(losses_training)
        training_result_loss = np.mean(result_losses_training)
        training_events_loss = np.mean(events_losses_training)
        training_time_loss = np.mean(time_losses_training)
        training_same_minute_event_loss = np.mean(same_minute_event_losses_training)
        training_accuracy = np.mean(accuracies_training)
        training_hg_loss = np.mean(hg_losses_training)
        training_ag_loss = np.mean(ag_losses_training)
        training_diff_loss = np.mean(diff_losses_training)

        training_loss_history.append(training_loss)
        training_result_loss_history.append(training_result_loss)
        training_events_loss_history.append(training_events_loss)
        training_time_loss_history.append(training_time_loss)
        training_same_minute_event_loss_history.append(training_same_minute_event_loss)
        training_accuracy_history.append(training_accuracy)
        training_hg_loss_history.append(training_hg_loss)
        training_ag_loss_history.append(training_ag_loss)
        training_diff_loss_history.append(training_diff_loss)

        if epoch + 1 == MAX_EPOCH and k + 1 == 1:
            output_events_file(event_pred_proba, time_pred_proba, target, teams, get_dated_filename('training.txt'))

            generated_events = generate_events(event_pred_proba, time_pred_proba)

            # Only get events during the games
            event_gen = generated_events[:, :, 0]
            time_gen = generated_events[:, :, 1]

            events_during_game, target_events_during_game, times_during_game, target_times_during_game = get_during_game_tensors(event_gen, time_gen, target, proba=False)

            events_count_dict = count_events(events_during_game)
            events_target_count_dict = count_events(target_events_during_game.data)
            plot_events_count(events_count_dict, events_target_count_dict, get_dated_filename('training.pdf'))

            times_count_dict = count_times(times_during_game)
            times_target_count_dict = count_times(target_times_during_game.data)
            plot_events_count(times_count_dict, times_target_count_dict, get_dated_filename('times_training.pdf'))

        ########## Validate the model ##########

        ########## NEVER USED ##########

        model.eval()

        losses_validation = []
        result_losses_validation = []
        events_losses_validation = []
        time_losses_validation = []
        same_minute_event_losses_validation = []
        accuracies_validation = []
        hg_losses_validation = []
        ag_losses_validation = []
        diff_losses_validation = []
        accuracy = 0
        validation_set_size = 0
        class_correct = list(0 for i in range(3))
        class_total = list(0 for i in range(3))
        for data, target, teams in valid_info:
            if CUDA:
                data = data.cuda()
                target = target.cuda()

            teams = list(zip(teams[0], teams[1]))

            if SAMPLE_VALID_AND_TEST:
                sampled_events, sampled_times, target_events, target_times, loss, events_loss, time_loss, same_minute_event_loss, result_loss, accuracy = model.sample_and_get_loss(target, teams)
            else:
                event_pred_proba, time_pred_proba, loss, events_loss, time_loss, same_minute_event_loss, result_loss = model.predict_proba_and_get_loss(data, target, teams)

            losses_validation.append(loss)
            result_losses_validation.append(result_loss)
            events_losses_validation.append(events_loss)
            time_losses_validation.append(time_loss)
            same_minute_event_losses_validation.append(same_minute_event_loss)
            accuracies_validation.append(accuracy)
        
        loss_validation = np.mean(losses_validation)
        validation_result_loss = np.mean(result_losses_validation)
        validation_events_loss = np.mean(events_losses_validation)
        validation_time_loss = np.mean(time_losses_validation)
        validation_same_minute_event_loss = np.mean(same_minute_event_losses_validation)
        validation_accuracy = np.mean(accuracies_validation)
        validation_hg_loss = np.mean(hg_losses_validation)
        validation_ag_loss = np.mean(ag_losses_validation)
        validation_diff_loss = np.mean(diff_losses_validation)

        validation_loss_history.append(loss_validation)
        validation_result_loss_history.append(validation_result_loss)
        validation_events_loss_history.append(validation_events_loss)
        validation_time_loss_history.append(validation_time_loss)
        validation_same_minute_event_loss_history.append(validation_same_minute_event_loss)
        validation_accuracy_history.append(validation_accuracy)
        validation_hg_loss_history.append(validation_hg_loss)
        validation_ag_loss_history.append(validation_ag_loss)
        validation_diff_loss_history.append(validation_diff_loss)

        ########## Test the model at this epoch ##########

        accuracy = 0
        test_set_size = 0
        losses_test = []
        result_losses_test = []
        events_losses_test = []
        time_losses_test = []
        same_minute_event_losses_test = []
        accuracies_test = []
        hg_losses_test = []
        ag_losses_test = []
        diff_losses_test = []
        for data, target, teams in test_loader:
            if CUDA:
                data = data.cuda()
                target = target.cuda()

            teams = list(zip(teams[0], teams[1]))
            data = Variable(data)
            target = Variable(target)

            if SAMPLE_VALID_AND_TEST:
                sampled_events, sampled_times, target_events, target_times, all_proba, loss, events_loss, time_loss, same_minute_event_loss, result_loss, accuracy = model.sample_and_get_loss(target, teams, return_proba=True)
            
                goal_home_proba = [[event_proba[GOAL_HOME] for event_proba in all_proba[batch_idx]] for batch_idx in range(len(all_proba))]
                goal_away_proba = [[event_proba[GOAL_AWAY] for event_proba in all_proba[batch_idx]] for batch_idx in range(len(all_proba))]
            else:
                event_proba, time_proba, loss, events_loss, time_loss, same_minute_event_loss, result_loss = model.predict_proba_and_get_loss(data, target, teams)


            losses_test.append(loss)
            result_losses_test.append(result_loss)
            events_losses_test.append(events_loss)
            time_losses_test.append(time_loss)
            same_minute_event_losses_test.append(same_minute_event_loss)
            accuracies_test.append(accuracy)

        loss_test = np.mean(losses_test)
        test_result_loss = np.mean(result_losses_test)
        test_events_loss = np.mean(events_losses_test)
        test_time_loss = np.mean(time_losses_test)
        test_same_minute_event_loss = np.mean(same_minute_event_losses_test)
        test_accuracy = np.mean(accuracies_test)
        test_hg_loss = np.mean(hg_losses_test)
        test_ag_loss = np.mean(ag_losses_test)
        test_diff_loss = np.mean(diff_losses_test)

        test_loss_history.append(loss_test)
        test_result_loss_history.append(test_result_loss)
        test_events_loss_history.append(test_events_loss)
        test_time_loss_history.append(test_time_loss)
        test_same_minute_event_loss_history.append(test_same_minute_event_loss)
        test_accuracy_history.append(test_accuracy)
        test_hg_loss_history.append(test_hg_loss)
        test_ag_loss_history.append(test_ag_loss)
        test_diff_loss_history.append(test_diff_loss)

        if best_test_accuracy < test_accuracy:
            print("New best!")
            best_test_accuracy = test_accuracy
            best_model = model

        print("%.6f accuracy at epoch %d" % (test_accuracy, epoch+1))
        print("%.6f test result loss at epoch %d" % (test_result_loss, epoch+1))
        print("%.6f test loss at epoch %d" % (loss_test, epoch+1))
        print("%.6f test events loss at epoch %d" % (test_events_loss, epoch+1))
        print("%.6f test time loss at epoch %d" % (test_time_loss, epoch+1))
        print("%.6f test same minute event loss at epoch %d" % (test_same_minute_event_loss, epoch+1))

        torch.save(model.state_dict(), "%s/all_epochs/%s" % (MODELS_DIR, get_dated_filename('model_%d.pt' % epoch)))

        # Only useful for checking how it changes at each epoch
        if k + 1 == 1:
            output_already_sampled_events_file(sampled_events, sampled_times, target, goal_home_proba, goal_away_proba, teams, "%s/%s" % ('all_epochs', get_dated_filename('test%d.txt' % epoch)))

        if epoch + 1 == MAX_EPOCH and k + 1 == 1:
            if SAMPLE_VALID_AND_TEST:
                output_already_sampled_events_file(sampled_events, sampled_times, target, goal_home_proba, goal_away_proba, teams, get_dated_filename('test.txt'))

                # Only get events during the games
                events_during_game, target_events_during_game, times_during_game, target_times_during_game = get_during_game_tensors(event_gen, time_gen, target, proba=False)

                events_count_dict = count_events([e for sublist in sampled_events for e in sublist])
                events_target_count_dict = count_events([e for sublist in target_events for e in sublist])
                plot_events_count(events_count_dict, events_target_count_dict, get_dated_filename('test.pdf'))
            
                times_count_dict = count_times(times_during_game)
                times_target_count_dict = count_times(target_times_during_game.data)
                plot_events_count(times_count_dict, times_target_count_dict, get_dated_filename('times_test.pdf'))
            else:
                output_events_file(event_proba, time_proba, target, teams, get_dated_filename('test.txt'))

                generated_events = generate_events(event_proba, time_proba)

                # Only get events during the games
                event_gen = generated_events[:, :, 0]
                time_gen = generated_events[:, :, 1]

                events_during_game, target_events_during_game, times_during_game, target_times_during_game = get_during_game_tensors(event_gen, time_gen, target, proba=False)

                events_count_dict = count_events(events_during_game)
                events_target_count_dict = count_events(target_events_during_game.data)
                plot_events_count(events_count_dict, events_target_count_dict, get_dated_filename('test.pdf'))

                times_count_dict = count_times(times_during_game)
                times_target_count_dict = count_times(target_times_during_game.data)
                plot_events_count(times_count_dict, times_target_count_dict, get_dated_filename('times_test.pdf'))




loss_histories = (
    ('Training loss', training_loss_history),
    ('Test loss', test_loss_history),
    )
plot_history(loss_histories, get_dated_filename('loss.pdf'), "Training, validation and test loss")

training_losses_histories = (
    ('Total loss', training_loss_history),
    ('Result loss', training_result_loss_history),
    ('Events loss', training_events_loss_history),
    ('Time loss', training_time_loss_history),
    ('Same minute event loss', training_same_minute_event_loss_history),
    )
plot_history(training_losses_histories, get_dated_filename('training_losses.pdf'), "Different training losses", thick=[0])

validation_losses_histories = (
    ('Total loss', validation_loss_history),
    ('Result loss', validation_result_loss_history),
    ('Events loss', validation_events_loss_history),
    ('Time loss', validation_time_loss_history),
    ('Same minute event loss', validation_same_minute_event_loss_history),
    )
plot_history(validation_losses_histories, get_dated_filename('validation_losses.pdf'), "Different validation losses", thick=[0])

test_losses_histories = (
    ('Total loss', test_loss_history),
    ('Result loss', test_result_loss_history),
    ('Events loss', test_events_loss_history),
    ('Time loss', test_time_loss_history),
    ('Same minute event loss', test_same_minute_event_loss_history),
    )
plot_history(test_losses_histories, get_dated_filename('test_losses.pdf'), "Different test losses", thick=[0])

result_losses_histories = (
    ('Predicted result loss', test_result_loss_history),
    ('Bookmaker loss', [BOOKMAKER_CE_LOSS] * len(test_result_loss_history)),
    ('Simple NN loss', [SIMPLE_NN_CE_LOSS] * len(test_result_loss_history)),
    )
plot_history(result_losses_histories, get_dated_filename('result_losses.pdf'), "Predicted result VS Bookmaker loss")

accuracies_histories = (
    ('RNN accuracy', test_accuracy_history),
    ('Bookmaker accuracy', [BOOKMAKER_ACCURACY] * len(training_accuracy_history)),
    ('Simple NN accuracy', [SIMPLE_NN_ACCURACY] * len(training_accuracy_history)),
    )
plot_history(accuracies_histories, get_dated_filename('accuracies.pdf'), "Different accuracies", ylabel='Accuracy')

print("Best test accuracy:", best_test_accuracy)

########## Save the model for later use ##########

torch.save(best_model.state_dict(), "%s/%s" % (MODELS_DIR, get_dated_filename('model.pt')))

