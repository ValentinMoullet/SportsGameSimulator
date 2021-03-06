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

from loader import TrainingSet, TestSet, BookmakersPred
from utils import *
from team import *
from parameters import *
from plot import *
from preprocessing import *
from model import NN


########## Parsing args ##########

# All leagues: ['D1', 'F1', 'E0', 'SP1', 'I1']

if len(sys.argv) > 1:
    league = sys.argv[1]
else:
    league = DEFAULT_LEAGUE

########## Compute bookmakers score ##########

if BOOKMAKERS_OVERVIEW:

    bookmakers_pred_loader = DataLoader(BookmakersPred(league=league), num_workers=4, shuffle=True)

    scores_bookmakers = []
    losses_bookmakers = []
    correct = 0
    accuracy = 0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    for y_pred, target in bookmakers_pred_loader:
        y_pred = Variable(y_pred)
        target = Variable(target)

        y_pred *= 1 / torch.sum(y_pred)

        loss = F.cross_entropy(y_pred, target)
        score = torch.exp(-loss).item()

        scores_bookmakers.append(score)
        losses_bookmakers.append(loss.item())

        accuracy += y_pred.data[0][target.data[0]]

    accuracy /= len(bookmakers_pred_loader)
    score_bookmakers = np.mean(scores_bookmakers)
    loss_bookmakers = np.mean(losses_bookmakers)

    print("Bookmakers loss:", loss_bookmakers)
    print("Bookmakers score:", score_bookmakers)
    print("Accuracy:", accuracy)
    print()

    print("Bookmakers predictions overview done.")
    print()
    print("##################################################")

########## Train our model ##########

game_info_df = pd.read_csv('../../data/football-events/ginf.csv')
game_info_df = game_info_df[game_info_df['league'] == league]

# Get teams
teams_to_idx = get_teams_to_idx(game_info_df)
teams = teams_to_idx.keys()

# Removing if teams didn't play both for home and away
game_info_df = game_info_df[(game_info_df['ht'].isin(teams)) & (game_info_df['at'].isin(teams))]

overall_best_params = []
overall_best_loss = 1000
for batch_size in BATCH_SIZES:
    train_loader = DataLoader(TrainingSet(game_info_df=game_info_df, teams_to_idx=teams_to_idx), num_workers=4, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TestSet(game_info_df=game_info_df, teams_to_idx=teams_to_idx), num_workers=4, batch_size=1, shuffle=False)

    nb_teams = next(iter(train_loader))[0].size(1) // 2

    for learning_rate in LEARNING_RATES:
        for size1, size2 in HIDDEN_LAYER_SIZES:
            for d_ratio in DROPOUT_RATES:
                print("Training with: batch_size=%d | learning_rate=%.6f | hidden_layer_sizes=(%d_%d) | dropout_rate=%.1f" % (batch_size, learning_rate, size1, size2, d_ratio))

                training_loss_history = []
                validation_loss_history = []
                test_loss_history = []

                validation_accuracy_history = []
                test_accuracy_history = []

                # Create training and validation split
                random_number = np.random.randint(1000000000)
                all_train_data, all_train_targets, all_valid_data, all_valid_targets = train_valid_split_k_fold(train_loader, K_FOLD, random_number)

                all_losses_training = []
                all_scores_validation = []
                all_losses_validation = []
                all_accuracy_validation = []
                all_scores_test = []
                all_losses_test = []
                all_accuracy_test = []

                folds_training_losses = [[0]*K_FOLD for _ in range(MAX_EPOCH)]
                folds_validation_losses = [[0]*K_FOLD for _ in range(MAX_EPOCH)]
                folds_validation_accuracies = [[0]*K_FOLD for _ in range(MAX_EPOCH)]
                folds_test_losses = [[0]*K_FOLD for _ in range(MAX_EPOCH)]
                folds_test_accuracies = [[0]*K_FOLD for _ in range(MAX_EPOCH)]
                for k in range(K_FOLD):
                    model = NN(nb_teams=nb_teams, learning_rate=learning_rate, hidden_layer_size1=size1, hidden_layer_size2=size2, d_ratio=d_ratio)

                    train_data = all_train_data[k]
                    train_targets = all_train_targets[k]
                    valid_data = all_valid_data[k]
                    valid_targets = all_valid_targets[k]

                    # Combine train/validation data and targets as tuples
                    train_data_and_targets = list(zip(train_data, train_targets))
                    valid_data_and_targets = list(zip(valid_data, valid_targets))

                    best_loss = (0,1000)
                    for epoch in tqdm(range(MAX_EPOCH)):

                        ########## Train the model ##########

                        losses_training = []
                        for data, target in train_data_and_targets:
                            loss = model.step(data, target)
                            losses_training.append(loss)

                        # Mean of the losses of training predictions
                        folds_training_losses[epoch][k] = np.mean(losses_training)

                        ########## Validate the model ##########

                        model.eval()

                        scores_validation = []
                        losses_validation = []
                        accuracy = 0
                        validation_set_size = 0
                        class_correct = list(0 for i in range(3))
                        class_total = list(0 for i in range(3))
                        for data, target in valid_data_and_targets:
                            y_pred_proba, loss = model.predict_proba_and_get_loss(data, target)

                            losses_validation.append(loss)

                            for i in range(len(target.data)):
                                validation_set_size += 1
                                accuracy += y_pred_proba.data[i][target.data[i]]

                        accuracy_validation = accuracy / validation_set_size
                        loss_validation = np.mean(losses_validation)

                        # Mean of the losses of validation predictions
                        folds_validation_losses[epoch][k] = loss_validation
                        folds_validation_accuracies[epoch][k] = accuracy_validation

                        ########## Test the model at this epoch ##########

                        if TRAINING_SET_RATIO < 1.0:

                            accuracy = 0
                            losses_test = []
                            class_correct = list(0. for i in range(3))
                            class_total = list(0. for i in range(3))
                            for data, target in test_loader:
                                data = Variable(data)
                                target = Variable(target)
                                y_pred_proba, loss = model.predict_proba_and_get_loss(data, target)

                                losses_test.append(loss)

                                accuracy += y_pred_proba.data[0][target.data[0]]

                            loss_test = np.mean(losses_test)
                            accuracy_test = accuracy / len(test_loader)

                            folds_test_losses[epoch][k] = loss_test
                            folds_test_accuracies[epoch][k] = accuracy_test

                        # Save the best model
                        if loss_validation < best_loss[1]:
                            best_loss = (epoch, loss_validation)

                        # Check that the model is not doing worst over the time
                        if best_loss[0] + MAX_EPOCH_WITHOUT_IMPROV < epoch:
                            print("Overfitting. Stopped at epoch {}." .format(epoch))
                            break

                    print("[k=%d] Best loss for params: %.5f at epoch %d" % (k, best_loss[1], best_loss[0]))
                
                print()

                # Compute mean over all folds
                limit_epoch = MAX_EPOCH
                for epoch in range(MAX_EPOCH):
                    if 0 in folds_validation_losses[epoch]:
                        limit_epoch = epoch
                        break

                    folds_training_losses[epoch] = np.mean(folds_training_losses[epoch])
                    folds_validation_losses[epoch] = np.mean(folds_validation_losses[epoch])
                    folds_validation_accuracies[epoch] = np.mean(folds_validation_accuracies[epoch])
                    folds_test_losses[epoch] = np.mean(folds_test_losses[epoch])
                    folds_test_accuracies[epoch] = np.mean(folds_test_accuracies[epoch])

                folds_training_losses = folds_training_losses[:limit_epoch]
                folds_validation_losses = folds_validation_losses[:limit_epoch]
                folds_validation_accuracies = folds_validation_accuracies[:limit_epoch]
                folds_test_losses = folds_test_losses[:limit_epoch]
                folds_test_accuracies = folds_test_accuracies[:limit_epoch]    

                training_loss_history.extend(folds_training_losses)
                validation_loss_history.extend(folds_validation_losses)
                validation_accuracy_history.extend(folds_validation_accuracies)
                test_loss_history.extend(folds_test_losses)
                test_accuracy_history.extend(folds_test_accuracies)

                print("--------------------\n")

                # Keep best one
                folds_best_validation_loss = min(folds_validation_losses)
                if folds_best_validation_loss < overall_best_loss:
                    overall_best_loss = folds_best_validation_loss
                    overall_best_params = (batch_size, learning_rate, (size1, size2), d_ratio)

                loss_histories = (
                    ('Training loss', training_loss_history),
                    ('Validation loss', validation_loss_history),
                    ('Test loss', test_loss_history))
                plot_history(loss_histories, '%s/%s' % (league, get_hyperparams_filename('loss.pdf', batch_size, learning_rate, size1, size2, d_ratio)), "Validation and training loss")

                accuracy_histories = (
                    ('Validation accuracy', validation_accuracy_history),
                    ('Test accuracy', test_accuracy_history))
                plot_history(accuracy_histories, '%s/%s' % (league, get_hyperparams_filename('accuracy.pdf', batch_size, learning_rate, size1, size2, d_ratio)), "Validation and test accuracy")

print("Best loss:", overall_best_loss)
print("Best parameters:", overall_best_params)

