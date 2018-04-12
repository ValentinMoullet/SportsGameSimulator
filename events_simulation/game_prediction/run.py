import sys, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from tqdm import tqdm
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

########## Train our model ##########

game_info_df = pd.read_csv('../../data/football-events/ginf.csv')
game_info_df = game_info_df[game_info_df['league'] == league]

# Get teams
teams_to_idx = get_teams_to_idx(game_info_df)
teams = teams_to_idx.keys()

# Removing if teams didn't play both for home and away
game_info_df = game_info_df[(game_info_df['ht'].isin(teams)) & (game_info_df['at'].isin(teams))]

np.random.seed(SEED)
msk = np.random.rand(len(game_info_df)) < TRAINING_SET_RATIO

train_loader = DataLoader(TrainingSet(game_info_df=game_info_df, teams_to_idx=teams_to_idx, msk=msk), num_workers=4, batch_size=CHOSEN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TestSet(game_info_df=game_info_df, teams_to_idx=teams_to_idx, msk=msk), num_workers=4, batch_size=1, shuffle=False)

nb_teams = next(iter(train_loader))[0].size(1) // 2

print(nb_teams)

training_loss_history = []
test_loss_history = []
test_accuracy_history = []

########## Train the model ##########

model = NN(nb_teams=nb_teams, learning_rate=CHOSEN_LEARNING_RATE, hidden_layer_size1=CHOSEN_HIDDEN_LAYER_SIZES[0], hidden_layer_size2=CHOSEN_HIDDEN_LAYER_SIZES[1], d_ratio=CHOSEN_DROPOUT_RATE)

best_loss = (1000, 0)
best_accuracy = (0, 0)
for epoch in tqdm(range(CHOSEN_EPOCH)):

    losses_training = []
    for data, target in train_loader:
        data = Variable(data)
        target = Variable(target)
        loss = model.step(data, target)
        losses_training.append(loss)

    # Mean of the losses of training predictions
    training_loss_history.append(np.mean(losses_training))

    ########## Test at this epoch ##########

    model.eval()

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

    test_loss_history.append(loss_test)
    test_accuracy_history.append(accuracy_test)

    if loss_test < best_loss[0]:
        best_loss = (loss_test, epoch)

    if accuracy_test > best_accuracy[0]:
        best_accuracy = (accuracy_test, epoch)

print("Best test loss: %.5f at epoch %d" % (best_loss[0], best_loss[1]))
print("Best test accuracy: %.5f at epoch %d" % (best_accuracy[0], best_accuracy[1]))

loss_histories = (
    ('Training loss', training_loss_history),
    ('Test loss', test_loss_history))
plot_history(loss_histories, '%s/results/%s' % (league, get_hyperparams_filename('loss.pdf')), "Test and training loss")

accuracy_histories = (
    ('Test accuracy', test_accuracy_history),)
plot_history(accuracy_histories, '%s/results/%s' % (league, get_hyperparams_filename('accuracy.pdf')), "Test accuracy")

########## Test the final model ##########

model.eval()

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

    # Compute confusion matrix info
    max_score, idx = y_pred_proba.max(1)
    class_correct[target.data[0]] += 1 if idx.data[0] == target.data[0] else 0
    class_total[target.data[0]] += 1

loss_test = np.mean(losses_test)
accuracy_test = accuracy / len(test_loader)

print("Test loss:", loss_test)
print("Test accuracy:", accuracy_test)

print()

for i in range(3):
    print('Accuracy of %d : %2d / %2d (%2d %%)' % (i, class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))


########## Get weights of first layer and plot to see if they seem to be features for teams ##########

plot_weights_teams_tsne(teams, model, '%s/weights/%s' % (league, get_hyperparams_filename('tsne.pdf')), "Test and training loss")
plot_weights_teams_pca(teams, model, '%s/weights/%s' % (league, get_hyperparams_filename('pca.pdf')), "Test and training loss")
plot_weights_teams_kernel_pca(teams, model, '%s/weights/%s' % (league, get_hyperparams_filename('kernel_pca.pdf')), "Test and training loss")

########## Save the model for later use ##########

torch.save(model.state_dict(), "%s/%s/%s" % (MODELS_DIR, league, get_hyperparams_filename('model.pt')))
