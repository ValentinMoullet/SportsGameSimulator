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

from loader import TrainingSetFirstHalf, TestSetFirstHalf
from parameters import *
from model import *
from plot import *
from utils import *


training_set = TrainingSetFirstHalf(batch_size=BATCH_SIZE)
train_loader = DataLoader(training_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

test_set = TestSetFirstHalf(batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

all_best_test_accuracy = 0
all_best_model = None
all_best_params = ''
for learning_rate in [1e-3, 1e-4]:
    for hidden1 in [10, 25]:
        for hidden2 in [10, 20]:
            for d_rate in [0.0, 0.5]:
                params = get_hyperparams_filename('', batch_size=BATCH_SIZE, learning_rate=learning_rate, hidden_layer_size1=hidden1, hidden_layer_size2=hidden2, dropout_rate=d_rate)
                
                print("Training with:", params)
                
                model = FirstHalfNN(40, learning_rate=learning_rate, hidden_layer_size1=hidden1, hidden_layer_size2=hidden2, d_rate=d_rate)

                training_loss_history = []
                training_accuracy_history = []

                test_loss_history = []
                test_accuracy_history = []

                best_model = None
                best_test_accuracy = 0
                for epoch in tqdm(range(MAX_EPOCH)):

                    ########## Train the model ##########

                    losses_training = []
                    accuracies_training = []
                    for data, target, teams in train_loader:
                        teams = list(zip(teams[0], teams[1]))
                        data = Variable(data)
                        target = Variable(target)

                        proba, loss = model.step(data, target, teams)

                        accuracy = 0
                        for batch in range(proba.size(0)):
                            accuracy += proba[batch, target[batch].item()].item()

                        accuracy /= proba.size(0)

                        losses_training.append(loss)
                        accuracies_training.append(accuracy)

                    training_loss = np.mean(losses_training)
                    training_accuracy = np.mean(accuracies_training)

                    training_loss_history.append(training_loss)
                    training_accuracy_history.append(training_accuracy)

                    ########## Test the model ##########

                    model.eval()

                    accuracy = 0
                    test_set_size = 0
                    losses_test = []
                    accuracies_test = []
                    for data, target, teams in test_loader:
                        teams = list(zip(teams[0], teams[1]))
                        data = Variable(data)
                        target = Variable(target)

                        proba, loss = model.predict_proba_and_get_loss(data, target, teams)
                        
                        accuracy = 0
                        for batch in range(proba.size(0)):
                            accuracy += proba[batch, target[batch].item()].item()

                        accuracy /= proba.size(0)
                        
                        losses_test.append(loss)
                        accuracies_test.append(accuracy)

                    loss_test = np.mean(losses_test)
                    test_accuracy = np.mean(accuracies_test)

                    test_loss_history.append(loss_test)
                    test_accuracy_history.append(test_accuracy)

                    if best_test_accuracy < test_accuracy:
                        best_test_accuracy = test_accuracy
                        best_model = model

                loss_histories = (
                    ('Training loss', training_loss_history),
                    ('Test loss', test_loss_history),
                    )
                filename = get_dated_filename(get_hyperparams_filename('loss.pdf', batch_size=BATCH_SIZE, learning_rate=learning_rate, hidden_layer_size1=hidden1, hidden_layer_size2=hidden2, dropout_rate=d_rate))
                plot_history(loss_histories, "%s/%s" % (SECOND_HALF_PRED_DIR, filename), "Training and test loss")

                accuracies_histories = (
                    ('Training accuracy', training_accuracy_history),
                    ('Test accuracy', test_accuracy_history),
                    )
                filename = get_dated_filename(get_hyperparams_filename('accuracy.pdf', batch_size=BATCH_SIZE, learning_rate=learning_rate, hidden_layer_size1=hidden1, hidden_layer_size2=hidden2, dropout_rate=d_rate))
                plot_history(accuracies_histories, "%s/%s" % (SECOND_HALF_PRED_DIR, filename), "Different accuracies")

                print("Best test accuracy:", best_test_accuracy)

                if best_test_accuracy > all_best_test_accuracy:
                    all_best_test_accuracy = best_test_accuracy
                    all_best_model = best_model
                    all_best_params = params

print("Overall best test accuracy:", all_best_test_accuracy)
print("With params:", all_best_params)

########## Save the model for later use ##########

torch.save(all_best_model.state_dict(), "%s/%s/%s" % (SECOND_HALF_PRED_DIR, MODELS_DIR, get_dated_filename('model.pt')))

