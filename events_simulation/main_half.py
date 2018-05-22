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

model = FirstHalfNN(40)

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
        
        losses_test.append(loss)
        accuracies_test.append(accuracy / proba.size(0))

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
plot_history(loss_histories, "%s/%s" % (SECOND_HALF_PRED_DIR, get_dated_filename('loss.pdf')), "Training and test loss")

accuracies_histories = (
    ('Training accuracy', training_accuracy_history),
    ('Test accuracy', test_accuracy_history),
    )
plot_history(accuracies_histories, "%s/%s" % (SECOND_HALF_PRED_DIR, get_dated_filename('accuracy.pdf')), "Different accuracies")

print("Best test accuracy:", best_test_accuracy)

########## Save the model for later use ##########

torch.save(best_model.state_dict(), "%s/%s/%s" % (SECOND_HALF_PRED_DIR, MODELS_DIR, get_dated_filename('model.pt')))

