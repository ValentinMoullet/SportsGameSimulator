import numpy as np
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
from parameters import *
from model import NN


########## Compute bookmakers score ##########

if BOOKMAKERS_OVERVIEW:
    
    bookmakers_pred_loader = DataLoader(BookmakersPred(league=LEAGUE), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

    scores_validation = []
    correct = 0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    for y_pred, target in bookmakers_pred_loader:
        y_pred = Variable(y_pred)
        target = Variable(target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, target)
        score = torch.exp(-loss).data[0]
        scores_validation.append(score)
        max_score, idx = y_pred.max(1)
        if idx.data[0] == target.data[0]:
            correct += 1

        class_correct[target.data[0]] += 1 if idx.data[0] == target.data[0] else 0
        class_total[target.data[0]] += 1

    accuracy = correct / len(bookmakers_pred_loader)
    score_epoch = np.mean(scores_validation)
    print("Bookmakers score: {:.5f} Bookmakers accuracy: {:.5f}" .format(score_epoch, accuracy))
    print()

    for i in range(3):
        print('Accuracy of %d : %2d / %2d (%2d %%)' % (i, class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))

    print('\n')

    print("Bookmakers predictions overview done.")
    print()
    print("##################################################")

########## Train our model ##########

train_loader = DataLoader(TrainingSet(league=LEAGUE), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

# Create training and validation split
train_data, train_targets, valid_data, valid_targets = train_valid_split(train_loader, VALIDATION_RATIO, SEED)

# Combine train/validation data and targets as tuples
train_data_and_targets = list(zip(train_data, train_targets))
valid_data_and_targets = list(zip(valid_data, valid_targets))

nb_teams = int(train_data[0].data.size(1) / 2)
        
model = NN(nb_teams=nb_teams)

epoch = 0
best_score = (0,0)
history = []
while True:
    
    # Shuffle the training data and targets in the same way
    shuffle(train_data_and_targets)

    # Train the model
    losses_training = []
    for data, target in train_data_and_targets:
        loss = model.step(data, target)
        losses_training.append(loss)

    # Make validation
    scores_validation = []
    correct = 0
    class_correct = list(0 for i in range(3))
    class_total = list(0 for i in range(3))
    for data, target in valid_data_and_targets:
        y_pred = model.predict(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, target)
        score = torch.exp(-loss).data[0]
        scores_validation.append(score)
        max_score, idx = y_pred.max(1)
        for i in range(BATCH_SIZE):
            if idx.data[i] == target.data[i]:
                correct += 1

            class_correct[target.data[i]] += 1 if idx.data[i] == target.data[i] else 0
            class_total[target.data[i]] += 1

    accuracy = correct / (len(valid_data_and_targets) * BATCH_SIZE)
    
    # Mean of the losses of training and validation predictions
    loss_epoch = np.mean(losses_training)
    score_epoch = np.mean(scores_validation)
    history.append((loss_epoch, score_epoch))
    print("Epoch: {} Training loss: {:.5f} Validation score: {:.5f} Accuracy: {:.5f}" .format(epoch, loss_epoch, score_epoch, accuracy))

    # Save the best model
    if score_epoch > best_score[1]:
        best_score = (epoch, score_epoch)

    # Check that the model is not doing worst over the time
    if best_score[0] + MAX_EPOCH_WITHOUT_IMPROV < epoch:
        print("Overfitting. Stopped at epoch {}." .format(epoch))
        break

    epoch += 1 

print("Best score on training set: %.5f at epoch %d" % (best_score[1], best_score[0]))
print()

########## Apply on test set ##########

if TRAINING_SET_RATIO < 1.0:

    test_loader = DataLoader(TestSet(league=LEAGUE), num_workers=4, batch_size=1, shuffle=False)

    print("Applying model on test set and predicting...")

    model.eval()

    scores_validation = []
    correct = 0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    for data, target in test_loader:
        data = Variable(data)
        target = Variable(target)
        y_pred = model.predict(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_pred, target)
        score = torch.exp(-loss).data[0]
        scores_validation.append(score)
        max_score, idx = y_pred.max(1)
        if idx.data[0] == target.data[0]:
            correct += 1

        class_correct[target.data[0]] += 1 if idx.data[0] == target.data[0] else 0
        class_total[target.data[0]] += 1

    accuracy = correct / len(test_loader)
    score_epoch = np.mean(scores_validation)
    print("Test score: {:.5f} Test accuracy: {:.5f}" .format(score_epoch, accuracy))
    print()

    for i in range(3):
        print('Accuracy of %d : %2d / %2d (%2d %%)' % (i, class_correct[i], class_total[i], 100 * class_correct[i] / class_total[i]))

    print('\n')

    print("Predictions done.")

