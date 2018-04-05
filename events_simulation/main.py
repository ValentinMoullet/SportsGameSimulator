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

from loader import TrainingSet
from parameters import *
from model import *
from plot import *


train_loader = DataLoader(TrainingSet(), num_workers=4, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMEvents(32, NB_EVENT_TYPES * 2 + 2, 3, batch_size=BATCH_SIZE)

training_loss_history = []
for epoch in tqdm(range(MAX_EPOCH)):

    ########## Train the model ##########

    model.hidden = model.init_hidden()

    losses_training = []
    for data, target in train_loader:
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

loss_histories = (
    ('Training loss', training_loss_history),)
plot_history(loss_histories, 'loss.pdf', "Training loss")

