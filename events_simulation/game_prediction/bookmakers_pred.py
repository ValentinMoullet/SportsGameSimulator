from tqdm import tqdm
tqdm.monitor_interval = 0

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from loader import BookmakersPred
from utils import *
from parameters import *


NB_SAMPLES = 1000

scores_bookmakers = []
losses_bookmakers = []
losses_bookmakers_sampled = []
accuracy = 0
accuracy_sampled = 0
total = 0
for league in ALL_LEAGUES:
    bookmakers_pred_loader = DataLoader(BookmakersPred(league=league, test_only=True), num_workers=4, shuffle=True)

    correct = 0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    for y_pred, target in bookmakers_pred_loader:
        y_pred = Variable(y_pred)
        target = Variable(target)

        y_pred *= 1 / torch.sum(y_pred)

        #print(y_pred)

        loss = F.cross_entropy(y_pred, target)
        score = torch.exp(-loss).item()

        y_sampled = torch.FloatTensor([[0, 0, 0]])
        for _ in range(NB_SAMPLES):
            gen = int(torch.multinomial(y_pred, 1)[0])
            #print(gen)
            y_sampled[0, gen] += 1

        y_sampled /= NB_SAMPLES

        #print(y_sampled)
        loss_sampled = F.cross_entropy(y_sampled, target)
        accuracy_sampled += y_sampled[0][target.data[0]]

        scores_bookmakers.append(score)
        losses_bookmakers.append(loss.item())
        losses_bookmakers_sampled.append(loss_sampled)

        accuracy += y_pred.data[0][target.data[0]]

        '''
        max_score, idx = y_pred.max(1)
        if idx.data[0] == target.data[0]:
            correct += 1
        
        class_correct[target.data[0]] += 1 if idx.data[0] == target.data[0] else 0
        class_total[target.data[0]] += 1
        '''

    total += len(bookmakers_pred_loader)

accuracy /= total
score_bookmakers = np.mean(scores_bookmakers)
loss_bookmakers = np.mean(losses_bookmakers)

accuracy_sampled /= total
loss_bookmakers_sampled = np.mean(losses_bookmakers_sampled)

print("Bookmakers loss:", loss_bookmakers)
print("Bookmakers score:", score_bookmakers)
print("Accuracy:", accuracy)
print()
print("Bookmakers loss sampled:", loss_bookmakers_sampled)
print("Accuracy sampled:", accuracy_sampled)
print()