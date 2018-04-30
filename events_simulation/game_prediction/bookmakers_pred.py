from tqdm import tqdm
tqdm.monitor_interval = 0

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from loader import BookmakersPred
from utils import *
from parameters import *


scores_bookmakers = []
losses_bookmakers = []
accuracy = 0
total = 0
for league in ALL_LEAGUES:
    bookmakers_pred_loader = DataLoader(BookmakersPred(league=league), num_workers=4, shuffle=True)

    correct = 0
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

print("Bookmakers loss:", loss_bookmakers)
print("Bookmakers score:", score_bookmakers)
print("Accuracy:", accuracy)
print()