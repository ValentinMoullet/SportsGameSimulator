import torch
import torch.nn as nn
from torch.autograd import Variable


class WeightedCrossEntropyLoss(nn.Module):
    
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss,self).__init__()
        self.weight = weight
        
    def forward(self, x, y):
        
        loss = Variable(torch.zeros(1))
        for i in range(len(y)):
            if self.weight is None:
                loss += -x[i][y[i]] + torch.log(torch.sum(torch.exp(x[i])))
            else:
                loss += Variable(self.weight)[y[i]] * (-x[i][y[i]] + torch.log(torch.sum(torch.exp(x[i]) * Variable(self.weight))))
        
        #loss = torch.sum(x[:, y] + torch.log(torch.sum(torch.exp(x))))

        #print(loss)
        return loss