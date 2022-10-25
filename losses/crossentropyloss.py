import torch
import torch.nn as nn
class CrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.config = config.get('args',None) #config['loss']['args']
        if self.config is not None:
            self.loss = nn.CrossEntropyLoss(self.config)
        else:
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input, target):
        return self.loss(input, target)