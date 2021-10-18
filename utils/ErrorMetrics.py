"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics import MeanSquaredError, MeanAbsoluteError

class iMAE(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(iMAE, self).__init__()
        self.mae = MeanAbsoluteError().to(device)

    def forward(self, outputs, target, *args):
        outputs = outputs / 1000.
        target = target / 1000.
        outputs[outputs == 0] = -1
        target[target == 0] = -1
        outputs = 1. / outputs
        target = 1. / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        return self.mae(outputs, target)

class MAE(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(MAE, self).__init__()
        self.mae = MeanAbsoluteError().to(device)

    def forward(self, outputs, target, *args): 
        return self.mae(outputs, target) * 1000

class RMSE(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(RMSE, self).__init__()
        self.rmse = MeanSquaredError(squared = False).to(device)

    def forward(self, outputs, target, *args):
        return self.rmse(outputs, target) * 1000 

class iRMSE(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(iRMSE, self).__init__()
        self.rmse = MeanSquaredError(squared = False).to(device)

    def forward(self, outputs, target, *args):

        outputs = outputs / 1000.
        target = target / 1000.
        outputs[outputs==0] = -1
        target[target==0] = -1
        outputs = 1. / outputs
        target = 1. / target
        outputs[outputs == -1] = 0
        target[target == -1] = 0
        return self.rmse(outputs, target) 
