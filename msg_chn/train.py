#!/usr/bin/env python3
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


import os
import sys
import datetime
import subprocess
import importlib
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader.DataLoaders import *
from modules.losses import *

# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn

# Use weights and biases tool
import wandb
if sys.version_info[:2] < (3, 5):
  print ("Error, code runs only on Python 3")
  sys.exit(1)

class objectview(dict):
    def __init__(self, *args, **kwargs):
        super(objectview, self).__init__(*args, **kwargs)
        self.__dict__ = self
# Enable cudnn
cudnn.enabled = True
cudnn.benchmark = True

# Set GPU ID for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.manual_seed(1)

# Parse Arguments
parser = argparse.ArgumentParser("./train.py")
parser.add_argument('-mode', 
                    action='store', 
                    dest='mode', 
                    default='train', 
                    help='"eval" or "train" mode. '
                    'Defaults at %(default)s',
                    )
parser.add_argument('-exp', 
                    action='store', 
                    dest='exp', 
                    default='exp_msg_chn',
                    help='Experiment name as in workspace directory'
                    'Defaults at %(default)s',
                    )
parser.add_argument('-m', '--model', 
                    action='store', 
                    dest='chkpt', 
                    default=None,  
                    nargs='?',   # None or number
                    help='Checkpoint number to load'
                    'Defaults at %(default)s',
                    )

parser.add_argument('-w', '--wandb', 
                    action='store_true', 
                    dest='useWandb', 
                    default = False, 
                    help= 'Start weights and biases analysis'
                    'Defaults at %(default)s',
                    )
parser.add_argument('-n', 
                    type = str,
                    action = 'store',
                    dest='runName', 
                    default = "", 
                    help= 'Name wandb run for clearer logging online. '
                    'Defaults at %(default)s',
                    )

parser.add_argument('-d', '--debug', 
                    action='store_true', 
                    dest='debug', 
                    default = False, 
                    help= 'Verbose for debugging. '
                    'Defaults at %(default)s',
                    )
parser.add_argument('--log', '-l',
                    type=str,
                    dest='logs',
                    default = datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
                    help='Directory to put the log data. Default:'
                    ' workspace/-exp/logs/date+time'
                    )

args = parser.parse_args()


# Path to the workspace directory
training_ws_path = 'workspace/'
exp = args.exp
exp_dir = os.path.join(training_ws_path, exp)
if '/home' in args.logs:
    log_dir = args.logs
else: 
    log_dir = exp_dir + '/logs/' + args.logs

try:
    os.mkdir(log_dir)
except OSError:
    print ("Creation of the directory %s failed,"
                        " already exists" 
                        % log_dir)
else:
    print ("Successfully created the directory %s "
                         % log_dir)
# Add the experiment's folder to python path
sys.path.append(exp_dir)

# Read parameters file
with open(os.path.join(exp_dir, 'params.json'), 'r') as fp:
    params = json.load(fp)

params['mode'] = args.mode
# print(args.)
if args.useWandb:
    print("Using Weights & Biases!")
    wandb.init(project = "semfire_depth",
               entity = "semfire",
               name = args.runName)
    wandb.config.update(params)
    params = wandb.config
else:
    params = objectview(params)

device = torch.device("cuda:" + str(params.gpu_id) if 
             torch.cuda.is_available() and params.use_gpu else "cpu")

# Dataloader
data_loader = (params.data_loader if 'data_loader' in params else 
             'KittiDataLoader')
dataloaders, dataset_sizes = eval(data_loader)(params, args.debug)

# Import the network file
f = importlib.import_module('network_' + exp)
model = f.network().to(device)
if params.use_gpu:
    model = model.cuda()

# Print summary of arguments 
print("*" * 60)
print(" " * 25, "INTERFACE")
print("*" * 60)
print("CUDA is available: ", torch.cuda.is_available())
print("Mode: ", args.mode)
print("Experiment name: ", args.exp)
print("Pre trained model/ checkpoint: ", args.chkpt)
print("Weights and Biases analysis: ", args.useWandb)
print("Weights and Biases runtime name: ", args.runName)
print("Device chosen: ", device)
print("Commit hash (training version): ", str(
  subprocess.check_output(['git', 'rev-parse', 
                           '--short', 'HEAD']).strip()))

# Number of parameters used for training
weights_rgb_enc = sum(p.numel()
                  for p in model.rgb_encoder.parameters())
weights_depth_enc = (sum(p.numel()
                         for p in model.depth_encoder1.parameters())
                  + sum(p.numel()
                          for p in model.depth_encoder2.parameters())
                  + sum(p.numel()
                          for p in model.depth_encoder3.parameters())
                  )
weights_dec = (sum(p.numel()
                  for p in model.depth_decoder1.parameters())
            + sum(p.numel()
                  for p in model.depth_decoder2.parameters())
            + sum(p.numel()
                  for p in model.depth_decoder3.parameters())
            )
total_weight = weights_rgb_enc + weights_depth_enc + weights_dec

# Print values
if args.debug:
    print("*" * 60)
    print(" " * 18, "Number of Parameters")
    print("*" * 60)
    print("Number of Parameters for the RGB Encoder: ",
       '{:,.0f}'.format(weights_rgb_enc))
    print("Number of Parameters for the Depth Encoder: ",  
       '{:,.0f}'.format(weights_depth_enc))
    print("Number of Parameters for the Decoder:", 
          '{:,.0f}'.format(weights_dec), "\n")
    print("Total number of parameters: ", '{:,.0f}'.format(total_weight))
    weights_grad = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)
    print("Total number of parameters that requires_grad: ", 
              '{:,.0f}'.format(weights_grad), '\n')

# Import the trainer
t = importlib.import_module('trainers.' + params.trainer)

if args.mode == 'train':
    mode = 'train'  # train    eval
    sets = ['train', 'val', 'test']  # train  selval
elif args.mode == 'eval':
    mode = 'eval'  # train    eval
    sets = ['test', 'val']  # train  selval

# Objective function
objective = locals()[params.loss]()

# Optimize only parameters that requires_grad
parameters = filter(lambda p: p.requires_grad, model.parameters())

# The optimizer
optimizer = getattr(optim, params.optimizer)(parameters,
                    lr=params.lr,
                    weight_decay=params.weight_decay)

# Decay LR by a factor of 0.1 every exp_dir7 epochs
lr_decay = lr_scheduler.StepLR(optimizer, 
                               step_size=params.lr_decay_step, 
                               gamma=params.lr_decay)

mytrainer = t.KittiDepthTrainer(model, params, optimizer, objective, 
                                lr_decay, dataloaders, dataset_sizes, 
                                debug = args.debug, workspace_dir=log_dir, 
                                sets=sets, device = device,
                                use_load_checkpoint=args.chkpt, weights=params.weights_layers,
                                useWandb = args.useWandb, runName = args.runName)

if mode == 'train':
    # train the network
    net = mytrainer.train()  #
else:
    net = mytrainer.evaluate()





