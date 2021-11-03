# Pkg Libraries
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImage import *
from utils.ErrorMetrics import *
from trainers.trainer import Trainer  # from CVLPyDL repo
from dataloader.DataLoaders import *
from modules.losses import *

# Sys
import random
import time
import os, os.path
import sys
import datetime
import subprocess
import importlib
import json
import argparse

# Image libraries
import numpy as np
from PIL import Image
import cv2

# PyTorch Library
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from math import floor

# Fix CUDNN error for non-contiguous inputs
import torch.backends.cudnn as cudnn

class objectview(dict):
    def __init__(self, *args, **kwargs):
        super(objectview, self).__init__(*args, **kwargs)
        self.__dict__ = self

class infer_img(Trainer):
    def __init__(self, path, use_load_checkpoint=None):
        sys.path.append(path)

        try:
            with open(os.path.join(path, 'params.json'), 'r') as fp:
                params = json.load(fp)
            self.params = objectview(params)

        except Exception as e:
          print(e)
          print("Error opening cfg.yaml file from trained model.")
          quit()

        self.device = torch.device("cuda:" + str(self.params.gpu_id) if 
                       torch.cuda.is_available() and self.params.use_gpu else "cpu")


        f = importlib.import_module('network_exp_msg_chn')
        self.model = f.network().to(self.device)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # objective = locals()[params.loss]()
        objective = MSELoss() #Hard coded
        optimizer = getattr(optim, self.params.optimizer)(parameters,
                            lr=self.params.lr,
                            weight_decay=self.params.weight_decay)
        lr_decay = lr_scheduler.StepLR(optimizer, 
                                       step_size=self.params.lr_decay_step, 
                                       gamma=self.params.lr_decay)
        # Call the constructor of the parent class (trainer)
        super(infer_img, self).__init__(self.model, optimizer, lr_decay, 
                                                objective, workspace_dir=use_load_checkpoint, 
                                                use_gpu=self.params.use_gpu)
        for w in self.model.parameters():
          w.requires_grad = False
        self.start_checkpoint(use_load_checkpoint)

        if self.params.use_gpu:
            self.model = self.model.cuda()


    def evaluate(self, input_d, input_rgb):
        # Load last save checkpoint
        # tranform for original shape
        transform = transforms.Compose([transforms.CenterCrop(
                                              (input_d.shape[0], 
                                               input_d.shape[1]))])
        with torch.no_grad():
            torch.cuda.synchronize()
            # Tensorize images before input to neural network
            inputs_d, C, inputs_rgb = self.tensor_imgs(input_d, input_rgb)

            # Device conversion to GPU (if available)
            inputs_d = inputs_d.to(self.device)
            C = C.to(self.device)
            inputs_rgb = inputs_rgb.to(self.device)

            # Run network and output depth completed image
            outputs = self.net(inputs_d, inputs_rgb, self.params.weights_layers)

            # Output conversion to numpy
            outputs = outputs[0]
            im = outputs[0, :, :, :].detach().data.cpu().numpy() # Extract output tensor for depth 
            output_d = np.transpose(im, (1, 2, 0))

            # Transform back to PIL to get original shape instead of multiple of 
            # 16 for the neural network
            output_a = np.array(output_d*256).astype(np.uint16)
            pil_img = Image.fromarray(output_a[:,:,0])
            pil_img = np.array(transform(pil_img)).astype(np.uint16)
            output_a = np.array(pil_img/256).astype(np.float32)

            torch.cuda.empty_cache()

            return self.original_depth(inputs_d), output_a, C

    def original_depth(self, input_depth):
        # Original depth as uint8
        inputs_d_np = input_depth[0,:,:,:].detach().cpu().numpy()
        inputs_d = np.transpose(inputs_d_np, (1, 2, 0)).astype(np.uint8)

        return inputs_d

    def tensor_imgs(self, input_d, rgb_input):
        # Change to ROS
        data = Image.fromarray(input_d)
        rgb =  Image.fromarray(rgb_input)
        # get sizes
        H_multiple_16, W_multiple_16 = self.modify_image_size(rgb)
        self.transform = transforms.Compose([transforms.CenterCrop(
                                              (H_multiple_16, W_multiple_16))])


        # Apply transformations if given
        data = self.transform(data)
        rgb = self.transform(rgb)

        # Convert to numpy
        data = np.array(data, dtype=np.float64)

        # define the certainty
        C = (data > 0).astype(float)
        data = data / 256.0  # [8bits]a

        # Expand dims into Pytorch format
        data = np.expand_dims(data, 0)
        C = np.expand_dims(C, 0)

        data = np.expand_dims(data, 0)
        C = np.expand_dims(C, 0)
        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)

        # Convert RGB image to tensor

        rgb = np.array(rgb, dtype=np.float16)
        rgb /= 255

        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = np.expand_dims(rgb, 0)

        rgb = torch.tensor(rgb, dtype=torch.float)

        return data, C, rgb


    def modify_image_size(self, img):
        # Crop center for all images. They need to have same size for the neural network
        w = 10000
        h = 10000
        # print(type(img))
        W, H = img.size
        if w > W or h > H: # Save lowest sizes 
            w = W 
            h = H
        W_multiple_16 = 16 * floor(w/16) # Find lowest multiple of 16 for Cnn layers
        H_multiple_16 = 16 * floor(h/16)

        return H_multiple_16, W_multiple_16

    def start_checkpoint(self, use_load_checkpoint):
        if use_load_checkpoint != None:
            if isinstance(use_load_checkpoint, int):
                if use_load_checkpoint > 0:
                    print('=> Loading checkpoint {} ...'.format(use_load_checkpoint))
                    if self.load_checkpoint(use_load_checkpoint):
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
                elif use_load_checkpoint == -1:
                    print('=> Loading last checkpoint ...')
                    if self.load_checkpoint():
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
            elif isinstance(use_load_checkpoint, str):
                print('Loading checkpoint from : ' + use_load_checkpoint)
                if self.load_checkpoint(use_load_checkpoint):
                    # print(self.load_checkpoint)
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')



