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
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.KittiDepthDataset import KittiDepthDataset
import random
import glob
num_worker = 8

def KittiDataLoader(params, debug = False):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    norm_factor = params.data_normalize_factor
    invert_depth = params.invert_depth
    ds_dir = params.dataset_dir
    rgb_dir = params.rgb_dir if 'rgb_dir' in params else None
    rgb2gray = params.rgb2gray if 'rgb2gray' in params else False
    fill_depth = params.fill_depth if 'fill_depth' in params else False
    flip = params.flip if ('flip' in params) else False
    dataset = params.dataset if 'dataset' in params else 'KittiDepthDataset'
    num_worker = 8

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    ###### Training Set ######
    train_data_path = os.path.join(ds_dir, 'train/img')
    train_gt_path = os.path.join(ds_dir, 'train/lbl')
    train_rgb_path = os.path.join(ds_dir, 'train/rgb')

    if params.transform_type == 'center':
        train_transform = True 

    else:
        train_transform = False

    image_datasets['train'] = eval(dataset)(train_data_path, train_gt_path, setname='train',
                                                use_transform=train_transform, norm_factor=norm_factor,
                                                invert_depth=invert_depth, rgb_dir=train_rgb_path, 
                                                rgb2gray=rgb2gray, fill_depth=fill_depth, 
                                                flip=flip, debug = debug)
    # Select the desired number of images from the training set
    if params.train_on != 'full':
        image_datasets['train'].data = image_datasets['train'].data[0:params.train_on]  # file directions
        image_datasets['train'].gt = image_datasets['train'].gt[0:params.train_on]

    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params.train_batch_size,
                                      num_workers=num_worker)
    dataset_sizes['train'] = {len(image_datasets['train'])}


    ###### Validation Set ######
    val_data_path = os.path.join(ds_dir, 'valid/img')
    val_gt_path = os.path.join(ds_dir, 'valid/lbl')
    val_rgb_path = os.path.join(ds_dir, 'valid/rgb')


    val_transform = True 


    image_datasets['val'] = eval(dataset)(val_data_path, val_gt_path, setname='val', 
                                          use_transform = val_transform, norm_factor=norm_factor, 
                                          invert_depth=invert_depth, rgb_dir=val_rgb_path, 
                                          rgb2gray=rgb2gray, fill_depth=fill_depth, 
                                          flip=flip, debug = debug)
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params.val_batch_size,
                                    num_workers=num_worker)
    dataset_sizes['val'] = {len(image_datasets['val'])}


    # ###### Test set ######
    test_data_path = os.path.join(ds_dir, 'test/img')
    test_gt_path = os.path.join(ds_dir, 'test/lbl')
    test_rgb_path = os.path.join(ds_dir, 'test/rgb')


    image_datasets['test'] = eval(dataset)(test_data_path, test_gt_path, setname='test', use_transform=val_transform,
                                           norm_factor=norm_factor, invert_depth=invert_depth,
                                           rgb_dir=test_rgb_path, rgb2gray=rgb2gray, fill_depth=fill_depth,
                                           debug = debug)

    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=False, batch_size=params.test_batch_size,
                                     num_workers=num_worker)
    dataset_sizes['test'] = {len(image_datasets['test'])}


    return dataloaders, dataset_sizes







