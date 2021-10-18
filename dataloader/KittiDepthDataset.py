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

from PIL import Image
import os
import torch
import numpy as np
import glob
import torchvision
import random
import time
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision import transforms
from math import floor
class KittiDepthDataset(Dataset):
    def __init__(self, data_path, gt_path, setname='train', use_transform=False, norm_factor=256, invert_depth=False,
                 rgb_dir=None, rgb2gray=False, fill_depth = False, flip = False, blind = False, debug = False):
        self.data_path = data_path
        self.gt_path = gt_path
        self.setname = setname
        self.use_transform = use_transform
        self.norm_factor = norm_factor
        self.invert_depth = invert_depth
        self.rgb_dir = rgb_dir
        self.rgb2gray = rgb2gray
        self.fill_depth = fill_depth
        self.flip = flip
        self.blind = blind
        self.debug = debug
        self.data = list(sorted(glob.iglob(self.data_path + "/*.png", recursive=True)))
        self.gt = list(sorted(glob.iglob(self.gt_path + "/*.png", recursive=True)))
        self.rgb = list(sorted(glob.iglob(self.rgb_dir + "/*.jpg", recursive=True)))
        if self.debug:
            self.print_directories()
        H_multiple_16, W_multiple_16 = self.modify_image_size()
        self.transform = transforms.Compose([transforms.CenterCrop((H_multiple_16, W_multiple_16))])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

  
        data = Image.open(str(self.data[item]))
        gt = Image.open(str(self.gt[item]))

        # Get image name to save results
        name_image = os.path.basename(self.data[item])
        name_image = name_image.split('.', 2)[0]

        # Read RGB images
        rgb_path = self.rgb[item]

        rgb = Image.open(str(rgb_path))

        if self.rgb2gray:
            t = torchvision.transforms.Grayscale(1)
            rgb = t(rgb)

        W, H = data.size

        # Apply transformations if given
        if self.use_transform:
            data = self.transform(data)
            gt = self.transform(gt)
            rgb = self.transform(rgb)

        if not self.use_transform and self.setname == 'train':
            crop_lt_u = random.randint(0, W - 720)
            crop_lt_v = random.randint(0, H - 528)
            data = data.crop((crop_lt_u, crop_lt_v, crop_lt_u+720, crop_lt_v+ 528))
            gt = gt.crop((crop_lt_u, crop_lt_v, crop_lt_u + 720, crop_lt_v + 528))
            rgb = rgb.crop((crop_lt_u, crop_lt_v, crop_lt_u + 720, crop_lt_v + 528))


        if self.flip and random.randint(0, 1) and self.setname == 'train':
            data = data.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to numpy
        data = np.array(data, dtype=np.float64)
        gt = np.array(gt, dtype=np.float16)



        # blind
        if self.blind and (self.setname == 'train'):
            blind_start = random.randint(100, H - 50)
            data[blind_start:blind_start+50, :] = 0

        # define the certainty
        C = (data > 0).astype(float)
        data = data / self.norm_factor  # [8bits]a
        gt = gt / self.norm_factor

        # Expand dims into Pytorch format
        data = np.expand_dims(data, 0)
        gt = np.expand_dims(gt, 0)
        C = np.expand_dims(C, 0)

        # Convert to Pytorch Tensors
        data = torch.tensor(data, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)

        # Convert depth to disparity
        if self.invert_depth:
            data[data == 0] = -1
            data = 1 / data
            data[data == -1] = 0

            gt[gt == 0] = -1
            gt = 1 / gt
            gt[gt == -1] = 0

        # Convert RGB image to tensor

        rgb = np.array(rgb, dtype=np.float16)
        rgb /= 255

        if self.rgb2gray:
            rgb = np.expand_dims(rgb, 0)
        else:
            rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.tensor(rgb, dtype=torch.float)

        c, w, h = rgb.size()

        """
        Figure it out if this is needed later. Objective of this code:
        prints the percentage of depth points and save image with specific number for
        ablation study when changing the weights of rgb or depth encoder
        """ 
        # print (self.data[item])
        # if '150.png' in str(self.rgb[item]):
        #     print('\n\n')
            # print("*"*60)
            # print("Number of depth points in the image: ", torch.count_nonzero(data))
            # print("Number of depth points in the grount truth: ", torch.count_nonzero(gt))
            # print("Number of points in the rgb: ", torch.count_nonzero(rgb))
            # print("Percentage gt points in the area: ", ((torch.count_nonzero(gt)/(w*h))*100), '%')
            # print("Percentage depth points in the area: ", ((torch.count_nonzero(data)/(w*h))*100), '%')
            # print("Percentage rgb points in the area: ", ((torch.count_nonzero(rgb)/(w*h*c))*100), '%')
            # print("size tensor: ", data.size(), gt.size(), rgb.size())
            # flag_save_image = True

        return data, C, gt, rgb, name_image

    def modify_image_size(self):
        # Crop center for all images. They need to have same size for the neural network
        w = 10000
        h = 10000
        for img in self.data:
            img = Image.open(str(img))
            W, H = img.size
            if w > W or h > H: # Save lowest sizes 
                w = W 
                h = H
        W_multiple_16 = 16 * floor(w/16) # Find lowest multiple of 16 for Cnn layers
        H_multiple_16 = 16 * floor(h/16)
        if  self.debug:
            print("Original image size: ", w, h)
            print("Modified image size: ", W_multiple_16, H_multiple_16, '\n')
            
        return H_multiple_16, W_multiple_16

    def print_directories(self):
        assert (len(self.gt) == len(self.data) == len(self.rgb))
        print("*" * 60)
        print(" " * 20, "Dataset", self.setname)
        print("*" * 60)
        print("Depth Directory: ", self.data_path)
        print("RGB Directory: ", self.rgb_dir)
        print("Label Directory: ", self.gt_path)
        print("No of depth images: ", len(self.data))
        print("No of rgb images: ", len(self.rgb))
        print("No of labeled images: ", len(self.gt))