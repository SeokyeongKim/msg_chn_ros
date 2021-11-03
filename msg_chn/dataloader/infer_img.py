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

def infer(self, depth_img, bgr_img, verbose=True, color=True, flip=False):
	    # get sizes
    original_h, original_w, original_d = bgr_img.shape

    # resize
    bgr_img = cv2.resize(bgr_img, (self.data_w, self.data_h),
                         interpolation=cv2.INTER_LINEAR)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    


    # to tensor
    rgb_tensor = torch.from_numpy(rgb_img)
    depth_tensor = torch.from_numpy(depth_img)

    # to gpu
    if self.gpu:
      rgb_tensor = rgb_tensor.cuda()
      depth_tensor = depth_tensor.cuda()

    # permute and normalize
    rgb_tensor = (rgb_tensor.float() / 255.0 - self.means) / self.stds
    rgb_tensor = rgb_tensor.permute(2, 0, 1)

    # add batch dimension
    rgb_tensor = rgb_tensor.unsqueeze(0)

    # gpu?
    with torch.no_grad():
      start = time.time()
      # infer
      logits = self.model(rgb_tensor)

      if flip:
        # flip and infer
        rgb_tensor = rgb_tensor.flip(3)
        logits += self.model(rgb_tensor).flip(3)

      argmax = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
      if self.gpu:
        torch.cuda.synchronize()
      time_to_infer = time.time() - start

      # print time
      if verbose:
        print("Time to infer: ", time_to_infer)
        if flip:
          print("Doing flip-inference")
      # resize to original size
      argmax = cv2.resize(argmax, (original_w, original_h),
                          interpolation=cv2.INTER_NEAREST)

      # color (if I don't want it, just return original image)
      color_mask = bgr_img
      if color:
        color_mask = self.colorizer.do(argmax).astype(np.uint8)

    return argmax, color_mask
