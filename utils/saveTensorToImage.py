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
import os
import sys
import cv2
import numpy as np
import wandb

# This function takes a 4D tensor in the form NxCxWxH and save it to images according to idxs
class saveTensorToImage():
    def __init__(self, path, debug = False):
        self.img_dir = self.create_dir(path)
        self.debug = debug


    def do(self, outputs, input_d, inputs_rgb, name_images, labels):
        images = []
        names = []
        random_imgs = []
        a = 0
        custom_name = ['output_', 'output_colored_', 'bottom_colored_', 'input_colored_', 'overlaid_img_']
        for output in range(outputs.size(0)):
            name = name_images[output]
            im = outputs[output, :, :, :].detach().data.cpu().numpy() # Extract output tensor for depth 
            output_d = np.transpose(im, (1, 2, 0))

            im = labels[output, :, :, :].detach().data.cpu().numpy() # Extract output tensor for depth 
            gt = np.transpose(im, (1, 2, 0))
            # Eliminate any negative value for visualization purposes
            output_d[output_d < 0] = 0
            output_d = output_d.astype(np.uint8) # Transpose for common image shape

            input_depth = self.original_depth(input_d) # Get original image for normalization
            input_depth_normal, output_depth_normal, gt_normal = self.normalize_img(input_depth, output_d, gt)
            _, output_depth_bottom_normal, _ = self.normalize_img(input_depth,
                                                                          output_d,
                                                                          gt,
                                                                          True)
            # overlaid_img = self.overlay_input_output(input_depth_normal, output_depth_normal)

            if self.debug:

                print("Original output depth max value: ",   output_d.max(), 
                                                             output_d.min())
                print("Original input depth max value: ",    input_depth.max(), 
                                                             input_depth.min())
                print("Original gt depth max value: ",       gt.max(), 
                                                             gt.min())
                print("Normalized output depth max value: ", output_depth_normal.max(), 
                                                             output_depth_normal.min())
                print("Normalized input depth max value: ",  input_depth_normal.max(), 
                                                             input_depth_normal.min())
                print("Normalized input gt max value: ",     gt_normal.max(), 
                                                             gt_normal.min())


            # Color images
            gt_colored = self.color_image(gt_normal)
            output_colored = self.color_image(output_depth_normal)
            input_colored = self.color_image(input_depth_normal, True)
            output_bottom_colored = self.color_image(output_depth_bottom_normal)
            # overlaid_colored = self.color_image(overlaid_img)

            # Append results to save images
            results_save = [output_d, 
                            output_colored, 
                            output_bottom_colored, 
                            input_colored] 
                            # overlaid_colored]

            results = [wandb.Image(self.invert_colors(gt_colored)), 
                       wandb.Image(self.invert_colors(output_colored)), 
                       wandb.Image(self.invert_colors(input_colored)), 
                       wandb.Image(self.invert_colors(inputs_rgb[output]))]

            images.append(results_save)
            names.append(name)
            random_imgs.append(results)

        # Save image in the log directory
        for i, imgs in enumerate(images):
            for j, img in enumerate(imgs):
                self.save_image(custom_name[j], name_images[i], img)

        return random_imgs



    def original_depth(self, input_depth):
        # Original depth as uint8
        inputs_d_np = input_depth[0,:,:,:].detach().cpu().numpy()
        inputs_d = np.transpose(inputs_d_np, (1, 2, 0)).astype(np.uint8)

        return inputs_d

    def overlay_input_output (self, inputs_d_normal, output_d_normal):

        _, mask = cv2.threshold(output_d_normal, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        overlaid_img = cv2.bitwise_and(output_d_normal, output_d_normal, mask=mask_inv)
        return cv2.add(overlaid_img, inputs_d_normal)

    def color_image(self, img, dilate = False):
        """
        Color normalized image with inverted jet (red means near, blue means far).
        """
        #Change colors
        if dilate:
            kernel = np.ones((4,4), np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)
        img_colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img_colored[:, :, [0, 2]] = img_colored[:, :, [2, 0]]
        img_colored[np.where((img_colored == [0,0,128]).all(axis = 2))] = [0,0,0]
        return img_colored


    def invert_colors(self, img):

        img[:, :, [0, 2]] = img[:, :, [2, 0]]

        return img

    def normalize_img(self, input_depth, output_depth, gt, bottom_half = False):
        """ 
        Normalize image to 0-255 for coloring using original/output depth as 
        reference to have the same color palette. For visualization only.
        """
        if (bottom_half):
            # Use only 2/3 of the images for SEMFIRE case. Visualization only!
            w, h, c = output_depth.shape
            y = int(h/3.5) 
            if self.debug:
                print("Shape input depth: ", input_depth.shape)
                print("Shape gt: ", gt.shape)
                print("Shape output depth: ", output_depth.shape)
                print('*' * 60)
                print("Before mask max/min values (output): ", output_depth.max(), output_depth.min())
                print("Before mask max/min values (input): ", input_depth.max(), input_depth.min())
                print("Before mask max/min values (gt)", gt.max(), gt.min())
            # Create mask and eliminate any value which does not belong to the mask
            mask = np.zeros((w,h), np.uint8)
            mask = cv2.rectangle(mask, (0,y), (h,w), (255), cv2.FILLED)
            output_depth = cv2.bitwise_and(output_depth, mask)
            input_depth = cv2.bitwise_and(input_depth, mask)
            gt = cv2.bitwise_and(gt.astype(np.uint8), mask)
            if self.debug:
                print("After bottom mask max/min values (output): ", output_depth.max(), output_depth.min())
                print("After bottom mask max/min values (input) ", input_depth.max(), input_depth.min())
                print("After bottom mask max/min values (gt) ", gt.max(), gt.min())


        if  gt.max() > max(input_depth.max(), output_depth.max()):
            inputs_d_normal = self.norm_calc(input_depth, gt)
            output_d_normal = self.norm_calc(output_depth, gt)
            gt_normal = self.norm_calc(gt, gt)

        else:
            if output_depth.max() > input_depth.max():
                inputs_d_normal = self.norm_calc(input_depth, output_depth)
                output_d_normal = self.norm_calc(output_depth, output_depth)
                gt_normal = self.norm_calc(gt, output_depth)

            else:
                inputs_d_normal = self.norm_calc(input_depth, input_depth)
                output_d_normal = self.norm_calc(output_depth, input_depth)
                gt_normal = self.norm_calc(gt, input_depth)


        return inputs_d_normal.astype(np.uint8), output_d_normal.astype(np.uint8), gt_normal.astype(np.uint8)

    def norm_calc(self, num, dem):
        return (num/dem.max())*255        

    def save_image(self, custom_name, orig_name, img):
        cv2.imwrite(self.img_dir + custom_name + orig_name + '.png', img)
    
    def create_dir(self, path):
        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory %s failed,"
                                " already exists\n" 
                                % path)
        else:
            print ("Successfully created the directory %s \n"
                                 % path)
        return path

