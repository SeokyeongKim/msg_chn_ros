#!/usr/bin/env python3

# STD
import sys
import time
import os
# ROS
import rospy
import roslib
import message_filters # synced callbacks
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image # to avoid PIL Image conflict
# numpy and scipy
import numpy as np
from torchvision import transforms
# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError


class depth_colored():
    def __init__(self):
        # Initialize Node
        node = "msg_chn_colored"
        rospy.init_node(node, anonymous=True)
        rospy.loginfo("Started node %s", rospy.get_name())

        # Create Ros-OpenCv Bridge
        self.bridge = CvBridge()

        # Get normalization factor
        self.norm_value = rospy.get_param('norm_value', 30.)

        # Get topics to subscribe
        camera_topic = rospy.get_param('/camera_topic',
                                       'dalsa_camera_720p')
        
        # Subscribers
        depth_sub = message_filters.Subscriber('/depth', Image, queue_size=10)
        camera_sub = message_filters.Subscriber(camera_topic, Image, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([depth_sub,
                                                          camera_sub], 
                                                          10, 0.1, 
                                                          allow_headerless=True)
        ts.registerCallback(self.pcl_caminfo_callback)
        rospy.loginfo("Read Subscribers %s and %s for %s", depth_sub.name, 
                                                           camera_sub.name, 
                                                           rospy.get_name())

        # Publisher
        self.depth_pub_colored = rospy.Publisher('/depth_completion/colored/image_raw', 
                                         Image, queue_size=10)

        self.overlaid_depth = rospy.Publisher('/depth_completion/overlaid/image_raw', 
                                         Image, queue_size=10)
        rospy.loginfo("Created Publishers %s  and %s for %s", self.depth_pub_colored.name, 
                                                              self.overlaid_depth.name,
                                                              rospy.get_name())


    def pcl_caminfo_callback(self, depth_img, dalsa):
        """
        Camera and depth messages callback. Only works when both images
        are in sync through ros timestamps. Function grabs images
        and color depth according to a normalization factor from yaml config.
        It overlays the colored depth with the original rgb image

        Input: RGB/NGR images (uint8) and Depth Images (float32)

        Output: Colored and Dilated Depth (uint8) and overlaid depth/image (uint8)
        """
        # Get dalsa to overlay img
        cv_dalsa = self.convert_image_cv(dalsa)

        # Get depth to color it
        cv_depth = self.convert_image_cv(depth_img)

        #Normalize depth according to norm value
        output_normal = self.normalize_img(cv_depth, 
                                           norm_value = self.norm_value)

        # Color image and overlay over camera image
        depth_colored = self.color_image(output_normal)
        overlay_img = cv2.addWeighted(cv_dalsa, 0.5, depth_colored, 0.5, 0)

        # Publish images
        self.publish_image(depth_colored, self.depth_pub_colored, depth_img.header)
        self.publish_image(overlay_img, self.overlaid_depth, depth_img.header)


    def publish_image(self, image, publisher, header, encoder = "passthrough"):
        # Simple definition to publish image in the publisher chosen
        # Convert OpenCV image to ROS image
        try:

            image_pub = self.bridge.cv2_to_imgmsg(image, encoder)
            image_pub.header.stamp = rospy.Time.now()  
            image_pub.header.frame_id = header.frame_id
            publisher.publish(image_pub)

        except CvBridgeError as e:
                print (e)

    def normalize_img(self, image_cv, bottom_half = False, norm_value = 30.):
        """ 
        Normalize image to 0-255 for coloring using original/output depth as 
        reference to have the same color palette. For visualization only.

        Bottom half flag for semantic localization

        Input: np array image (in meters)

        Output: np array image (uint8)
        """
        if (bottom_half):
            # Use only 2/3 of the images for SEMFIRE case. Visualization only!
            w, h, c = image_cv.shape
            y = int(h/3.5) 

            # Create mask and eliminate any value which does not belong to the mask
            mask = np.zeros((w,h, 1), np.uint8)
            mask = cv2.rectangle(mask, (0,y), (h,w), (255), cv2.FILLED)
            image_cv = cv2.bitwise_and(image_cv, mask)

        image_normal = self.norm_calc(image_cv, norm_value)
        image_normal[image_normal > 255] = 255

        return image_normal.astype(np.uint8)

    def norm_calc(self, num, dem):
        return (num/dem)*255    

    def color_image(self, image, dilate = False):
        """
        Color image with inversed jet (red near, blue far). Dilate img if necessary.
        Mainly used for very sparse depth points to be visible for papers and rviz

        Input: np array image mono(uint8)

        Output: np array image colored (uint8)
        """
        if dilate:
            kernel = np.ones((4,4), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)

        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        b, g, r = cv2.split(image)
        image = cv2.merge((r, g, b))
        # clean all points not coming from the lidar to black
        image[np.where((image == [0,0,128]).all(axis = 2))] = [0,0,0]

        return image

    
    def convert_image_cv(self, ros_image):
        # Convert ROS image to OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "passthrough")
            return cv_image
        except CvBridgeError as e:
            print(e)

    def run(self):
        """
        Enters the main loop for processing messages.
        """
        rospy.spin()
    
def main():
  # Run new class
  d_map = depth_colored()
  # Spin until ctrl + c
  d_map.run()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        sys.exit(0)