#!/usr/bin/env python3
# Futures
from __future__ import print_function
########################################
__author__ = "Maria Eduarda Andrada"
__license__ = "GNU GPLv3"
__version__ = "0.2"
__maintainer__ = "Maria Eduarda Andrada"
__email__ = "duda.andrada@isr.uc.pt"
########################################

# STD
import sys
import time
import os
# ROS
import rospy
import roslib
import message_filters # synced callbacks
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ImageMsg # to avoid PIL Image conflict
# numpy and scipy
import numpy as np
from torchvision import transforms
# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError 

class msg_ros():
	def __init__(self):
		# Start node
		rospy.init_node("msc_chn_ros", anonymous=True)
		rospy.loginfo("Started node %s", rospy.get_name())

		# Grab absolute path for the neural network
		abs_path = rospy.get_param("abs_path")
		rospy.loginfo("Abs path is: %s", abs_path)
		sys.path.insert(0, abs_path + '/msg_chn/')

		# Get config file parameteres 
		checkpoint = rospy.get_param("path_model")
		net_path = rospy.get_param("network_path")
		camera_topic = rospy.get_param('camera_topic')
		depth_topic = rospy.get_param('depth_topic')

		# Start neural network, library starts here because of absolute path
		from infer_img import infer_img
		self.inference = infer_img(net_path, checkpoint)
		# Subscribers
		camera_sub = message_filters.Subscriber(camera_topic,
                    ImageMsg, queue_size = 1)
		depth_sub = message_filters.Subscriber(depth_topic,
                    ImageMsg, queue_size = 1)
		ts = message_filters.ApproximateTimeSynchronizer([camera_sub,
                                                          depth_sub], 
                                                          100, 0.1, 
                                                          allow_headerless=True)
		ts.registerCallback(self.callback)
		rospy.loginfo("Read Subscribers %s and %s for %s", depth_sub.name, 
		                                                   camera_sub.name, 
		                                                   rospy.get_name())

		# Start CV Bridge in Python3 -- Only works in Python3
		self.bridge = CvBridge()

		# Publishers
		self.depth_pub = rospy.Publisher('/depth_completion/image_raw', ImageMsg, queue_size=1)
		rospy.loginfo("Created Publisher %s for %s", self.depth_pub.name, 
		                                                 rospy.get_name())


	def callback(self, camera_msg, depth_msg):
		"""
		Camera and depth messages callback. Only works when both images
		are in sync through ros timestamps. Function grabs images
		and runs the neural network which extrapolate missing depth
		information

		Input: RGB/NGR images (uint8) and Depth Images (float32)

		Output: Depth extrapolated image (float32)
		"""
		camera_cv = self.convert_image_cv(camera_msg)
		depth_cv = np.array(self.convert_image_cv(depth_msg).copy())
		depth_cv = (depth_cv*256).astype(np.uint16)

		# Runs neural network MSG-CHN and outputs in meters
		input_d, output, _ = self.inference.evaluate(depth_cv, camera_cv)
		output[output < 0] = 0

		self.publish_image(output, self.depth_pub, depth_msg.header)


	def convert_image_cv(self, ros_image, encoder = "passthrough"):
	    # Convert ROS image to OpenCV
	    try:
	        cv_image = self.bridge.imgmsg_to_cv2(ros_image, encoder)
	        return cv_image
	    except CvBridgeError as e:
	        print(e)

	def publish_image(self, image, publisher, 
	                  camera_header, encoder = "passthrough"):
		# Simple definition to publish image in the publisher chosen
		# Convert OpenCV image to ROS image
		try:
			image_pub = self.bridge.cv2_to_imgmsg(image, encoder)
			image_pub.header.stamp = rospy.Time.now() # Get header from depth 
			image_pub.header.frame_id = camera_header.frame_id
			publisher.publish(image_pub)

		except CvBridgeError as e:
				print (e)    

	def run(self):
	    """
	    Enters the main loop for processing messages.
	    """
	    rospy.spin()

def main():
	node = msg_ros()
	node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        sys.exit(0)