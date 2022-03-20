#!/usr/bin/python3
# Functional libraries
import rospy
import numpy as np
from time import sleep, time
from math import cos,sin,pi
from cv_bridge import CvBridge
import json
from helper_functions import *

# Messages for topic and service communication
from std_msgs.msg import String
from utils.msg import IMU
from utils.msg import localisation
#from utils.srv import subscribing
from sensor_msgs.msg import Image
import os


def camera_callback(data):
    global bridge
    global image_data
    image_data = bridge.imgmsg_to_cv2(data, "bgr8")
    #print(image_data)


#main
if __name__ == '__main__':


    #define bridge
    bridge = CvBridge()
    image_data = np.zeros((480,640,3), np.uint8)

    rospy.Subscriber("/automobile/image_raw", Image, camera_callback)

    while not rospy.is_shutdown():
        if image_data is not None:
            print(image_data)
            cv.imshow("image", image_data)
            key = cv.waitKey(1)
            if key == 27:
                break

