#!/usr/bin/python3

# license removed for brevity
from tkinter import Frame
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import json
from std_msgs.msg import String
import numpy as np
import math
import time

from std_srvs.srv import Empty
#from car_plugin.msg import Response

from time import sleep

#response_topic_name="/automobile/feedback"
class Automobile_Data():
    def __init__(self, reference_topic_name="/control/reference"):
        """

        :param command_topic_name: drive and stop commands are sent on this topic
        :param response_topic_name: "ack" message is received on this topic
        """
        # Publisher node
        rospy.init_node('reference_publisher', anonymous=False)     
        self.publisher = rospy.Publisher('/automobile/command', String, queue_size=1)
        #self.pub = rospy.Publisher(reference_topic_name, String, queue_size=1)

        #camera stuff
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.image_sub = rospy.Subscriber("/automobile/image_raw", Image, self.callback)
        #rospy.spin() # spin() simply keeps python from exiting until this node is stopped

        # Wait for publisher to register to roscore -
        # This is a temporary fix. Problem with timing
        sleep(1)



    def _send_reference(self, speed, angle):
        """Transmite the command to the remotecontrol receiver. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe. 
        """
        data = {}
        data['action']        =  '2'
        data['steerAngle']    =  float(2*angle)
        reference = json.dumps(data)
        self.publisher.publish(reference)

    def callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # cv2.imshow("Frame preview", self.cv_image)
        # key = cv2.waitKey(1)





    