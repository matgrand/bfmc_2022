#!/usr/bin/python3
from mimetypes import init
import string
from tkinter import Frame
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import json
from std_msgs.msg import String
import numpy as np
from std_srvs.srv import Empty
#from car_plugin.msg import Response
from time import sleep
from helper_functions import *

from LaneKeepingReloaded import LaneKeepingReloaded
from  car_data import Automobile_Data

from generate_standard_route import SimpleController, TrainingTrajectory


# map = cv.imread('src/models_pkg/track/materials/textures/2021_Medium.png')
map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')

if __name__ == '__main__':

    #init windows
    cv.namedWindow("Poly") 
    cv.namedWindow("Frame preview") 
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL)
    
    #yeeeee lets copy
    lane_keeping = LaneKeepingReloaded(640, 480)

    # init trajectory
    trajectory = TrainingTrajectory(nodes_ahead=100, speed=0.1)

    #init controller
    controller = SimpleController(k1=1.0, k2=1.0, k3=1.0)


    # Example: publish a reference every 2 seconds
    angle_ref = 5.0# [deg]
    speed_ref = 15.0# [m/s]
    publish_time = 0.01# [s]


    try:
        # init camera publishe
        car = Automobile_Data()

        while not rospy.is_shutdown():

            # Get the image from the camera
            frame = car.cv_image

            # Get the position of the car
            pos = car.pos
            print(f"IMU:\n {pos}")
            #extract all floats in String pos, using '\n' and ':' as separators
            pos_list = pos.split('\n')
            pos_list = [x.split(':') for x in pos_list]
            x = float(pos_list[-2][1])
            y = float(pos_list[-1][1])
            yaw = float(pos_list[2][1])
            #draw points at the wheels
            tmp = np.copy(map)
            draw_car(tmp, x, y, yaw)
            
            cv.imshow("2D-MAP", tmp)

            #lane keeping
            angle_ref, both_lanes, poly = lane_keeping.lane_keeping_pipeline(frame)

            #show poly_image
            cv.imshow("Poly", poly)
            cv.imshow("Frame preview", frame)
            cv.waitKey(1)

            #car control
            #car._send_reference(speed_ref,angle_ref)
            rospy.sleep(publish_time)
            print("Published reference: speed = %.2f, angle = %.2f" % (speed_ref, angle_ref))
    
    except rospy.ROSInterruptException:
        pass
      