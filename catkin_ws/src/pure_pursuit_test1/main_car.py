#!/usr/bin/python3
from mimetypes import init
import string
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
import os

from  automobile_data import Automobile_Data
from helper_functions import *
from simple_controller2 import SimpleController

# PARAMETERS
sample_time = 0.01 # [s]
max_angle = 30.0    # [deg]
max_speed = 0.5  # [m/s]
desired_speed = 0.07 # [m/s]
# CONTROLLER
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 1.0 #1.5 perpedndicular error gain
k3 = 1.5 #1.5 yaw error gain
#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.2 # feedforward gain

if __name__ == '__main__':

    # init the car data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True)

    # init controller
    controller = SimpleController(k1=k1, k2=k2, k3=k3, ff=ff_curvature)

    start_time_stamp = 0.0

    car.stop()

    print('Stopped')
    
    sleep(1)

    car.drive_speed(speed=desired_speed)

    try:
        while not rospy.is_shutdown():
            # Get the image from the camera
            frame = car.cv_image.copy()

            speed_ref, angle_ref, net_out = controller.get_nn_control(frame, desired_speed)

            car.drive_angle(angle=np.rad2deg(angle_ref))

            #prints
            # clear stdout for a smoother display
            os.system('cls' if os.name=='nt' else 'clear')
            # print(f"Coeffs:\n{coeffs}")
            print(f'Net out:\n {net_out}')
     
            cv.imshow("Frame preview", frame)
            cv.waitKey(1)
            sleep(0.01)

    except rospy.ROSInterruptException:
        pass
      