#!/usr/bin/python3
from operator import truediv
import cv2 as cv
import rospy
import numpy as np
from time import sleep
from time import time
import os

from automobile_data import Automobile_Data
from helper_functions import *
from controller3 import Controller
from detection import Detection

# PARAMETERS
sample_time = 0.01 # [s]
max_angle = 30.0    # [deg]
max_speed = 0.5  # [m/s]
desired_speed = 0.0 # [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 3.0 #2.0 perpedndicular error gain
k3 = 1. #1.5 yaw error gain #inversely proportional
#dt_ahea  = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain


if __name__ == '__main__':

    # init the car data
    car = Automobile_Data(trig_cam=True, trig_gps=False, 
                            trig_bno=True, trig_enc=True, trig_control=True, 
                            trig_estimation=False)
    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature)

    #init detection
    detect = Detection(trafficlight=True, sign=True, obstacle=False, show_ROI=True)

    start_time_stamp = 0.0

    car.stop()
    car.drive_angle(0.0)

    print("Starting...")

    sleep(1.5)

    car.drive_speed(speed=desired_speed)

    #tests
    previous_e3 = 0.0

    try:
        going_forward = True
        backward_timer_start = 0.0

        last_sign_detected = None

        while not rospy.is_shutdown():
            # start timer in milliseconds
            start_time_stamp = time() * 1000.0

            # Get the image from the camera
            frame = car.cv_image.copy()

            os.system('cls' if os.name=='nt' else 'clear')

            detect.classify_traffic_light(frame)

            detect.classify_sign(frame)

            

            # key = cv.waitKey(1)
            # if key == 27:
            #     car.stop()
            #     cv.destroyAllWindows()
            #     break
            sleep(0.3)
    except rospy.ROSInterruptException:
        pass
      