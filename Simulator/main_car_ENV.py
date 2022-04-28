#!/usr/bin/python3
from this import d
import cv2 as cv
import rospy
import numpy as np
from time import sleep
from time import time
import os
import signal


from automobile_data_simulator import AutomobileDataSimulator     # I/O car manager
from environmental_data_simulator import EnvironmentalData
from helper_functions import *                  # helper functions

# PARAMETERS
DESIRED_SPEED = 0.1    # [m/s]

# Pure pursuit controller parameters
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 0.0 #2.0 perpedndicular error gain
k3 = 0.7 #1.5 yaw error gain 
#dt_ahea  = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

# init the car flow of data
car = AutomobileDataSimulator(trig_control=True, trig_cam=True, trig_gps=True, trig_bno=True, trig_enc=True, trig_sonar=True)
env = EnvironmentalData(trig_semaphore=True, trig_v2v=True, trig_v2x=True)

rate = rospy.Rate(100)

speed_ref, angle_ref = 0.0, 0.0    

# Handler for exiting the program with CTRL+C
def handler(signum, frame):
    print("Exiting ...")
    car.stop()
    cv.destroyAllWindows()
    exit(1)
signal.signal(signal.SIGINT, handler)


if __name__ == '__main__':

    # RESET THE CAR POSITION
    car.stop()
    print("Starting in 1 seconds...")
    sleep(1)
    car.drive_speed(speed=DESIRED_SPEED)
    env.publish_obstacle('TRAFFICLIGHT',car.x, car.y)

    try:
        while not rospy.is_shutdown():
            os.system('cls' if os.name=='nt' else 'clear')
            print(env)
            sleep(0.1)

    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass
      