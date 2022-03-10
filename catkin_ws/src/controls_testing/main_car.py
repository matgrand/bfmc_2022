#!/usr/bin/python3
from mimetypes import init
import cv2 as cv
import rospy
from time import sleep
import os

from  automobile_data import Automobile_Data
from helper_functions import *



# PARAMETERS
sample_time = 0.01 # [s]
max_angle = 30.0    # [deg]
max_speed = 0.5  # [m/s]
desired_speed = 0.2 # [m/s]
# CONTROLLER
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 1.0 #1.5 perpedndicular error gain
k3 = 1.5 #1.5 yaw error gain
#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.2 # feedforward gain

if __name__ == '__main__':

    # init the car data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True, trig_stateEst=False)

    start_time_stamp = 0.0

    pause1 = 5 #.02
    pause2 = 3 #.05

    car.stop()

    print('Stopped')

    #tests
    car.activate_encoder()
    # output = " "
    # ser = serial.Serial('/dev/ttyACM0', 256000, 8, 'N', 1, timeout=1)
    # while True:
    #     print("----")
    #     while output != "":
    #         output = ser.readline()
    #         print(output)
    #     output = " "
    
    sleep(1)

    flag = True

    try:
        while not rospy.is_shutdown():
            os.system('cls' if os.name=='nt' else 'clear')
            # # Get the image from the camera
            # frame = car.cv_image.copy()

            speed = desired_speed if flag else -desired_speed
            car.drive_speed(speed=speed)
            print('Driving')

            sleep(pause1)
            print(f'encoder position is {car.enc_pos} m')

            car.stop()
            print('Stopped')

            sleep(pause1)
            # car.activate_encoder()

            #angle = 3 if flag else -3
            flag = not flag
            #car.drive_angle(angle=angle)
            #print(f'Turning with angle {angle}')

            #sleep(pause2)

            # sleep(8)

            # cv.imshow("Frame preview", frame)
            # cv.waitKey(1)
            # sleep(0.01)

    except rospy.ROSInterruptException:
        pass
      
