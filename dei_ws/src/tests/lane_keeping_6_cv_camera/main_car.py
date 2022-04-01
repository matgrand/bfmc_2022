#!/usr/bin/python3
from this import d
import cv2 as cv
import rospy
import numpy as np
from time import sleep
from time import time
import os

from automobile_data import Automobile_Data     # I/O car manager
from controller3 import Controller              # Lane Keeping
from maneuvers import Maneuvers                 # Maneuvers
from helper_functions import *                  # helper functions
from detection import Detection                 # detection

# PARAMETERS
SAMPLE_TIME = 0.01      # [s]
WAIT_TIME_STOP = 3.0    # [s]
DESIRED_SPEED = 0.2    # [m/s]
OBSTACLE_DISTANCE = 0.20 # [m]

SLOWDOWN_DIST_VALUE = 0.2     # [m]
STOP_DIST_VALUE = 0.6         # [m]

SHOW_IMGS = True

# Pure pursuit controller parameters
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 0.0 #2.0 perpedndicular error gain
k3 = 0.99 #1.5 yaw error gain 
#dt_ahea  = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

if __name__ == '__main__':

    #load camera with opencv
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)

    # init the car flow of data
    car = Automobile_Data(trig_control=True, trig_cam=False, trig_gps=False, trig_bno=True, trig_enc=True, trig_sonar=True)
    maneuver = Maneuvers()

    # initialize detector
    detect = Detection()

    if SHOW_IMGS:
        cv.namedWindow('Frame preview', cv.WINDOW_NORMAL)
        cv.resizeWindow('Frame preview', 320, 240)
        cv.namedWindow('lane_detection', cv.WINDOW_NORMAL)
        cv.resizeWindow('lane_detection', 200, 200)

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature, training=False)

    start_time_stamp = 0.0      # [s]
    angle_ref = 0.0             # [deg]

    # RESET THE CAR POSITION
    car.stop()
    print("Starting in 3 seconds...")
    sleep(3)
    car.drive_speed(speed=DESIRED_SPEED)

    try:

        while not rospy.is_shutdown():
            start_time_stamp = time() * 1000.0# [ms] loop start time

            # Get the image from the camera
            # frame = car.cv_image.copy()
            ret, frame = cap.read()
            if not ret:
                print("No image from camera")
                frame = np.zeros((480, 640, 3), np.uint8)
                continue

            # -------------------- LANE KEEPING --------------------
            #Neural network control
            lane_info = detect.detect_lane(frame, show_ROI=SHOW_IMGS)
            e2, e3, point_ahead = lane_info
            curv = 0.0001
            speed_ref, angle_ref = controller.get_control(e2, e3, curv, DESIRED_SPEED)

            dist = detect.detect_stop_line(frame, show_ROI=SHOW_IMGS)

            # #stopping logic
            if 0.0 < dist < 0.35:
                print('Slowing down')
                speed_ref = DESIRED_SPEED * 0.5
                if 0.0 < dist < 0.15:
                    print('Stopping')
                    speed_ref = 0.0


            # -------------------- SIGNS ---------------------------
            sign = detect.detect_sign(frame, show_ROI=SHOW_IMGS)

            # -------------------- INTERSECTION --------------------

            # -------------------- ACTUATION --------------------
            car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))

            # -------------------- LOOP SAMPLING TIME --------------------
            #r.sleep()
            # sleep(0.1)

        

            # -------------------- DEBUG --------------------
            os.system('cls' if os.name=='nt' else 'clear')
            print(f'Net out:\n {lane_info}')

            #project point ahead
            print(f'angle_ref: {np.rad2deg(angle_ref)}')
            print(f"point_ahead: {point_ahead}")

            print(f'Loop Time: {time() * 1000.0 - start_time_stamp} ms')
            print(f'FPS: {1.0/(time() * 1000.0 - start_time_stamp)*1000.0}')
            print(f'current speed: {car.speed_meas}')
            print(f'yaw: {car.yaw}')
            print(f'Lane detection time = {detect.avg_lane_detection_time:.2f} ms')
            print(f'Sign detection time = {detect.avg_sign_detection_time:.2f} ms')

            if SHOW_IMGS:
                #project point ahead
                frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
                if proj is not None:
                    #convert proj to cv2 point
                    proj = (int(proj[0]), int(proj[1]))
                    #draw line from bottmo half to proj
                    cv.line(frame, (320,479), proj, (200, 200, 100), 2)

                cv.imshow('Frame preview', frame)
                key = cv.waitKey(1)
                if key == 27:
                    car.stop()
                    sleep(1.5)
                    cv.destroyAllWindows()
                    print("Shutting down...")
                    exit(0)

    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass
      