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
DESIRED_SPEED = 0.2     # [m/s]
OBSTACLE_DISTANCE = 0.20 # [m]

SLOWDOWN_DIST_VALUE = 0.2     # [m]
STOP_DIST_VALUE = 0.6         # [m]

# Pure pursuit controller parameters
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 0.0 #2.0 perpedndicular error gain
k3 = 1.2 #1.5 yaw error gain #inversely proportional
#dt_ahea  = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

if __name__ == '__main__':

    # init the car flow of data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True, trig_enc=True, trig_sonar=True)
    maneuver = Maneuvers()

    # initialize detector
    detect = Detection(trafficlight=True, sign=True, obstacle=False, show_ROI=False)

    # cv.namedWindow('Frame preview', cv.WINDOW_NORMAL)
    # cv.resizeWindow('Frame preview', 320, 240)

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature)

    start_time_stamp = 0.0      # [s]
    # speed_ref = 0.15            # [m/s]
    angle_ref = 0.0             # [deg]

    r = rospy.Rate(10)

    # RESET THE CAR POSITION
    car.stop()
    print("Starting in 1 seconds...")
    sleep(1)
    car.drive_speed(speed=DESIRED_SPEED)

    try:
        last_sign_detected = None

        
        # while not rospy.is_shutdown():
        #     car.drive_angle(15.0)
        #     car.drive_speed(0.2)
        #     car.reset_rel_pose()
        #     while 0 <= car.yawLoc < np.pi:
        #         # print(car.yawLoc)
        #         #print(f'yaw: {car.yaw}')
        #         sleep(0.01)
        # car.stop()
        # exit(0)
        while not rospy.is_shutdown():
            start_time_stamp = time() * 1000.0# [ms] loop start time

            # Get the image from the camera
            frame = car.cv_image.copy()

            # -------------------- OBSTACLE --------------------
            # if last_sign_detected is 'cross_walk':
            #     print('slow down for crosswalk')
            #     car.drive(0.5*DESIRED_SPEED)

            # if car.obstacle_ahead<OBSTACLE_DISTANCE:
            #     car.drive_speed(0.0)# stop
            #     print('found an obstacle')
            #     sleep(1)
            #     print('waiting for the obstacle to go away')
            #     while car.obstacle_ahead_median<OBSTACLE_DISTANCE:
            #         #print(f'this is obstacle ahead: {car.obstacle_ahead_median:.5f} m')
            #         sleep(0.1)
            #     print('wait after the obstacle is gone')
            #     sleep(3)
            #     car.drive_speed(speed=DESIRED_SPEED)
            
            # -------------------- SIGN CLASSIFIER --------------------
            sign, sign_conf = detect.classify_sign(frame, conf_threshold=0.9)
            last_sign_detected = detect.last_sign_detected if detect.last_sign_detected is not 'NO_sign' else last_sign_detected
            
            # -------------------- LANE KEEPING --------------------
            speed_ref, angle_ref, net_out, point_ahead = controller.get_nn_control(frame, DESIRED_SPEED)

            # -------------------- INTERSECTION --------------------
            # dist = net_out[2]# dist returns a function inversely proportional to the distance from an intersection line

            # # Slow down
            # if dist > SLOWDOWN_DIST_VALUE: 
            #     car.drive_speed(speed=0.5*DESIRED_SPEED)
            #     print('slowing down for intersection')

            # # Stop
            # if dist > STOP_DIST_VALUE:# approaching an intersection
            #     if last_sign_detected is 'stop':
            #         car.drive_speed(0.0)
            #         print('STOP INTERSECTION: wait for 3s')
            #         sleep(WAIT_TIME_STOP)
            #         # -------------------- INTERSECTION NAVIGATION --------------------
            #         # maneuver.go_straight(car, ds=0.1)# restart open loop for 5 centimeters
            #         # maneuver.turn_right(car)# restart open loop for 5 centimeters
            #     elif last_sign_detected is 'priority':
            #         print('PRIORITY INTERSECTION: keep going')# keep going
            #         car.drive_speed(speed=DESIRED_SPEED)
            #     else:
            #         print('INTERSECTION but no intersection sign detected')# keep going
            #         car.drive_speed(speed=DESIRED_SPEED)

            # # -------------------- TRAFFIC LIGHT --------------------
            # traffic_light, tl_conf, tl_color = detect.classify_traffic_light(frame,conf_threshold=0.6)
            # if tl_color is 'RED':
            #     car.drive_speed(0.0)
            #     sleep(0.5)
            #     print('waiting for green')
            #     while tl_color is not 'GREEN':
            #         traffic_light, tl_conf, tl_color = detect.classify_traffic_light(car.cv_image)
            #         sleep(0.05)
            #     print('traffic light is green')
            #     car.drive_speed(speed=DESIRED_SPEED)


            ##    maneuver.go_straight(car,ds=0.3)

            # -------------------- ACTUATION --------------------
            #car.drive_speed(speed=speed_ref)
            car.drive_angle(angle=np.rad2deg(angle_ref), direct=False)# steer the car

            # -------------------- PARKING --------------------
            if last_sign_detected is 'park':
                car.drive_speed(0.0)
                car.reset_rel_dist()
                while car.distLoc < 1.4:
                    speed_ref, angle_ref, net_out, point_ahead = controller.get_nn_control(car.cv_image, DESIRED_SPEED)
                    car.drive_angle(angle=np.rad2deg(angle_ref))
                car.drive_speed(0.0)
                sleep(3)
                print('starting parking maneuver')
                maneuver.parallel_parking(car)
            
            # -------------------- LOOP SAMPLING TIME --------------------
            #r.sleep()
            sleep(0.1)

            # -------------------- DEBUG --------------------
            os.system('cls' if os.name=='nt' else 'clear')
            print(f'Net out:\n {net_out}')

            #project point ahead
            print(f'angle_ref: {np.rad2deg(angle_ref)}')
            print(f"point_ahead: {point_ahead}")
            # print(f'Dist: {dist:.3f}')
            # frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
            # if proj is not None:
            #     #convert proj to cv2 point
            #     proj = (int(proj[0]), int(proj[1]))
            #     #draw line from bottmo half to proj
            #     cv.line(frame, (320,479), proj, (50, 50, 250), 2)

            print(f'Loop Time: {time() * 1000.0 - start_time_stamp} ms')
            print(f'FPS: {1.0/(time() * 1000.0 - start_time_stamp)*1000.0}')
            print(f'current speed: {car.speed_meas}')
            print(f'yaw: {car.yaw}')
            print(f'last sign detected: {last_sign_detected}, with confidence {detect.last_sign_conf:.2f}')

            # cv.imshow("Frame preview", frame)
            # key = cv.waitKey(1)
            # if key == 27:
            #     car.stop()
            #     cv.destroyAllWindows()
            #     break
            # sleep(0.2)       

    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass
      