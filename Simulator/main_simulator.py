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
import os

from LaneKeepingReloaded import LaneKeepingReloaded
from  car_data import Automobile_Data

from generate_standard_route import SimpleController, TrainingTrajectory


# map = cv.imread('src/models_pkg/track/materials/textures/2021_Medium.png')
map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')

if __name__ == '__main__':

    #init windows
    #cv.namedWindow("Poly") 
    #cv.namedWindow("Frame preview") 
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL)
    
    #lane keeping
    #lane_keeping = LaneKeepingReloaded(640, 480)

    # PARAMETERS
    publish_time = 0.01 # [s]
    max_angle = 30.0    # [deg]
    max_speed = 0.5  # [m/s]
    desired_speed = 0.25 # [m/s]
    nodes_ahead = 200 # [nodes] how long the trajectory will be
    # init trajectory
    trajectory = TrainingTrajectory(nodes_ahead=nodes_ahead, speed=desired_speed)

    #init controller
    k1 = 4.0 #4.0 gain error parallel to direction (speed)
    k2 = 2.0 #2.0 perpedndicular error gain
    k3 = 1.5 #1.5 yaw error gain
    dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
    ff_curvature = 1.0 # feedforward gain
    controller = SimpleController(k1=k1, k2=k2, k3=k3, ff=ff_curvature)

    start_time_stamp = 0.0

    os.system('rosservice call /gazebo/reset_simulation')

    try:
        # init camera publishe
        car = Automobile_Data()

        while car.time_stamp < 1:
            sleep(0.1)

        start_time_stamp = car.time_stamp
            
        while not rospy.is_shutdown():
            # Get the image from the camera
            frame = car.cv_image

            #draw car on map
            tmp = np.copy(map)
            draw_car(tmp, car.x, car.y, car.yaw, color=(0, 255, 0))

            #lane keeping
            #angle_ref, both_lanes, poly = lane_keeping.lane_keeping_pipeline(frame)

            #FOLLOW predefined trajectory
            xd, yd, yawd, curv, finished = trajectory.get_reference(car.time_stamp - start_time_stamp, dt_ahead)
            if finished:
                print("Reached end of trajectory")
                break
            draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))

            #car control
            speed_ref, angle_ref = controller.get_control(car.x, car.y, car.yaw, xd, yd, yawd, desired_speed, curv)
            #convert angle to degrees
            angle_ref = angle_ref*180/np.pi
            #clip inputs
            speed_ref = np.min([speed_ref, max_speed])
            speed_ref = np.max([speed_ref, 0.0])
            angle_ref = np.min([angle_ref, max_angle])
            angle_ref = np.max([angle_ref, -max_angle])

            car._send_reference(speed_ref,angle_ref)
            rospy.sleep(publish_time)
            #print("Published reference: speed = %.2f, angle = %.2f" % (speed_ref, angle_ref))

            #prints
            # clear stdout for a smoother display
            os.system('cls' if os.name=='nt' else 'clear')
            print(f"x : {car.x:.3f}, y : {car.y:.3f}, yaw : {car.yaw*180/np.pi:.3f}")
            print(f"xd: {xd:.3f}, yd: {yd:.3f}, yawd: {yawd*180/np.pi:.3f}, curv: {curv*180/np.pi:.3f}")
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {controller.e3:.3f}")
            print(f"speed_ref: {speed_ref:.3f},    angle_ref: {angle_ref:.3f}")
            print(f"time remaining: {trajectory.total_time-(car.time_stamp - start_time_stamp):.3f} seconds")
    
            #show poly_image
            # cv.imshow("Poly", poly)
            cv.imshow("Frame preview", frame)
            cv.imshow("2D-MAP", tmp)
            cv.waitKey(1)

    except rospy.ROSInterruptException:
        pass
      