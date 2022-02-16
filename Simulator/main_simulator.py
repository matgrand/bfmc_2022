#!/usr/bin/python3
from mimetypes import init
import string
from tkinter import Frame
import cv2 as cv
from zmq import CURVE_SERVERKEY
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
import matplotlib.pyplot as plt

from LaneKeepingReloaded import LaneKeepingReloaded
from  automobile_data import Automobile_Data
from PathPlanning_advanced import *
from PathPlanning_advanced import PathPlanning

from simple_controller import SimpleController


# map = cv.imread('src/models_pkg/track/materials/textures/2021_Medium.png')
map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')

if __name__ == '__main__':

    #init windows
    #cv.namedWindow("Poly") 
    #cv.namedWindow("Frame preview") 
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL)
    
    #lane keeping
    #lane_keeping = LaneKeepingReloaded(640, 480)

    #initilize the car data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True)

    # PARAMETERS
    sample_time = 0.01 # [s]
    max_angle = 30.0    # [deg]
    max_speed = 0.5  # [m/s]
    desired_speed = 0.30 # [m/s]
    nodes_ahead = 400 # [nodes] how long the trajectory will be
    samples_per_edge = 100# [steps] how many steps per edge in graph
    # init trajectory
    path = PathPlanning(map, source=86, target=235)

    #init controller
    k1 = 0.0 #4.0 gain error parallel to direction (speed)
    k2 = 2.0 #2.0 perpedndicular error gain
    k3 = 1.5 #1.5 yaw error gain
    #dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
    ff_curvature = 1.0 # feedforward gain
    controller = SimpleController(k1=k1, k2=k2, k3=k3, ff=ff_curvature)

    start_time_stamp = 0.0

    car.stop()
    os.system('rosservice call /gazebo/reset_simulation')

    try:

        while car.time_stamp < 1:
            sleep(0.1)

        start_time_stamp = car.time_stamp

        path.compute_shortest_path(step_length=0.01)
        path.draw_path()

        while not rospy.is_shutdown():
            # Get the image from the camera
            frame = car.cv_image

            #draw car on map
            tmp = np.copy(map)
            #draw true car position
            draw_car(tmp, car.x_true, car.y_true, car.yaw, color=(0, 255, 0))

            #lane keeping
            #angle_ref, both_lanes, poly = lane_keeping.lane_keeping_pipeline(frame)

            #FOLLOW predefined trajectory
            xd, yd, yawd, curv, finished = path.get_reference(car)
            if finished:
                print("Reached end of trajectory")
                break
            #draw refrence car
            draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))

            #car control, unrealistic: uses the true position
            speed_ref, angle_ref = controller.get_control(car.x_true, car.y_true, car.yaw, xd, yd, yawd, desired_speed, curv)
            car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))

        
            #prints
            # clear stdout for a smoother display
            os.system('cls' if os.name=='nt' else 'clear')
            print(f"x : {car.x_true:.3f}, y : {car.y_true:.3f}, yaw : {np.rad2deg(car.yaw):.3f}")
            print(f"xd: {xd:.3f}, yd: {yd:.3f}, yawd: {np.rad2deg(yawd):.3f}, curv: {np.rad2deg(curv):.3f}")
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {controller.e3:.3f}")
            print(f"speed_ref: {speed_ref:.3f},    angle_ref: {angle_ref:.3f}")
            #print(f"time remaining: {trajectory.total_time-(car.time_stamp - start_time_stamp):.3f} seconds")
    
            #show poly_image
            # cv.imshow("Poly", poly)
            cv.imshow("Frame preview", frame)
            cv.imshow("2D-MAP", tmp)
            cv.waitKey(1)
            rospy.sleep(0.01)

    except rospy.ROSInterruptException:
        pass
      