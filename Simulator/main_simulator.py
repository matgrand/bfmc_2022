#!/usr/bin/python3
from mimetypes import init
import string
from tkinter import Frame
import cv2 as cv
from cv2 import fastNlMeansDenoising
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

from  automobile_data import Automobile_Data
from helper_functions import *
from PathPlanning_advanced import PathPlanning
from simple_controller2 import SimpleController


# map = cv.imread('src/models_pkg/track/materials/textures/2021_Medium.png')
map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')

training = False
# folder = 'training_imgs' 
folder = 'test_imgs'

# PARAMETERS
sample_time = 0.01 # [s]
max_angle = 30.0    # [deg]
max_speed = 0.5  # [m/s]
desired_speed = 0.3 # [m/s]
nodes_ahead = 400 # [nodes] how long the trajectory will be
samples_per_edge = 100# [steps] how many steps per edge in graph
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 1.5 #2.0 perpedndicular error gain
k3 = 1.5 #1.5 yaw error gain
#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.2 # feedforward gain
noise_std = np.deg2rad(25) # [rad] noise in the steering angle


if __name__ == '__main__':

    #init windows
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL)
    
    # init the car data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True)

    # init trajectory
    path = PathPlanning(map) #254 463

    # init controller
    controller = SimpleController(k1=k1, k2=k2, k3=k3, ff=ff_curvature, folder=folder, 
                                    training=training, noise_std=noise_std)

    start_time_stamp = 0.0

    car.stop()
    os.system('rosservice call /gazebo/reset_simulation')

    try:

        while car.time_stamp < 1:
            sleep(0.1)

        start_time_stamp = car.time_stamp

        #generate path
        path_nodes = [86,436,273,136,321,262,105,350,373,451,265,145,160,353,94,127,91,99,
                        97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355]
        path_nodes = [86,436,273,136,321,262]
        path.generate_path_passing_through(path_nodes, path_step_length) #[86,110,310,254,463] ,[86,436, 273,136,321,262,105,350,451,265]
        # [86,436, 273,136,321,262,105,350,373,451,265,145,160,353]
        path.draw_path()
        path.print_path_info()

        cv.waitKey(0)

        while not rospy.is_shutdown():
            tmp = np.copy(map)
            # Get the image from the camera
            frame = car.cv_image.copy()

            #draw true car position
            draw_car(tmp, car.x_true, car.y_true, car.yaw, color=(0, 255, 0))

            bounding_box = (0,0,0,0)
            sign = 'no_sign'
            # if training :
            #     bounding_box, frame, sign = add_sign(frame)

            #FOLLOW predefined trajectory
            xd, yd, yawd, curv, finished, path_ahead, info, coeffs = path.get_reference(car, desired_speed, 
                                                                                        frame=frame, training=training)
            controller.curr_data = [xd,yd,yawd,curv,path_ahead,info,coeffs,bounding_box,sign]
            if finished:
                print("Reached end of trajectory")
                car.stop()
                break
            #draw refrence car
            draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))

            #car control, unrealistic: uses the true position
            if training:
                #training
                speed_ref, angle_ref = controller.get_control(car.x_true, car.y_true, car.yaw, xd, yd, yawd, desired_speed, curv)
                controller.save_data(frame, folder)
            else:
                #Neural network control
                action = info[2]
                speed_ref, angle_ref, net_out = controller.get_nn_control(frame, desired_speed, action)

            
            car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))

        
            #prints
            # clear stdout for a smoother display
            os.system('cls' if os.name=='nt' else 'clear')
            print(f"x : {car.x_true:.3f}, y : {car.y_true:.3f}, yaw : {np.rad2deg(car.yaw):.3f}")
            print(f"xd: {xd:.3f}, yd: {yd:.3f}, yawd: {np.rad2deg(yawd):.3f}, curv: {np.rad2deg(curv):.3f}")
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {controller.e3:.3f}")
            print(f"speed_ref: {speed_ref:.3f},    angle_ref: {angle_ref:.3f}")
            print(f"INFO:\nState: {info[0]}\nNext: {info[1]}\nAction: {info[2]}\nDistance: {info[3]}")
            # print(f"Coeffs:\n{coeffs}")
            print(f'Net out:\n {net_out}') if not training else None
     
            # if not training:
            #     bb = net_out[5]
            #     sign = net_out[2]
            #     # if sign != 'no_sign':
            #     draw_bounding_box(frame, bb)

            #project path ahead
            frame, proj = project_onto_frame(frame, car, path_ahead) 

            cv.imshow("Frame preview", frame)
            cv.imshow("2D-MAP", tmp)
            cv.waitKey(1)
            rospy.sleep(0.01)

    except rospy.ROSInterruptException:
        pass
      