#!/usr/bin/python3
SIMULATOR = True # True: run simulator, False: run real car
SHOW_IMGS = True

import os, signal
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
print('Main brain starting...')
# from automobile_data import Automobile_Data
if SIMULATOR:
    from automobile_data_simulator import AutomobileDataSimulator
    from helper_functions import *
else: #PI
    from control.automobile_data_pi import AutomobileDataPi
    from control.helper_functions import *

from PathPlanning4 import PathPlanning
from controller3 import Controller
from detection import Detection
from brain import Brain

map = cv.imread('data/2021_VerySmall.png')
class_list = []
with open("data/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

LOOP_DELAY = 0.01
ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08

# PARAMETERS
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.3# [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain   #pure paralllel k2 = 10 is very good at remaining in the center of the lane
k3 = 0.6 #1.0 yaw error gain .8 with ff 
k3D = 0.08 #0.08 derivative gain of yaw error

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

#load camera with opencv
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FPS, 30)



if __name__ == '__main__':

    if SHOW_IMGS:
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame',640,480)
        # show windows
        cv.namedWindow('Path', cv.WINDOW_NORMAL)
        cv.resizeWindow('Path', 600, 600)
        cv.namedWindow('Map', cv.WINDOW_NORMAL)
        cv.resizeWindow('Map', 600, 600)


    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR else None
    os.system('rosservice call gazebo/unpause_physics') if SIMULATOR else None
    # sleep(1.5)
    if SIMULATOR:
        car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    else:
        car = AutomobileDataPi(trig_cam=False, trig_gps=False, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    sleep(1.5)
    car.encoder_distance = 0.0
    
    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        os.system('rosservice call gazebo/pause_physics') if SIMULATOR else None 
        cv.destroyAllWindows()
        sleep(.99)
        exit()
    signal.signal(signal.SIGINT, handler)
    
    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, k3D=k3D, ff=ff_curvature)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()

    #initiliaze the brain
    brain = Brain(car=car, controller=controller, detection=detect, path_planner=path, desired_speed=DESIRED_SPEED)

    if SHOW_IMGS:
        map1 = map.copy()
        draw_car(map1, car.x_true, car.y_true, car.yaw)
        cv.imshow('Map', map1)
        cv.waitKey(1)


    try:
        car.stop()
        fps_avg = 0.0
        fps_cnt = 0
        while not rospy.is_shutdown():
            os.system('cls' if os.name=='nt' else 'clear')

            loop_start_time = time()

            if SHOW_IMGS:
                map1 = map.copy()
                draw_car(map1, car.x, car.y, car.yaw, color=(180,0,0))
                color=(255,0,255) if car.trust_gps else (100,0,100)
                draw_car(map1, car.x_true, car.y_true, car.yaw, color=(0,180,0))
                draw_car(map1, car.x_est, car.y_est, car.yaw, color=color)
                cv.imshow('Map', map1)
                cv.waitKey(1)

            if not SIMULATOR:
                ret, frame = cap.read()
                brain.car.frame = frame
                if not ret:
                    print("No image from camera")
                    frame = np.zeros((240, 320, 3), np.uint8)
                    continue

            # RUN BRAIN
            brain.run()

            ## DEBUG INFO
            print(car)
            print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')
            print(f'Sign detection time = {detect.avg_sign_detection_time:.1f} [ms]')
            print(f'FPS = {fps_avg:.1f},  loop_cnt = {fps_cnt}')

            if SHOW_IMGS:
                frame = car.frame.copy()
                if brain.stop_line_distance_median is not None:
                    dist = brain.stop_line_distance_median - car.encoder_distance + 0.1
                    angle_to_stopline = diff_angle(car.yaw, get_yaw_closest_axis(car.yaw))
                    frame, _ = project_stopline(frame, car, dist, angle_to_stopline, color=(0,200,0))
                cv.imshow('frame', frame)
                if cv.waitKey(1) == 27:
                    cv.destroyAllWindows()
                    break
            
            sleep(LOOP_DELAY)
            loop_time = time() - loop_start_time
            fps_avg = (fps_avg * fps_cnt + 1.0 / loop_time) / (fps_cnt + 1)
            fps_cnt += 1

    except KeyboardInterrupt:
        print("Shutting down")
        car.stop()
        sleep(.5)
        cv.destroyAllWindows()
        exit(0)
    except rospy.ROSInterruptException:
        pass
      