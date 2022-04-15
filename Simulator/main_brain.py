#!/usr/bin/python3
SIMULATOR = True # True: run simulator, False: run real car

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

LOOP_DELAY = 0.05
ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08

# PARAMETERS
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.6# [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain   #pure paralllel k2 = 10 is very good at remaining in the center of the lane
k3 = 0.6 #1.0 yaw error gain .8 with ff 
k3D = 0.08 #0.08 derivative gain of yaw error

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain
noise_std = np.deg2rad(25) # [rad] noise in the steering angle

if __name__ == '__main__':

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame',640,480)
    # show windows
    cv.namedWindow('Path', cv.WINDOW_NORMAL)
    cv.resizeWindow('Path', 600, 600)
    cv.namedWindow('Map', cv.WINDOW_NORMAL)
    cv.resizeWindow('Map', 600, 600)


    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR else None
    if SIMULATOR:
        car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    else:
        car = AutomobileDataPi(trig_cam=True, trig_gps=False, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    
    
    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        cv.destroyAllWindows()
        exit(1)
    signal.signal(signal.SIGINT, handler)
    
    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, k3D=k3D, ff=ff_curvature)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()

    #initiliaze the brain
    brain = Brain(car=car, controller=controller, detection=detect, path_planner=path, desired_speed=DESIRED_SPEED)


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

            map1 = map.copy()
            draw_car(map1, car.x, car.y, car.yaw, color=(255,0,0))
            draw_car(map1, car.x_est, car.y_est, car.yaw, color=(255,0,255))
            draw_car(map1, car.x_true, car.y_true, car.yaw)
            cv.imshow('Map', map1)
            cv.waitKey(1)


            # RUN BRAIN
            brain.run()

            ## DEBUG INFO
            print(car)
            # print(f"x : {car.x_true:.3f} [m], y : {car.y_true:.3f} [m], yaw : {np.rad2deg(car.yaw):.3f} [deg]") 
            # print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {np.rad2deg(controller.e3):.3f}")
            # print(f'Current velocity: {car.filtered_encoder_velocity:.3f} [m/s]')
            # print(f'total distance travelled: {car.encoder_distance:.2f} [m]')
            # print()
            # print(f'Local: dist: {car.dist_loc:.2f} [m], x: {car.x_loc:.2f} [m], y: {car.y_loc:.2f} [m], yaw: {np.rad2deg(car.yaw_loc):.2f} [deg]')
            # print()
            # print(f'Front sonar distance:     {car.filtered_sonar_distance:.2f} [m]')
            print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')
            print(f'Sign detection time = {detect.avg_sign_detection_time:.1f} [ms]')
            print(f'FPS = {fps_avg:.1f},  loop_cnt = {fps_cnt}')

            cv.imshow('frame', car.frame)
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
      