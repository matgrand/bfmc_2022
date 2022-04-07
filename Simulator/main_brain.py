#!/usr/bin/python3

import os
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
print('Main brain starting...')
from automobile_data import Automobile_Data
from helper_functions import *
from PathPlanning4 import PathPlanning
from controller3 import Controller
from detection import Detection
from brain import Brain

map = cv.imread('data/2021_VerySmall.png')
class_list = []
with open("data/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

# MAIN CONTROLS
SIMULATOR = True # True: run simulator, False: run real car
training = False
generate_path = False if not training else True
# folder = 'training_imgs' 
folder = 'test_imgs'

# os.system('rosservice call /gazebo/reset_simulation')

LOOP_DELAY = 0.01
ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08

# PARAMETERS
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.7# [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain   #pure paralllel k2 = 10 is very good at remaining in the center of the lane
k3 = 0.6 #1.0 yaw error gain .8 with ff 

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain
noise_std = np.deg2rad(25) # [rad] noise in the steering angle


if __name__ == '__main__':

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame',640,480)

    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR else None
    car = Automobile_Data(simulator=SIMULATOR, trig_cam=True, trig_gps=True, trig_bno=True, 
                            trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)

    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature, folder=folder, 
                                    training=training, noise_std=noise_std)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()

    #initiliaze the brain
    brain = Brain(car=car, controller=controller, detection=detect, path_planner=path, desired_speed=DESIRED_SPEED)

    try:
        car.stop()
        fps_avg = 0.0
        fps_cnt = 0
        while not rospy.is_shutdown():
            os.system('cls' if os.name=='nt' else 'clear')
            loop_start_time = time()

            # RUN BRAIN
            brain.run()

            ## DEBUG INFO
            print(f"x : {car.x_true:.3f} [m], y : {car.y_true:.3f} [m], yaw : {np.rad2deg(car.yaw):.3f} [deg]") 
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {np.rad2deg(controller.e3):.3f}")
            print(f'total distance travelled: {car.tot_dist:.2f} [m]')
            print(f'Front sonar distance:     {car.obstacle_ahead_median:.2f} [m]')
            loop_time = time() - loop_start_time
            fps_avg = (fps_avg * fps_cnt + 1.0 / loop_time) / (fps_cnt + 1)
            fps_cnt += 1
            print(f'FPS = {fps_avg:.1f},  loop_cnt = {fps_cnt}')
            print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')
            print(f'Sign detection time = {detect.avg_sign_detection_time:.1f} [ms]')

            cv.imshow('frame', car.cv_image)
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break

            sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("Shutting down")
        car.stop()
        sleep(.5)
        cv.destroyAllWindows()
        exit(0)
    except rospy.ROSInterruptException:
        pass
      