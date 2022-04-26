#!/usr/bin/python3
import os, signal
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
print('Main brain starting...')
# from automobile_data import Automobile_Data
from automobile_data_simulator import AutomobileDataSimulator
from helper_functions import *

from detection import Detection

map = cv.imread('data/2021_VerySmall.png')
class_list = []
with open("data/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

LOOP_DELAY = 0.02
ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08


if __name__ == '__main__':

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame',640,480)
    cv.namedWindow('Map', cv.WINDOW_NORMAL)
    cv.resizeWindow('Map', 600, 600)


    # init the car data
    os.system('rosservice call /gazebo/reset_simulation')
    car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)

    
    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        cv.destroyAllWindows()
        sleep(0.1)
        exit()
    signal.signal(signal.SIGINT, handler)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()

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

            ################################################################
            # TEST DETECTION HERE
            # sign = detect.detect_sign(car.frame, show_ROI=True)

            #test drive distance
            print('driving 0.5')
            car.drive_distance(0.5)
            sleep(15)
            print('stopping')
            car.stop()
            sleep(3)
            print('driving 0.1')
            car.drive_distance(0.9)
            sleep(15)
            car.stop()
            sleep(3)




            ################################################################

            map1 = map.copy()
            draw_car(map1, car.x, car.y, car.yaw, color=(180,0,0))
            color=(255,0,255) if car.trust_gps else (100,0,100)
            draw_car(map1, car.x_true, car.y_true, car.yaw, color=(0,180,0))
            draw_car(map1, car.x_est, car.y_est, car.yaw, color=color)
            cv.imshow('Map', map1)
            cv.waitKey(1)

            ## DEBUG INFO
            print(car)
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
      