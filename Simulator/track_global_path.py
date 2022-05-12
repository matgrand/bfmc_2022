#!/usr/bin/python3
SIMULATOR = False # True: run simulator, False: run real car
SHOW_IMGS = True

import os, signal
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
print('Track global tracking starting...')
# from automobile_data import Automobile_Data
if SIMULATOR:
    from automobile_data_simulator import AutomobileDataSimulator
    from helper_functions import *
else: #PI
    from control.automobile_data_pi import AutomobileDataPi
    from control.helper_functions import *

from PathPlanning4 import PathPlanning
from controller3 import Controller
from controllerSP import ControllerSpeed
from detection import Detection
from brain import Brain
from environmental_data_simulator import EnvironmentalData

map = cv.imread('data/2021_VerySmall.png')

# PARAMETERS
TARGET_FPS = 30.0
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.3# [m/s]
CURVE_SPEED = 0.8# [m/s]
path_step_length = 0.01 # [m]

LOOP_DELAY = 0.01
CHECKPOINTS = [86,99,112,337,463,240]

K1 = 0.0
K2 = 0.0
K3 = .9
K3D = 0.0
FF = 0.0
DIST_POINT_AHEAD = 0.5

#load camera with opencv
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FPS, 30)


if __name__ == '__main__':

    if SHOW_IMGS:
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame',640,480)
        cv.namedWindow('Map', cv.WINDOW_NORMAL)
        cv.resizeWindow('Map', 600, 600)


    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR else None
    if SIMULATOR:
        car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    else:
        car = AutomobileDataPi(trig_cam=False, trig_gps=False, trig_bno=True, 
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

    controller = Controller(k1=K1,k2=K2,k3=K3, k3D=K3D, ff=FF, cm_ahead=int(100*DIST_POINT_AHEAD))

    path_planner = PathPlanning(map)

    path_planner.generate_path_passing_through(CHECKPOINTS)
    path_planner.draw_path()
    path = path_planner.path

    if SHOW_IMGS:
        map1 = map
        draw_car(map1, car.x_true, car.y_true, car.yaw)
        cv.imshow('Map', map1)
        cv.waitKey(1)

    try:
        car.drive_speed(0.0)


        fps_avg = 0.0
        fps_cnt = 0
        while True:
        #TODO add the stopline waiting
            if car.trust_gps or True:
                break
            sleep(0.1)
            print('Waiting for GPS...')

        while not rospy.is_shutdown():
            # os.system('cls' if os.name=='nt' else 'clear')

            loop_start_time = time()
            # frame = car.frame.copy()


            # stopline_x, stopline_y, stopline_angle = detect.detect_stop_line2(car.frame, show_ROI=True)

            # frame, _ = project_stopline(frame, car, stopline_x, stopline_y, stopline_angle, color=(0,200,0))
            if not SIMULATOR:
                ret, frame = cap.read()
                if not ret:
                    print("No image from camera")
                    frame = np.zeros((240, 320, 3), np.uint8)
                    continue

            #control
            car_est_pos = np.array([car.x_est, car.y_est])
            idx_car_on_path = np.argmin(np.linalg.norm(path-car_est_pos, axis=1))
            p_ahead = path[min(len(path)-1, idx_car_on_path+int(100*DIST_POINT_AHEAD))]
            e3 = diff_angle(np.arctan2(p_ahead[1]-car_est_pos[1], p_ahead[0]-car_est_pos[0]),car.yaw)
            
            output_speed, output_angle = controller.get_control(e2=0, e3=e3, curv=0, desired_speed=DESIRED_SPEED)
            print(f'Speed: {output_speed:.2f}, Angle: {output_angle:.1f}')
            car.drive(output_speed, np.rad2deg(output_angle))

            ################################################################
            if SHOW_IMGS:
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
            print(f'FPS = {fps_avg:.1f},  loop_cnt = {fps_cnt}')

            if SHOW_IMGS:        
                cv.imshow('frame', frame)
                if cv.waitKey(1) == 27:
                    cv.destroyAllWindows()
                    break
            
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
      