#!/usr/bin/python3

import os
import cv2 as cv
from zmq import CURVE
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
print('Main simulator starting...')
from automobile_data import Automobile_Data
from helper_functions import *
from PathPlanning3 import PathPlanning
from controller3 import Controller
from detection import Detection

map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')
class_list = []
with open("models/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

# MAIN CONTROLS
simulator_flag = True # True: run simulator, False: run real car
training = False
generate_path = False if not training else True
# folder = 'training_imgs' 
folder = 'test_imgs'

# os.system('rosservice call /gazebo/reset_simulation')

LOOP_DELAY = 0.0001
ACTUATION_DELAY = 0.15#0.15
VISION_DELAY = 0.03#0.08

# PARAMETERS
sample_time = 0.01 # [s]
desired_speed = 0.5# [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain
k3 = .99 #1.0 yaw error gain .8 with ff 

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain
noise_std = np.deg2rad(0.0) # [rad] noise in the steering angle


if __name__ == '__main__':

    #init windows
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL) if simulator_flag else None

    # cv.namedWindow('Detection', cv.WINDOW_NORMAL)
    
    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if simulator_flag else None
    car = Automobile_Data(simulator=simulator_flag, trig_cam=True, trig_gps=True, trig_bno=True, 
                            trig_enc=False, trig_control=True, trig_estimation=False)
    car.stop()

    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature, folder=folder, 
                                    training=training, noise_std=noise_std)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()


    try:
        #generate path
        if generate_path:
            path_nodes = [86,436,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
                            97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355]
            path_nodes = [86,110,436,467] #,273,136,321,262]
            path.generate_path_passing_through(path_nodes, path_step_length) #[86,110,310,254,463] ,[86,436, 273,136,321,262,105,350,451,265]
            # [86,436, 273,136,321,262,105,350,373,451,265,145,160,353]
            path.draw_path()
            path.print_path_info()

        while not rospy.is_shutdown():
            loop_start_time = time()

            tmp = np.copy(map)
            # Get the image from the camera
            frame = car.cv_image.copy()
            sleep(VISION_DELAY)

            #FOLLOW predefined trajectory
            if generate_path:
                reference = path.get_reference(car, desired_speed, frame=frame, training=training)
                xd, yd, yawd, curv, finished, path_ahead, info = reference
                #controller training data generation        
                controller.curr_data = [xd,yd,yawd,curv,path_ahead,info]
                if finished:
                    print("Reached end of trajectory")
                    car.stop()
                    sleep(.8)
                    break
                #draw refrence car
                draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))


            ## CONTROL
            if training:
                #car control, unrealistic: uses the true position
                #training
                speed_ref, angle_ref, point_ahead = controller.get_training_control(car, path_ahead, desired_speed, curv)
                controller.save_data(frame, folder)
                dist = info[3]
                if dist is not None and 0.0 < dist < 0.5:
                    mod_dist = ((1.0-dist)*10.0)**3
                    speed_ref = 0.8 * speed_ref
            else:
                #Neural network control
                lane_info = detect.detect_lane(frame)
                e2, e3, dist, curv, point_ahead = lane_info

                speed_ref, angle_ref = controller.get_control(e2, e3, curv, desired_speed)

                # # #stopping logic
                # if dist > 1.0:
                #     print('Slowing down')
                #     speed_ref = desired_speed * 0.8
                #     if dist > 3.0:
                #         print('Stopping')
                #         car.stop()
                #         sleep(1)
                #         speed_ref = desired_speed
                #         car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
                #         sleep(0.3)

            ## ACTUATION
            sleep(ACTUATION_DELAY)
            car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))

            ## VISUALIZATION
            #project path ahead
            if generate_path:
                frame, proj = project_onto_frame(frame, car, path_ahead, color=(0,0,100))
            #project point ahead
            frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
            if proj is not None:
                #convert proj to cv2 point
                proj = (int(proj[0]), int(proj[1]))
                #draw line from bottmo half to proj
                cv.line(frame, (320,479), proj, (200, 200, 100), 2)
            
            #draw true car position
            draw_car(tmp, car.x_true, car.y_true, car.yaw, color=(0, 255, 0))

            #draw lateral error
            frame = cv.line(frame, (320,479), (320-int(np.clip(controller.e2*1000, -319, 319)),479), (0,200,0), 3)  
            
            #draw curvature
            radius = project_curvature(frame, car, curv)

            ## DEBUG INFO
            os.system('cls' if os.name=='nt' else 'clear')
            print(f"x : {car.x_true:.3f}, y : {car.y_true:.3f}, yaw : {np.rad2deg(car.yaw):.3f}") 
            print(f"xd: {xd:.3f}, yd: {yd:.3f}, yawd: {np.rad2deg(yawd):.3f}, curv: {curv:.3f}") if generate_path else None
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {np.rad2deg(controller.e3):.3f}")
            print(f'desired_speed = {desired_speed:.3f}, speed_ref = {speed_ref:.3f}, angle_ref = {np.rad2deg(angle_ref):.3f}')
            print(f"INFO:\nState: {info[0]}\nNext: {info[1]}\nAction: {info[2]}\nDistance: {info[3]}") if generate_path else None
            print(f'MOD_DIST: {mod_dist:.3f}') if dist is not None and 0.0 < dist < 0.5 and training else None
            # print(f"Coeffs:\n{coeffs}")
            print(f'Net out:\n {lane_info}') if not training else None
            print(f'e_yaw: {e3}\ndist: {dist}\ncurv: {100*curv}') if not training else None
            print(f'Curvature radius = {radius:.2f}')
            print(f'FPS = {1/(time()-loop_start_time):.3f}')

            cv.imshow("Frame preview", frame)
            # cv.imshow('SIGNS ROI', signs_roi)
            # cv.imshow('FRONT ROI', front_obstacle_roi)
            cv.imshow("2D-MAP", tmp) if simulator_flag else None
            key = cv.waitKey(1)
            if key == 27:
                car.stop()
                sleep(.5)
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
      