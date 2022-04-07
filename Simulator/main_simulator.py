#!/usr/bin/python3

import os
import cv2 as cv
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
SIMULATOR = True # True: run simulator, False: run real car
training = False
generate_path = False if not training else True
# folder = 'training_imgs' 
folder = 'test_imgs'

# os.system('rosservice call /gazebo/reset_simulation')

LOOP_DELAY = 0.000
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

    #init windows
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL) if SIMULATOR else None
    cv.resizeWindow("2D-MAP", 800, 800) if SIMULATOR else None

    # cv.namedWindow('Detection', cv.WINDOW_NORMAL)
    
    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR else None
    car = Automobile_Data(simulator=SIMULATOR, trig_cam=True, trig_gps=True, trig_bno=True, 
                            trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    car.stop()

    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature, folder=folder, 
                                    training=training, noise_std=noise_std)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()


    try:
        car.stop()
        #generate path
        if generate_path:
            path_nodes = [86,428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
                            97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355]
            # path_nodes = [86,110,428,467] #,273,136,321,262]
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
                reference = path.get_reference(car, DESIRED_SPEED, frame=frame, training=training)
                xd, yd, yawd, curv, finished, path_ahead, info = reference
                #controller training data generation        
                controller.curr_data = [xd,yd,yawd,curv,path_ahead,info]
                if finished:
                    print("Reached end of trajectory")
                    car.stop()
                    sleep(1.8)
                    break
                #draw refrence car
                draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))


            ## CONTROL
            if training:
                #car control, unrealistic: uses the true position
                #training
                speed_ref, angle_ref, point_ahead = controller.get_training_control(car, path_ahead, DESIRED_SPEED, curv)
                controller.save_data(frame, folder)
                dist = info[3]
                if dist is not None and 0.0 < dist < 0.5:
                    speed_ref = 0.8 * speed_ref
            else:
                #Neural network control
                lane_info = detect.detect_lane(frame)
                e2, e3, point_ahead = lane_info
                curv = 0.0001
                speed_ref, angle_ref = controller.get_control(e2, e3, curv, DESIRED_SPEED)

                dist = detect.detect_stop_line(frame)

                points_local_path, _ = detect.estimate_local_path(frame)

                # #stopping logic
                if 0.0 < dist < 0.25:
                    print('Slowing down')
                    speed_ref = DESIRED_SPEED * 0.2
                    if 0.0 < dist < 0.1:
                        speed_ref = DESIRED_SPEED * 0.02
                        # print('Stopping')
                        # car.stop()
                        # sleep(0.4)
                        # speed_ref = DESIRED_SPEED
                        # car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
                        # sleep(0.2)

                #Traffic signs
                sign = detect.detect_sign(frame, show_ROI=True)

            ## ACTUATION
            sleep(ACTUATION_DELAY)
            car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
            # car.drive_angle(angle=angle_ref, direct=True)

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
                cv.line(frame, (320//2,479//2), proj, (200, 200, 100), 2)

            #project seq of points ahead
            if training:
                frame, proj = project_onto_frame(frame, car, np.array(controller.seq_points_ahead), False, color=(100, 200, 100))
            else: 
                frame, proj = project_onto_frame(frame, car, points_local_path, False, color=(100, 200, 100))
            
            #draw true car position
            draw_car(tmp, car.x_true, car.y_true, car.yaw, color=(0, 255, 0))

            #draw lateral error
            frame = cv.line(frame, (320,479), (320-int(np.clip(controller.e2*1000, -319, 319)),479), (0,200,0), 3)  
            
            #draw curvature
            # radius = project_curvature(frame, car, curv)

            ## DEBUG INFO
            os.system('cls' if os.name=='nt' else 'clear')
            print(f"x : {car.x_true:.3f} [m], y : {car.y_true:.3f} [m], yaw : {np.rad2deg(car.yaw):.3f} [deg]") 
            print(f"xd: {xd:.3f} [m], yd: {yd:.3f} [m], yawd: {np.rad2deg(yawd):.3f} [deg], curv: {curv:.3f}") if generate_path else None
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {np.rad2deg(controller.e3):.3f}")
            print(f'DESIRED_SPEED = {DESIRED_SPEED:.3f} [m/s], speed_ref = {speed_ref:.3f} [m/s], angle_ref = {np.rad2deg(angle_ref):.3f} [deg]')
            print(f"INFO:\nState: {info[0]}\nNext: {info[1]}\nAction: {info[2]}\nDistance: {info[3]}") if generate_path else None
            print(f'distance line ahead :     {float(dist):.2f} [m]') if not training else None
            print(f'total distance travelled: {car.tot_dist:.2f} [m]')
            print(f'Front sonar distance:     {car.obstacle_ahead_median:.2f} [m]')
            print(f'Net out:\n {lane_info}') if not training else None
            print(f'e_yaw: {e3:.2f} [rad] \ncurv: {100*curv}') if not training else None
            # print(f'Curvature radius = {radius:.2f}')
            print(f'FPS = {1/(time()-loop_start_time):.1f}')
            print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')
            print(f'Sign detection time = {detect.avg_sign_detection_time:.1f} [ms]')

            cv.imshow("Frame preview", frame)
            # cv.imshow('SIGNS ROI', signs_roi)
            # cv.imshow('FRONT ROI', front_obstacle_roi)
            cv.imshow("2D-MAP", tmp) if SIMULATOR else None
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
      