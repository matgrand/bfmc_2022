#!/usr/bin/python3
# from brain import SIMULATOR_FLAG, SHOW_IMGS
SIMULATOR_FLAG = True # True: run simulator, False: run real car

import os, signal
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
from shutil import get_terminal_size

os.system('clear')
print('Main simulator starting...')
if SIMULATOR_FLAG:
    from automobile_data_simulator import AutomobileDataSimulator
    from helper_functions import *
else: #PI
    from control.automobile_data_pi import AutomobileDataPi
    from control.helper_functions import *
from helper_functions import *
from PathPlanning3 import PathPlanning
from controller3 import Controller
from detection import Detection

map = cv.imread('Simulator/src/models_pkg/track/materials/textures/2021_VerySmall.png')
class_list = []
with open("Simulator/data/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()] 

# MAIN CONTROLS
training = True
generate_path = True if training else False
folder = 'training_imgs' 
folder = 'test_imgs'
PATH_NODES = [86, 116,115,116,453,466,465,466,465,466,465,466,465,466,465,466,465,
                428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
                115,116,298,236,274,225,349,298,244,115,116,428,116,428,466,465,466,465,466,465,
                97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355,
                87,428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
                97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355]
PATH_NODES = [86,116,115,116,115,116,115,116,115,110,428,466,249] #,273,136,321,262]
# PATH_NODES = [86, 110, 464, 145, 278]
if training and folder == 'training_imgs':
    print('WARNING, WE ARE ABOUT TO OVERWRITE THE TRAINING DATA! ARE U SURE TO CONTINUE?')
    sleep(5)
    print('Starting in 1 sec...')
    sleep(0.5)
    sleep(0.1)

# os.system('rosservice call /gazebo/reset_simulation')

LOOP_DELAY = 0.00
ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08

FPS_TARGET = 30.0

# PARAMETERS
sample_time = 0.01 # [s]
DESIRED_SPEED = 0.6#0.15# [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain   #pure paralllel k2 = 10 is very good at remaining in the center of the lane
k3 = 1.2 #1.0 yaw error gain .8 with ff 
DISTANCE_AHEAD_PURE_PURSUIT = 0.6 #[m]

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain
noise_std = np.deg2rad(15.0) #np.deg2rad(27.0) # [rad] noise in the steering angle


if __name__ == '__main__':

    #init windows
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL) if SIMULATOR_FLAG else None
    cv.resizeWindow("2D-MAP", 800, 800) if SIMULATOR_FLAG else None
    cv.namedWindow("Frame preview", cv.WINDOW_NORMAL) if SIMULATOR_FLAG else None
    cv.resizeWindow("Frame preview", 640, 480) if SIMULATOR_FLAG else None

    # cv.namedWindow('Detection', cv.WINDOW_NORMAL)
    
    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') if SIMULATOR_FLAG else None
    if SIMULATOR_FLAG:
        car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    else:
        car = AutomobileDataPi(trig_cam=True, trig_gps=False, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    car.stop()


    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        os.system('rosservice call gazebo/pause_physics') if SIMULATOR_FLAG else None 
        cv.destroyAllWindows()
        sleep(.99)
        exit()
    signal.signal(signal.SIGINT, handler)

    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, dist_point_ahead=DISTANCE_AHEAD_PURE_PURSUIT, ff=ff_curvature, folder=folder, 
                                    training=training, noise_std=noise_std)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()


    try:
        car.stop()
        #generate path
        if generate_path:
            path_nodes = PATH_NODES
            path.generate_path_passing_through(path_nodes, path_step_length)
            path.draw_path()
            path.print_path_info()

        os.system('rosservice call gazebo/unpause_physics') if SIMULATOR_FLAG else None 

        car.drive_speed(DESIRED_SPEED)

        while not rospy.is_shutdown():
            loop_start_time = time()

            tmp = np.copy(map)
            # Get the image from the camera
            frame = car.frame.copy()
            sleep(VISION_DELAY)

            #FOLLOW predefined trajectory
            if generate_path:
                CAR_LENGTH = 0.35
                MAX_VISIBLE_DIST = 0.65
                MAX_VALUE = 1.0
                NO_VALUE = 10
                reference = path.get_reference(car, DESIRED_SPEED, frame=frame, training=training, limit_search_to=int(100*(MAX_VALUE+CAR_LENGTH)))
                xd, yd, yawd, curv, finished, path_ahead, info = reference
                dist = info[3] #distance from stopline
                if int(dist*100)+5 < len(path_ahead):
                    closest_stopline = path.path_stoplines[np.argmin(np.linalg.norm(path.path_stoplines - path_ahead[int(dist*100)], axis=1))]
                    angle_stopline = get_yaw_closest_axis(np.arctan2(path_ahead[int(dist*100)+5][1]-path_ahead[int(dist*100)-5][1],
                                                                                            path_ahead[int(dist*100)+5][0]-path_ahead[int(dist*100)-5][0]))
                    tmp = cv.circle(tmp, mR2pix(closest_stopline), 50, (255,0,255), 3) #draw stopline on map
                    tmp = cv.circle(tmp, mR2pix(path_ahead[0]), 10, (0,255,0), 2) #draw car center reference wrt to path (com)
                    frame, proj = project_onto_frame(frame, car, closest_stopline, True, color=(250, 0, 250), thickness=10) #and on frame
                    #generate training points for stopline localisation
                    rot_matrix = np.array([[np.cos(car.yaw), -np.sin(car.yaw)], [np.sin(car.yaw), np.cos(car.yaw)]])
                    cp_w = np.array([car.x_true, car.y_true]) #cp = car position, w = world frame
                    cp_sl = cp_w - closest_stopline #translate to stop line frame
                    cp_sl = cp_sl @ rot_matrix #rotate to car frame
                    car_angle_to_stopline = diff_angle(car.yaw, angle_stopline)
                    slp_cf = -cp_sl # stop line position in car frame
                    stopline_x_train = slp_cf[0] - CAR_LENGTH
                    if stopline_x_train < 0.0: 
                        stopline_x_train = MAX_VALUE + np.random.uniform(-0.1, 0.1)
                        car_angle_to_stopline = 0.0
                        print("stopline_x_train < 0.0")
                    elif stopline_x_train > MAX_VISIBLE_DIST:
                        print("stopline_x_train > MAX_VISIBLE_DIST")
                        stopline_x_train = MAX_VISIBLE_DIST + np.random.uniform(0.0, 0.15)
                        car_angle_to_stopline = diff_angle(car.yaw, yawd)
                    stopline_y_train = slp_cf[1]
                    stopline_yaw_train = car_angle_to_stopline
                    print(f'Stopline -> x: {stopline_x_train}, y: {stopline_y_train}, yaw: {stopline_yaw_train}')
                
                    #project onto frame
                    frame, _ = project_stopline(frame, car, stopline_x_train, stopline_y_train, car_angle_to_stopline, color=(0,200,0))
                else:
                    stopline_x_train = NO_VALUE
                    stopline_y_train = 0.0 
                    stopline_yaw_train = diff_angle(car.yaw, yawd)
                #controller training data generation        
                controller.curr_data = [xd,yd,yawd,curv,path_ahead,stopline_x_train,stopline_y_train,stopline_yaw_train,info]
                if finished:
                    print("Reached end of trajectory")
                    car.stop()
                    os.system('rosservice call gazebo/pause_physics') if SIMULATOR_FLAG else None 
                    sleep(1.8)
                    break
                #draw refrence car
                draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))


            ## CONTROL
            if training:
                #car control, unrealistic: uses the true position
                #training
                speed_ref, angle_ref, point_ahead = controller.get_training_control(car, path_ahead, DESIRED_SPEED, curv)
                controller.save_data(car.frame, folder)
                # dist = info[3]
                # if dist is not None and 0.3 < dist < 0.8:
                #     speed_ref = 0.4 * speed_ref
            else:
                #Neural network control
                # lane_info = detect.detect_lane(frame)
                # e2, e3, point_ahead = lane_info
                lane_info = e3, point_ahead = detect.detect_lane_ahead(frame)
                e2 = 0.0
                curv = 0.0001
                speed_ref, angle_ref = controller.get_control(e2, e3, curv, DESIRED_SPEED)

                dist = detect.detect_stop_line(frame)

                points_local_path, _ = detect.estimate_local_path(frame)

                # #stopping logic
                if 0.25 < dist < 0.65:
                    print('Slowing down')
                    # frame, _ = project_stopline(frame, car, dist, angle_stopline, color=(0,200,0))
                    speed_ref = DESIRED_SPEED * 0.4
                    if 0.0 < dist < 0.1:
                        speed_ref = DESIRED_SPEED * 0.2
                        # print('Stopping')
                        # car.stop()
                        # sleep(0.4)
                        # speed_ref = DESIRED_SPEED
                        # car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
                        # sleep(0.2)

                #Traffic signs
                # sign = detect.detect_sign(frame, show_ROI=True)

            ## ACTUATION
            sleep(ACTUATION_DELAY)
            # car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
            car.drive_angle(angle=np.rad2deg(angle_ref))

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
                _ = project_curvature(frame, car, curv)
                pass
            else: 
                frame, proj = project_onto_frame(frame, car, points_local_path, False, color=(100, 200, 100))
            
            #draw true car position
            draw_car(tmp, car.x_true, car.y_true, car.yaw, color=(0, 255, 0))

            #draw lateral error
            frame = cv.line(frame, (320,479), (320-int(np.clip(controller.e2*1000, -319, 319)),479), (0,200,0), 3)  
            
            #draw curvature
            # radius = project_curvature(frame, car, curv)

            ## DEBUG INFO
            print('\n' * get_terminal_size().lines, end='')
            print('\033[F' * get_terminal_size().lines, end='')
            if dist is None: dist=10
            # print(f'Stop line distance: {dist:.2f}, Angle: {np.rad2deg(angle_to_stopline):.2f}') #if not training else None
            print(f'Curvature: {curv:.2f}')
            print(f"x : {car.x_true:.3f} [m], y : {car.y_true:.3f} [m], yaw : {np.rad2deg(car.yaw):.3f} [deg]") 
            print(f"xd: {xd:.3f} [m], yd: {yd:.3f} [m], yawd: {np.rad2deg(yawd):.3f} [deg], curv: {curv:.3f}") if generate_path else None
            print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {np.rad2deg(controller.e3):.3f}")
            print(f'DESIRED_SPEED = {DESIRED_SPEED:.3f} [m/s], speed_ref = {speed_ref:.3f} [m/s], angle_ref = {np.rad2deg(angle_ref):.3f} [deg]')
            print(f"INFO:\nState: {info[0]}\nNext: {info[1]}\nAction: {info[2]}\nDistance: {info[3]}") if generate_path else None
            print(f'distance line ahead :     {float(dist):.2f} [m]') if not training else None
            print(f'total distance travelled: {car.encoder_distance:.2f} [m]')
            print(f'Front sonar distance:     {car.filtered_sonar_distance:.2f} [m]')
            print(f'Net out:\n {lane_info}') if not training else None
            print(f'e_yaw: {e3:.2f} [rad] \ncurv: {100*curv}') if not training else None
            # print(f'Curvature radius = {radius:.2f}')
            print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')

            cv.imshow("Frame preview", frame)
            cv.imwrite(f'test_imgs/frame{int(loop_start_time*1000)}.png', frame)
            frame = frame[int(frame.shape[0]*(2/5)):,:]
            cv.imshow("Stopline", frame)
            # cv.imshow('SIGNS ROI', signs_roi)
            # cv.imshow('FRONT ROI', front_obstacle_roi)
            cv.imshow("2D-MAP", tmp) if SIMULATOR_FLAG else None
            # cv.imwrite(f'test_imgs/map{int(loop_start_time*1000)}.png', tmp) #very heavy
            key = cv.waitKey(1)
            if key == 27:
                car.stop()
                sleep(.5)
                cv.destroyAllWindows()
                break

            # sleep(LOOP_DELAY)
            curr_time = time()
            loop_time = curr_time - loop_start_time
            print(f'FPS = {1/(time()-loop_start_time):.0f}, capped at: {FPS_TARGET:.0f}')
            if loop_time < 1/FPS_TARGET:
                print(f'time to sleep: {1000*(1/FPS_TARGET - loop_time):.3f} [ms]')
                sleep(1/FPS_TARGET - loop_time)

    except KeyboardInterrupt:
        print("Shutting down")
        car.stop()
        sleep(.5)
        cv.destroyAllWindows()
        exit(0)
    except rospy.ROSInterruptException:
        pass
      