#!/usr/bin/python3
import os, signal, rospy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from time import sleep, time
from shutil import get_terminal_size
from src.automobile_data_simulator import AutomobileDataSimulator
from src.helper_functions import *
from path_nn_controller import PathPlanning, Controller, Detection

EVAL_MODE = True
# EVAL_MODE = False

SHOW_PLOTS = EVAL_MODE #and False

ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08
FPS_TARGET = 30.0

## FOLDERS
FOLDER = 'Simulator/training/' 

#############################################################################################################################################################
## CAR
os.system('rosservice call /gazebo/reset_simulation') 
CAR = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)

#############################################################################################################################################################
## PATH PLANNING
PATH_NODES = [86, 116,115,116,453,466,465,466,465,466,465,466,465,466,465,466,465,
                428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
                115,116,298,236,274,225,349,298,244,115,116,428,116,428,466,465,466,465,466,465,
                97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355,
                87,428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
                97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355]
# PATH_NODES = [86,116,115,116,115,116,115,116,115,110,428,466,249] #,273,136,321,262]
PATH_NODES = [86,427,257,110,348,85]
# PATH_NODES = [86,127]
PATH = PathPlanning() 
#generate path
PATH.generate_path_passing_through(PATH_NODES)
PATH.draw_path()
PATH.print_path_info()

#############################################################################################################################################################
## DATA GENERATION
DISTANCE_AHEAD_PURE_PURSUIT = 0.6#0.6 #[m]
PATH_AHEAD_DISTANCES = [.4, .6, .8, 1.0, 1.2, 1.4] #[m] must be in increasing order
speed_log, steer_log, he_log, seq_he_log, seq_rel_angles_log = [], [], [], [], [] # empty lists for data storing
frames = [] #all frames

#############################################################################################################################################################
## CONTROLLER
K = 1.2 #1.0 yaw error gain .8 with ff 
CONTROLLER = Controller(K)
STEER_NOISE_STD = np.deg2rad(18.0) if not EVAL_MODE else np.deg2rad(5.0) # [rad] noise in the steering angle
STEER_FRAME_CHAMGE_MEAN = 10 #avg frames after which the steering noise is changed
STEER_FRAME_CHAMGE_STD = 8 #frames max "deviation"
STEER_NOISE = MyRandomGenerator(0.0, STEER_NOISE_STD, STEER_FRAME_CHAMGE_MEAN, STEER_FRAME_CHAMGE_STD)
DESIRED_SPEED = .6#0.15# [m/s]
SPEED_NOISE_STD = 0.5 if not EVAL_MODE else 0.0 #[m/s] noise in the speed
SPEED_FRAME_CHAMGE_MEAN = 30 #avg frames after which the speed noise is changed
SPEED_FRAME_CHAMGE_STD = 20 #frames max "deviation"
SPEED_NOISE = MyRandomGenerator(-SPEED_NOISE_STD, SPEED_NOISE_STD, SPEED_FRAME_CHAMGE_MEAN, SPEED_FRAME_CHAMGE_STD, np.random.uniform)

#############################################################################################################################################################
## DETECTION
DETECTION = Detection()

#############################################################################################################################################################
## PLOTS
if SHOW_PLOTS:
    heading_errors = np.zeros(int(FPS_TARGET*3), dtype=np.float32)
    est_heading_errors = np.zeros(int(FPS_TARGET*3), dtype=np.float32)
    fig, ax = plt.subplots()
    ax.cla()
    he_line, = ax.plot(heading_errors, 'b-')
    ehe_line, = ax.plot(est_heading_errors, 'r-')
    ax.set_ylim(-np.deg2rad(30), np.deg2rad(30))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Heading error [rad]')
    ax.grid(True)
    plt.show(block=False)
    plt.pause(0.001)



if __name__ == '__main__':
    #init windows
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL) 
    cv.resizeWindow("2D-MAP", 800, 800) 
    cv.namedWindow("Frame preview", cv.WINDOW_NORMAL) 
    cv.resizeWindow("Frame preview", 640, 480) 

    CAR.stop()

    #stop the CAR with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        CAR.stop()
        os.system('rosservice call gazebo/pause_physics')  
        cv.destroyAllWindows()
        sleep(.99)
        exit()
    signal.signal(signal.SIGINT, handler)

    CAR.stop()
    loop_cnt = 0

    os.system('rosservice call gazebo/unpause_physics') 

    while not rospy.is_shutdown():
        loop_start_time = time()
        loop_cnt += 1

        #############################################################################################################################################################
        ## FRAME 
        tmap = np.copy(PATH.map)
        # Get the image from the camera
        frame = CAR.frame.copy()
        #blur
        frame = cv.blur(frame, (2,2))
        frame = cv.resize(frame, (320, 240), interpolation=cv.INTER_AREA)
        frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

        sleep(VISION_DELAY)

        #############################################################################################################################################################
        ## REFERENCE TRAJECTORY
        reference = PATH.get_reference(CAR, DISTANCE_AHEAD_PURE_PURSUIT, PATH_AHEAD_DISTANCES)
        heading_error, point_ahead, seq_heading_errors, seq_relative_angles, path_ahead, finished = reference
        #logging for data saving
        he_log.append(heading_error)
        seq_he_log.append(seq_heading_errors)
        seq_rel_angles_log.append(seq_relative_angles)
        speed_log.append(CAR.speed)
        steer_log.append(CAR.steer)
        
        #############################################################################################################################################################
        ## NEURAL NETWORK ESTIMATE
        nn_estimate = DETECTION.detect_lane_ahead(frame)
        # nn_estimate = DETECTION.detect_lane(frame)
        est_heading_error, est_point_ahead = nn_estimate

        #############################################################################################################################################################
        ## CONTROL
        steer_angle = CONTROLLER.get_control(heading_error, DISTANCE_AHEAD_PURE_PURSUIT)
        steer_angle = steer_angle + STEER_NOISE.get_noise()

        #############################################################################################################################################################
        ## ACTUATION
        sleep(ACTUATION_DELAY)
        # CAR.drive(speed=speed_ref, angle=np.rad2deg(steer_angle))
        CAR.drive_angle(angle=np.rad2deg(steer_angle))
        CAR.drive_speed(speed=DESIRED_SPEED + SPEED_NOISE.get_noise())

        #############################################################################################################################################################
        ## EXIT CONDITON
        if finished:
            print("Reached end of trajectory")
            CAR.stop()
            os.system('rosservice call gazebo/pause_physics') 
            sleep(.2)
            if not EVAL_MODE:
                #saving data
                np.save(FOLDER+'speed_log', speed_log)
                np.save(FOLDER+'steer_log', steer_log)
                np.save(FOLDER+'he_log', he_log)
                np.save(FOLDER+'seq_he_log', seq_he_log)
                np.save(FOLDER+'seq_rel_angles_log', seq_rel_angles_log)
                # np.savez_compressed(FOLDER+'frames', frames)
                np.save(FOLDER+'frames', frames) 
                print("Data saved")
            else: 
                print('Evaluation mode, not saving any datapoints')
            exit()

        #############################################################################################################################################################
        ## VISUALIZATIONS
        #draw true CAR position
        draw_car(tmap, CAR.x_true, CAR.y_true, CAR.yaw, color=(0, 255, 0))

        #project PATH ahead
        frame, proj = project_onto_frame(frame, CAR, path_ahead, color=(0,0,100))

        #project point ahead
        frame, proj = project_onto_frame(frame, CAR, point_ahead, False, color=(200, 200, 100))
        frame, est_proj = project_onto_frame(frame, CAR, est_point_ahead, False, color=(200, 100, 200))
        if proj is not None:
            #convert proj to cv2 point
            proj = (int(proj[0]), int(proj[1]))
            est_proj = (int(est_proj[0]), int(est_proj[1]))
            draw_angle(frame, heading_error, color=(200-50, 200-50, 100-50))
            draw_angle(frame, est_heading_error, color=(200-50, 100-50, 200-50))

        #project seq of points ahead
        frame = draw_seq_points_ahead(frame, CAR, seq_relative_angles, PATH_AHEAD_DISTANCES, color=(0,0,255))

        #############################################################################################################################################################
        ## PLOTS
        if SHOW_PLOTS:
            if loop_cnt % 3 == 0:
                heading_errors = np.roll(heading_errors, -1)
                heading_errors[-1] = heading_error
                est_heading_errors = np.roll(est_heading_errors, -1)
                est_heading_errors[-1] = est_heading_error
            
            if loop_cnt % 15 == 0:
                #update plot data
                he_line.set_ydata(heading_errors)
                ehe_line.set_ydata(est_heading_errors)
                plt.pause(0.001)
                plt.draw()

        #############################################################################################################################################################
        ## DEBUG INFO
        print('\n' * get_terminal_size().lines, end='')
        print('\033[F' * get_terminal_size().lines, end='')
        print(f'speed = {CAR.speed:.2f} [m/s],  steer = {CAR.steer:.2f} [deg]')
        print(f"x : {CAR.x_true:.3f} [m], y : {CAR.y_true:.3f} [m], yaw : {np.rad2deg(CAR.yaw):.3f} [deg]") 
        print(f'total distance travelled: {CAR.encoder_distance:.2f} [m]')
        print(f'point_ahead:     {point_ahead}, heading_error:     {np.rad2deg(heading_error):.3f} [deg]')
        print(f'est_point_ahead: {est_point_ahead}, est_heading_error: {np.rad2deg(est_heading_error):.3f} [deg]')
        print(f'seq_heading_errors: {seq_heading_errors}')
        print(f'loop_cnt = {loop_cnt}')

        # show imgs
        cv.imshow("Frame preview", frame)
        cv.imshow("2D-MAP", tmap) 
        cv.waitKey(1)

        curr_time = time()
        loop_time = curr_time - loop_start_time
        print(f'FPS = {1/(time()-loop_start_time):.0f}, capped at: {FPS_TARGET:.0f}')
        if loop_time < 1/FPS_TARGET:
            print(f'time to sleep: {1000*(1/FPS_TARGET - loop_time):.3f} [ms]')
            sleep(1/FPS_TARGET - loop_time)


      