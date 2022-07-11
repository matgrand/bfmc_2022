#!/usr/bin/python3
import os, signal, rospy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from time import sleep, time
from shutil import get_terminal_size
from automobile_data_simulator import AutomobileDataSimulator
from helper_functions import *
from PathPlanning import PathPlanning
from data_generation_controller import Controller
from detection import Detection

ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08
FPS_TARGET = 30.0

## FOLDERS
# FOLDER = 'training_imgs' 
FOLDER = 'test_imgs'
if FOLDER == 'training_imgs':
    print('WARNING, WE ARE ABOUT TO OVERWRITE THE TRAINING DATA! ARE U SURE TO CONTINUE?')
    sleep(5)
    print('Starting in 1 sec...')
    sleep(0.5)
    sleep(0.1)

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
PATH_NODES = [86,116,115,116,115,116,115,116,115,110,428,466,249] #,273,136,321,262]
PATH_NODES = [86,427,257,110,348,85]
PATH = PathPlanning() 
#generate path
PATH.generate_path_passing_through(PATH_NODES)
PATH.draw_path()
PATH.print_path_info()

#############################################################################################################################################################
## CONTROLLER
DISTANCE_AHEAD_PURE_PURSUIT = 0.6#0.6 #[m]
K = 1.2 #1.0 yaw error gain .8 with ff 
CONTROLLER = Controller(K)
STEER_NOISE_STD = np.deg2rad(5.0) #np.deg2rad(15.0) # [rad] noise in the steering angle
STEER_FRAME_CHAMGE_MEAN = 10 #avg frames after which the steering noise is changed
STEER_FRAME_CHAMGE_STD = 8 #frames max "deviation"
STEER_NOISE = MyRandomGenerator(0.0, STEER_NOISE_STD, STEER_FRAME_CHAMGE_MEAN, STEER_FRAME_CHAMGE_STD)
DESIRED_SPEED = .6#0.15# [m/s]
SPEED_NOISE_STD = 0*0.5 #[m/s] noise in the speed
SPEED_FRAME_CHAMGE_MEAN = 30 #avg frames after which the speed noise is changed
SPEED_FRAME_CHAMGE_STD = 20 #frames max "deviation"
SPEED_NOISE = MyRandomGenerator(-SPEED_NOISE_STD, SPEED_NOISE_STD, SPEED_FRAME_CHAMGE_MEAN, SPEED_FRAME_CHAMGE_STD, np.random.uniform)

#############################################################################################################################################################
## DETECTION
DETECTION = Detection()

#############################################################################################################################################################
## PLOTS
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
        tmp = np.copy(PATH.map)
        # Get the image from the camera
        frame = CAR.frame.copy()
        sleep(VISION_DELAY)

        #############################################################################################################################################################
        ## REFERENCE TRAJECTORY
        reference = PATH.get_reference(CAR, DISTANCE_AHEAD_PURE_PURSUIT)
        heading_error, point_ahead, seq_heading_errors, seq_points_ahead, path_ahead, finished = reference

        #############################################################################################################################################################
        ## NEURAL NETWORK ESTIMATE
        nn_estimate = DETECTION.detect_lane_ahead(frame)
        nn_estimate = DETECTION.detect_lane(frame)
        est_heading_error, est_point_ahead = nn_estimate

        #############################################################################################################################################################
        ## CONTROL
        steer_angle = CONTROLLER.get_control(heading_error, DISTANCE_AHEAD_PURE_PURSUIT)

        #############################################################################################################################################################
        ## ACTUATION
        sleep(ACTUATION_DELAY)
        # CAR.drive(speed=speed_ref, angle=np.rad2deg(steer_angle))
        CAR.drive_angle(angle=np.rad2deg(steer_angle + STEER_NOISE.get_noise()))
        CAR.drive_speed(speed=DESIRED_SPEED + SPEED_NOISE.get_noise())

        #############################################################################################################################################################
        ## EXIT CONDITON
        if finished:
            print("Reached end of trajectory")
            CAR.stop()
            os.system('rosservice call gazebo/pause_physics') 
            sleep(.8)
            exit()

        #############################################################################################################################################################
        ## VISUALIZATIONS
        #draw true CAR position
        draw_car(tmp, CAR.x_true, CAR.y_true, CAR.yaw, color=(0, 255, 0))

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

            # #draw line from bottmo half to proj
            # cv.line(frame, (320//2,479//2), proj, (200, 200, 100), 2)
            # cv.line(frame, (320//2,479//2), est_proj, (200, 100, 200), 2)

        #project seq of points ahead
        # frame, proj = project_onto_frame(frame, CAR, np.array(CONTROLLER.seq_points_ahead), False, color=(100, 200, 100))

        #############################################################################################################################################################
        ## PLOTS
        #plot heading error and estimate
        if loop_cnt % 1 == 0:
            heading_errors = np.roll(heading_errors, -1)
            heading_errors[-1] = heading_error
            est_heading_errors = np.roll(est_heading_errors, -1)
            est_heading_errors[-1] = est_heading_error
        
        if loop_cnt % 10 == 0:
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
        print(f'loop_cnt = {loop_cnt}')

        # show imgs
        cv.imshow("Frame preview", frame)
        cv.imshow("2D-MAP", tmp) 
        cv.waitKey(1)

        curr_time = time()
        loop_time = curr_time - loop_start_time
        print(f'FPS = {1/(time()-loop_start_time):.0f}, capped at: {FPS_TARGET:.0f}')
        if loop_time < 1/FPS_TARGET:
            print(f'time to sleep: {1000*(1/FPS_TARGET - loop_time):.3f} [ms]')
            sleep(1/FPS_TARGET - loop_time)


      