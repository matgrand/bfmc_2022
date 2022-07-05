#!/usr/bin/python3
import os, signal, rospy
import cv2 as cv
import numpy as np
from time import sleep, time
from shutil import get_terminal_size

os.system('clear')
print('Main simulator starting...')
from automobile_data_simulator import AutomobileDataSimulator
from helper_functions import *
from PathPlanning import PathPlanning
from controller import Controller
from detection import Detection

LOOP_DELAY = 0.00
ACTUATION_DELAY = 0.0#0.15
VISION_DELAY = 0.0#0.08

FPS_TARGET = 30.0

map = cv.imread('Simulator/src/models_pkg/track/materials/textures/2021_VerySmall.png')

# MAIN CONTROLS
folder = 'training_imgs' 
folder = 'test_imgs'
# PATH_NODES = [86, 116,115,116,453,466,465,466,465,466,465,466,465,466,465,466,465,
#                 428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
#                 115,116,298,236,274,225,349,298,244,115,116,428,116,428,466,465,466,465,466,465,
#                 97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355,
#                 87,428,273,136,321,262,105,350,94,168,136,321,262,373,451,265,145,160,353,94,127,91,99,
#                 97,87,153,275,132,110,320,239,298,355,105,113,145,110,115,297,355]
# PATH_NODES = [86,116,115,116,115,116,115,116,115,110,428,466,249] #,273,136,321,262]
# PATH_NODES = [86, 110, 464, 145, 278]
PATH_NODES = [86,255,110,346,85]

if folder == 'training_imgs':
    print('WARNING, WE ARE ABOUT TO OVERWRITE THE TRAINING DATA! ARE U SURE TO CONTINUE?')
    sleep(5)
    print('Starting in 1 sec...')
    sleep(0.5)
    sleep(0.1)


# PARAMETERS
DESIRED_SPEED = 2#0.15# [m/s]
# CONTROLLER
k1 = 0.0 #0.0 gain error parallel to direction (speed)
k2 = 0.0 #0.0 perpenddicular error gain   #pure paralllel k2 = 10 is very good at remaining in the center of the lane
k3 = 1.2 #1.0 yaw error gain .8 with ff 
DISTANCE_AHEAD_PURE_PURSUIT = 0.6 #[m]

#dt_ahead = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain
noise_std = np.deg2rad(0.0) #np.deg2rad(15.0) # [rad] noise in the steering angle


if __name__ == '__main__':

    #init windows
    cv.namedWindow("2D-MAP", cv.WINDOW_NORMAL) 
    cv.resizeWindow("2D-MAP", 800, 800) 
    cv.namedWindow("Frame preview", cv.WINDOW_NORMAL) 
    cv.resizeWindow("Frame preview", 640, 480) 
    
    # init the car data
    os.system('rosservice call /gazebo/reset_simulation') 
    car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)
    car.stop()

    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        os.system('rosservice call gazebo/pause_physics')  
        cv.destroyAllWindows()
        sleep(.99)
        exit()
    signal.signal(signal.SIGINT, handler)

    # init trajectory
    path = PathPlanning(map) 

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, dist_point_ahead=DISTANCE_AHEAD_PURE_PURSUIT, ff=ff_curvature, folder=folder, 
                                    training=True, noise_std=noise_std)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()

    car.stop()
    #generate path
    path.generate_path_passing_through(PATH_NODES)
    path.draw_path()
    path.print_path_info()
    print('Path generated!')

    os.system('rosservice call gazebo/unpause_physics') 

    car.drive_speed(DESIRED_SPEED)

    while not rospy.is_shutdown():
        loop_start_time = time()

        tmp = np.copy(map)
        # Get the image from the camera
        frame = car.frame.copy()
        sleep(VISION_DELAY)

        #############################################################################################################################################################
        ## REFERENCE TRAJECTORY
        CAR_LENGTH = 0.35
        MAX_VISIBLE_DIST = 0.65
        MAX_VALUE = 1.0
        NO_VALUE = 10
        reference = path.get_reference(car, DESIRED_SPEED, limit_search_to=int(100*(MAX_VALUE+CAR_LENGTH)))
        xd, yd, yawd, curv, finished, path_ahead, info = reference

        if finished:
            print("Reached end of trajectory")
            car.stop()
            os.system('rosservice call gazebo/pause_physics') 
            sleep(1.8)
            exit()
        #draw reference car
        draw_car(tmp, xd, yd, yawd, color=(0, 0, 255))

        #############################################################################################################################################################
        ## CONTROL
        speed_ref, angle_ref, point_ahead = controller.get_training_control(car, path_ahead, DESIRED_SPEED, curv)
        controller.save_data(car.frame, folder)

        #############################################################################################################################################################
        ## ACTUATION
        sleep(ACTUATION_DELAY)
        # car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
        car.drive_angle(angle=np.rad2deg(angle_ref))

        #############################################################################################################################################################
        ## VISUALIZATIONS
        #project path ahead
        frame, proj = project_onto_frame(frame, car, path_ahead, color=(0,0,100))

        #project point ahead
        frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
        if proj is not None:
            #convert proj to cv2 point
            proj = (int(proj[0]), int(proj[1]))
            #draw line from bottmo half to proj
            cv.line(frame, (320//2,479//2), proj, (200, 200, 100), 2)

        #project seq of points ahead
        # frame, proj = project_onto_frame(frame, car, np.array(controller.seq_points_ahead), False, color=(100, 200, 100))
        # _ = project_curvature(frame, car, curv)

        #draw true car position
        draw_car(tmp, car.x_true, car.y_true, car.yaw, color=(0, 255, 0))

        #draw lateral error
        frame = cv.line(frame, (320,479), (320-int(np.clip(controller.e2*1000, -319, 319)),479), (0,200,0), 3)  
        
        #############################################################################################################################################################
        ## DEBUG INFO
        print('\n' * get_terminal_size().lines, end='')
        print('\033[F' * get_terminal_size().lines, end='')
        # print(f'Stop line distance: {dist:.2f}, Angle: {np.rad2deg(angle_to_stopline):.2f}') #if not training else None
        print(f'Curvature: {curv:.2f}')
        print(f"x : {car.x_true:.3f} [m], y : {car.y_true:.3f} [m], yaw : {np.rad2deg(car.yaw):.3f} [deg]") 
        print(f"xd: {xd:.3f} [m], yd: {yd:.3f} [m], yawd: {np.rad2deg(yawd):.3f} [deg], curv: {curv:.3f}") 
        print(f"e1: {controller.e1:.3f}, e2: {controller.e2:.3f}, e3: {np.rad2deg(controller.e3):.3f}")
        print(f'DESIRED_SPEED = {DESIRED_SPEED:.3f} [m/s], speed_ref = {speed_ref:.3f} [m/s], angle_ref = {np.rad2deg(angle_ref):.3f} [deg]')
        print(f'total distance travelled: {car.encoder_distance:.2f} [m]')
        print(f'Front sonar distance:     {car.filtered_sonar_distance:.2f} [m]')
        # print(f'Curvature radius = {radius:.2f}')
        print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')

        # show imgs
        cv.imshow("Frame preview", frame)
        cv.imwrite(f'test_imgs/frame{int(loop_start_time*1000)}.png', frame)
        frame = frame[int(frame.shape[0]*(2/5)):,:]
        cv.imshow("Stopline", frame)
        cv.imshow("2D-MAP", tmp) 
        cv.waitKey(1)


        # sleep(LOOP_DELAY)
        curr_time = time()
        loop_time = curr_time - loop_start_time
        print(f'FPS = {1/(time()-loop_start_time):.0f}, capped at: {FPS_TARGET:.0f}')
        if loop_time < 1/FPS_TARGET:
            print(f'time to sleep: {1000*(1/FPS_TARGET - loop_time):.3f} [ms]')
            sleep(1/FPS_TARGET - loop_time)


      