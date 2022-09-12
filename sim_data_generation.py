# %%
from audioop import tomono
import os, signal, rospy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from time import sleep, time
from shutil import get_terminal_size
from Simulator.src.automobile_data_simulator import AutomobileDataSimulator
from Simulator.src.helper_functions import *
from path_nn_controller import PathPlanning, Controller, Detection

LAPS = 60
imgs = []
locs = []

REVERSED_PATH = False #if True the path is Anti-clockwise
STEER_NOISE_STD = np.deg2rad(20.0) # [rad] noise in the steering angle
STEER_FRAME_CHAMGE_MEAN = 10 #avg frames after which the steering noise is changed
STEER_FRAME_CHAMGE_STD = 8 #frames max "deviation"
CONTROLLER_DIST_AHEAD = .35 #pure pursuit controller distance ahead
DESIRED_SPEED = .99# [m/s]
TARGET_FPS = 30.0

path = np.load('sparcs/sparcs_path_precise.npy').T # load path from file
#add 2 to all x
path[:,0] += 2.5
path[:,1] += 2.5

map = cv.imread('Simulator/src/models_pkg/track/materials/textures/test_VerySmall.png')


#initializations
os.system('rosservice call /gazebo/reset_simulation') 
CAR = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=False)

#car placement in simulator
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
rospy.wait_for_service('/gazebo/set_model_state')
state_msg = ModelState()
state_msg.model_name = 'automobile'
state_msg.pose.position.x = 0
state_msg.pose.position.y = 0
state_msg.pose.position.z = 0.032939
state_msg.pose.orientation.x = 0
state_msg.pose.orientation.y = 0
state_msg.pose.orientation.z = 0
state_msg.pose.orientation.w = 0
def place_car(x,y,yaw):
    x,y,yaw = rear2com(x,y,yaw) #convert from rear axle to center of mass
    qx = 0
    qy = 0
    qz = np.sin(yaw/2)
    qw = np.cos(yaw/2)
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y - 15.0
    state_msg.pose.orientation.z = qz
    state_msg.pose.orientation.w = qw
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    resp = set_state(state_msg)
    sleep(0.1)

def save_data(imgs,locs):
    imgs = np.array(imgs)
    locs = np.array(locs)
    np.savez_compressed('sim_dataset0.npz', imgs=imgs, locs=locs)


CONTROLLER = Controller(1.0)

STEER_NOISE = MyRandomGenerator(0.0, STEER_NOISE_STD, STEER_FRAME_CHAMGE_MEAN, STEER_FRAME_CHAMGE_STD)
SPEED_NOISE_STD = 0.0  #[m/s] noise in the speed
SPEED_FRAME_CHAMGE_MEAN = 30 #avg frames after which the speed noise is changed
SPEED_FRAME_CHAMGE_STD = 20 #frames max "deviation"
SPEED_NOISE = MyRandomGenerator(-SPEED_NOISE_STD, SPEED_NOISE_STD, SPEED_FRAME_CHAMGE_MEAN, SPEED_FRAME_CHAMGE_STD, np.random.uniform)

sleep(0.8)

LAP_Y_TRSH = 2.54 + 2.5
START_X = 5.03
START_Y = LAP_Y_TRSH
START_YAW = np.deg2rad(90.0) + np.pi

cv.namedWindow('map', cv.WINDOW_NORMAL)
cv.resizeWindow('map', 400, 400)
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', 320,  240)

#lap logic
lap = -1
lap_time = time()
last_lap_time = 0.0
prev_y = CAR.y_true

#place car at start position
place_car(START_X, START_Y, START_YAW)

while not rospy.is_shutdown():
    loop_start = time()

    x, y, yaw = CAR.x_true, CAR.y_true, CAR.yaw
    frame = CAR.frame
    tmp_map = map.copy()

    locs.append(np.array([x-2.5, y-2.5, yaw]))

    tmp_frame = frame.copy()
    
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.resize(frame, (320, 240), interpolation=cv.INTER_AREA)
    imgs.append(frame)

    #heading error
    he, pa, d = get_heading_error(x,y,yaw,path,CONTROLLER_DIST_AHEAD)

    #calculate control
    steer_angle = CONTROLLER.get_control(he, CONTROLLER_DIST_AHEAD)
    steer_angle = steer_angle + STEER_NOISE.get_noise()

    #move car
    CAR.drive(speed=DESIRED_SPEED, angle=np.rad2deg(steer_angle))

    cv.imshow('frame', tmp_frame)
    draw_car(tmp_map,x,y,yaw)
    tmp_map = tmp_map[int(tmp_map.shape[0]/3):,:int(tmp_map.shape[1]*2/3)]
    cv.imshow('map', tmp_map)
    if cv.waitKey(1) == 27:
        CAR.stop()
        break

    #laps
    curr_y = y
    if (prev_y > LAP_Y_TRSH and curr_y < LAP_Y_TRSH and not REVERSED_PATH) or (prev_y < LAP_Y_TRSH and curr_y > LAP_Y_TRSH and REVERSED_PATH): #count lap
        lap+=1
        lap_time = time()
        last_lap_time = lap_time - last_lap_time
        if lap>=LAPS:
            # saving locations
            #stop the car
            print('Stopping')
            CAR.stop()
            save_data(imgs, locs)
            break
        #invert direction at half the
        if lap==LAPS//2: 
            CAR.stop()
            place_car(START_X, START_Y, START_YAW+np.pi)
            path = np.flip(path, axis=0)

    prev_y = curr_y



    #calculate fps
    loop_time = time() - loop_start
    fps = 1.0 / loop_time

    print(f'x: {x:.2f}, y: {y:.2f}, yaw: {np.rad2deg(yaw):.2f}, fps: {fps:.2f}, lap: {lap+1}/{LAPS}')
    print(f'he: {np.rad2deg(he):.2f}, lap time: {time()-lap_time:.2f}, last lap time: {last_lap_time:.2f}\n')

    if loop_time < 1/TARGET_FPS:
        sleep(1/TARGET_FPS - loop_time)

cv.destroyAllWindows()



