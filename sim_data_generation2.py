
import os, signal, rospy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep, time
from Simulator.src.helper_functions import *
from tqdm import tqdm # progress bar

LAPS = 10
imgs = []
locs = []

STEER_NOSIE_STDS_DEG = np.linspace(0, 20, 11)
POSITION_NOISE_STD = np.linspace(0, 0.1, 11)
file_name = 'saved_tests/sim_ds_'

SHOW_FRAMES = False
TARGET_FPS = 30

spath = np.load('sparcs/sparcs_path_precise.npy').T # load spath from file
#add 2 to all x
spath[:,0] += 2.5
spath[:,1] += 2.5

#create yaw sequence
yaws = np.zeros(spath.shape[0])
for i in range(len(spath)-1):
    yaws[i] = np.arctan2(spath[i+1,1]-spath[i,1], spath[i+1,0]-spath[i,0])
yaws[-1] = yaws[-2]
spath = np.vstack((spath.T, yaws)).T
print(f'spath shape: {spath.shape}')
#decimate path
path = spath[::12]

map = cv.imread('Simulator/src/models_pkg/track/materials/textures/test_VerySmall.png')


#initializations
os.system('rosservice call /gazebo/reset_simulation') 

#car placement in simulator
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
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
    sleep(0.02)

def save_data(imgs,locs, name):
    imgs = np.array(imgs)
    locs = np.array(locs)
    np.savez_compressed(name, imgs=imgs, locs=locs)
    print(f'saved data to {name}')
    # print('not saving, testing')

frame = None
bridge = CvBridge()

rospy.init_node('gazebo_move')
def camera_callback(data) -> None:
    """Receive and store camera frame
    :acts on: self.frame
    """        
    global frame, bridge
    frame = bridge.imgmsg_to_cv2(data, "bgr8")

camera_sub = rospy.Subscriber('/automobile/image_raw', Image, camera_callback)

for sn_std, pos_std in zip(STEER_NOSIE_STDS_DEG, POSITION_NOISE_STD):
    print(f'noise std: {sn_std}, pos std: {pos_std}')
    ds_name = f'{file_name}{sn_std:.0f}.npz'
    if os.path.exists(ds_name):
        print(f'{ds_name} already exists, skipping')
        continue
    imgs = []
    locs = []

    sleep(0.9)

    LAP_Y_TRSH = 2.54 + 2.5
    START_X = 5.03
    START_Y = LAP_Y_TRSH
    START_YAW = np.deg2rad(90.0) + np.pi

    if SHOW_FRAMES:
        cv.namedWindow('map', cv.WINDOW_NORMAL)
        cv.resizeWindow('map', 400, 400)
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 320,  240)

    while frame is None:
        print('waiting for frame')
        sleep(0.01)

    for i in tqdm(range(len(path))):
        loop_start = time()

        xp,yp,yawp = path[i]

        tmp_map = map.copy()

        #add noise  
        y_error = np.random.normal(0, pos_std)
        yaw_error = np.random.normal(0, np.deg2rad(sn_std))
        e = np.array([0, y_error])
        R = np.array([[np.cos(yawp), -np.sin(yawp)], [np.sin(yawp), np.cos(yawp)]])
        e = R @ e
        x = xp + e[0]
        y = yp + e[1]
        yaw = yawp + yaw_error

        #place car
        place_car(x,y,yaw)

        locs.append(np.array([x-2.5, y-2.5, yaw]))

        tmp_frame = frame.copy()
        
        img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (320, 240), interpolation=cv.INTER_AREA)
        imgs.append(img)

        #move car

        if SHOW_FRAMES:
            he, _, _ = get_heading_error(x, y, yaw, spath[:, :2], 0.5)
            draw_car(tmp_map,x,y,yaw)
            tmp_frame = draw_angle(tmp_frame, he)
            tmp_map = tmp_map[int(tmp_map.shape[0]/3):,:int(tmp_map.shape[1]*2/3)]
            cv.imshow('frame', tmp_frame)
            cv.imshow('map', tmp_map)
            if cv.waitKey(1) == 27:
                break



        #calculate fps
        loop_time = time() - loop_start
        fps = 1.0 / loop_time

        # print(f'NOISE: {sn_std}')
        # print(f'x: {x:.2f}, y: {y:.2f}, yaw: {np.rad2deg(yaw):.2f}, fps: {fps:.2f}')


        if loop_time < 1/TARGET_FPS:
            sleep(1/TARGET_FPS - loop_time)

    save_data(imgs, locs, ds_name)

    cv.destroyAllWindows()



