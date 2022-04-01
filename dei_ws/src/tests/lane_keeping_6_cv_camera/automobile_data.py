#!/usr/bin/python3
# Functional libraries
import rospy
import numpy as np
from time import sleep, time
from math import cos,sin,pi
from cv_bridge import CvBridge
import json
from helper_functions import *
import collections

# Messages for topic and service communication
from std_msgs.msg import String, Float32
from utils.msg import IMU
from utils.msg import localisation
#from utils.srv import subscribing
from sensor_msgs.msg import Image
import os


# State estimation
#from estimation import EKFCar

# Vehicle driving parameters
MIN_SPEED = -0.3                    # [m/s]     minimum speed
MAX_SPEED = 0.5                     # [m/s]     maximum speed
MAX_STEER = 27.0                    # [deg]     maximum steering angle
MAX_DSTEER = np.deg2rad(40.0)       # [rad/s]   maximum steering speed

# Vehicle parameters
LENGTH = 0.45  			            # [m]       car body length
WIDTH = 0.18   			            # [m]       car body width
BACKTOWHEEL = 0.10  		        # [m]       distance of the wheel and the car body
WHEEL_LEN = 0.03  			        # [m]       wheel raduis
WHEEL_WIDTH = 0.03  		        # [m]       wheel thickness
WB = 0.26  			                # [m]       wheelbase
GBOX_RATIO = 30                     # []        gearbox ratio (should be 9*pi)

# Camera parameters
FRAME_WIDTH = 640           # [pix]     frame width
FRAME_HEIGHT = 480          # [pix]     frame height
# position and orientation wrt the car frame
CAM_X = 0.0                 # [m]
CAM_Y = 0.0                 # [m]
CAM_Z = 0.2                 # [m]
CAM_ROLL = 0.0              # [rad]
CAM_PITCH = np.deg2rad(20)  # [rad]
CAM_YAW =  0.0              # [rad]
CAM_FOV = 1.085594795       # [rad]
CAM_F = 1.0                 # []        focal length
# scaling factors
CAM_Sx = 10.0               # [pix/m]
CAM_Sy = 10.0               # [pix/m]
CAM_Ox = 10.0               # [pix]
CAM_Oy = 10.0               # [pix]
CAM_K = np.array([[CAM_F*CAM_Sx,      0.0,            CAM_Ox],
                  [ 0.0,              CAM_F*CAM_Sy,   CAM_Oy],
                  [ 0.0,              0.0,            1.0]])
assert CAM_K.shape == (3,3) # make sure the matrix is 3x3

# PID activation
PID_ENABLE = True

# EKF constants
EST_X0 = np.array([1.5, 15-6.0,0.9*pi]).reshape(-1,1)       # [m,m,rad] initial state of the EKF estimator
EST_REL_POS_TS = 0.001                                       # [s]       relative positioning sampling time (must be > max(IMU:Ts,ENC:Ts))
EST_EKF_STATE_TS = 0.01                                     # [s]       Extended Kalman Filter sampling time (must be > max(GPS:Ts,IMU:Ts))

# Topics for feedback from Nucleo board
FDBK_STEER_TOPIC = '/steer_feedback'
FDBK_BREAK_TOPIC = '/break_feedback'
FDBK_SPEED_TOPIC = '/speed_feedback'
FDBK_ENC_TOPIC = '/encoder_feedback'

# ROS constants
ROS_PAUSE = 0.0001           # [s]   pause after publicing on a ROS topic
ROS_REPEAT = 50             # []    number of times to send enbaling commands, to be sure command arrives

# Control compensation
SPEED_COMPENSATION = 1.0
STEER_COMPENSATION = 1.0 #0.79

#controls paramters
# STEER_UPDATE_TIME = 1.0
# STEER_INCREMENT = 20 #[deg]

STEER_UPDATE_TIME = 0.05 #[s]
STEER_INCREMENT = 5 #[deg]

SPEED_UPDATE_TIME = 0.2 # [s]
SPEED_INCREMENT = 0.1 # [m/s]

class Automobile_Data():
    def __init__(self,
                simulator=False,
                trig_cam=False,
                trig_gps=False,
                trig_bno=False,
                trig_enc=False,
                trig_control=True,
                trig_estimation=False,
                trig_sonar=False,
                speed_buff_len = 40):
        """Manage flow of data with the car

        :param trig_control: trigger on commands, defaults to True
        :type trig_control: bool, optional

        :param trig_cam: trigger on camera, defaults to False
        :type trig_cam: bool, optional

        :param trig_gps: trigger on GPS, defaults to False
        :type trig_gps: bool, optional

        :param trig_bno: trigger on IMU, defaults to False
        :type trig_bno: bool, optional

        :param trig_stateEst: trigger on state estimation, defaults to False
        :type trig_stateEst: bool, optional
        
        :param trig_enc: trigger on encoder, defaults to False
        :type trig_enc: bool, optional

        :param speed_buff_len: length of the speed buffer
        :type speed_buff_len: int, optional
        """        

        self.simulator_flag = simulator # flag to know if we are in simulator or not'
        if not self.simulator_flag:
            from utils.srv import subscribing
        
        # State of the car
        self.x_true = 0.0           # [m]       true:x coordinate (used in simulation and SPARCS)
        self.y_true = 0.0           # [m]       true:y coordinate (used in simulation and SPARCS)
        self.time_stamp = 0.0       # [s]       true:time stamp (used in simulation)
        self.x = 0.0                # [m]       GPS:x global coordinate
        self.y = 0.0                # [m]       GPS:y global coordinate
        self.roll = 0.0             # [rad]     IMU:roll angle of the car
        self.roll_deg = 0.0         # [deg]     IMU:roll angle of the car
        self.pitch = 0.0            # [rad]     IMU:pitch angle of the car
        self.pitch_deg = 0.0        # [deg]     IMU:pitch angle of the car
        self.yaw = 0.0              # [rad]     IMU:yaw angle of the car
        self.yaw_deg = 0.0          # [deg]     IMU:yaw angle of the car
        self.accel_x = 0.0          # [m/ss]    IMU:accelx angle of the car
        self.accel_y = 0.0          # [m/ss]    IMU:accely angle of the car
        self.accel_z = 0.0          # [m/ss]    IMU:accelz angle of the car
        self.gyrox = 0.0            # [rad/s]   IMU:gyrox angular vel of the car
        self.gyroy = 0.0            # [rad/s]   IMU:gyroy angular vel of the car
        self.gyroz = 0.0            # [rad/s]   IMU:gyroz angular vel of the car
        self.cv_image = np.zeros((FRAME_WIDTH, FRAME_HEIGHT))
                                    # [ndarray] CAM:image of the camera
        self.speed_meas = 0.0       # [m/s]     ENC:speed measure of the car from encoder
        self.speed_meas_buffer = collections.deque(maxlen=speed_buff_len)   # FIFO queue for median filter on the encoder speed measurement
        self.speed_meas_median = 0.0

        self.xEst = 0.0             # [m]       EST:x EKF estimated global coordinate
        self.yEst = 0.0             # [m]       EST:y EKF estimated global coordinate
        self.yawEst = 0.0           # [rad]     EST:yaw EKF estimated
        
        self.xLoc = 0.0             # [m]       local:x local coordinate
        self.yLoc = 0.0             # [m]       local:y local coordinate
        self.yawLoc = 0.0           # [rad]     local:yaw local
        self.yawLoc_o = 0.0         # [rad]     local:yaw offset
        self.distLoc = 0.0          # [m]       local:absolute distance, length of local trajectory

        #velocity control feedback
        self.dist_from_last_call = 0.0    # [m]       local:distance traveed from last drive speed command
        self.time_last_call = 0.0         # [s]       local:time from last drive speed command

        self.obstacle_ahead = 0.0   # [m]       SONAR:distance of an obstacle ahead
        self.obstacle_ahead_buffer = collections.deque(maxlen=20)   # FIFO queue for median filter on the encoder speed measurement
        self.obstacle_ahead_median = 3.0

        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.cam_x = CAM_X
        self.cam_y = CAM_Y
        self.cam_z = CAM_Z
        self.cam_roll = CAM_ROLL
        self.cam_pitch = CAM_PITCH
        self.cam_yaw = CAM_YAW
        self.cam_fov = CAM_FOV
        self.cam_K = CAM_K

        #control
        self.target_steer = 0.0
        self.prev_steer = 0.0
        self.target_speed = 0.0
        self.prev_speed = 0.0        

        #self.prev_est_time = 0.0    # [s]   used to compute estimator sampling time
        #self.fist_stateEst_callback = True                  # bool to check first call of estimator
        
        #self.rear_xEst = 0.0                        # estimated rear axis x [m]
        #self.rear_yEst = 0.0                        # estimated rear axis y [m]

        # Input to the car (control action)
        self.speed = 0.0            # [m/s]     MOTOR:speed
        self.steer = 0.0            # [rad]     SERVO:steering angle
        self.steer_ack = False      # [bool]    acknowledge trigger on STEER command
        self.speed_ack = False      # [bool]    acknowledge trigger on SPEED command
        self.break_ack = False      # [bool]    acknowledge trigger on BREAK command

        # I/O interface
        # caller to the service for creating subscribers to get feedback from Nucleo
        caller = rospy.ServiceProxy('/command_feedback_en', subscribing) if not self.simulator_flag else None
        # publisher: send command to the car
        rospy.init_node('automobile_data', anonymous=False)
        rospy.sleep(1)  # wait for publisher to register to roscore
        # subscribers
        if trig_control:
            # control stuff
            self.pub = rospy.Publisher('/automobile/command', String, queue_size=1)
            # caller(subscribing=True,code='SPED',topic=FDBK_SPEED_TOPIC) if not self.simulator_flag else None
            # caller(subscribing=True,code='STER',topic=FDBK_STEER_TOPIC) if not self.simulator_flag else None
            # caller(subscribing=True,code='BRAK',topic=FDBK_BREAK_TOPIC) if not self.simulator_flag else None
            # self.activate_PID(PID_ENABLE)
            # feedback from commands
            # self.sub_speed_ack = rospy.Subscriber(FDBK_SPEED_TOPIC, String, self.speed_callback) if not self.simulator_flag else None
            # self.sub_steer_ack = rospy.Subscriber(FDBK_STEER_TOPIC, String, self.steer_callback) if not self.simulator_flag else None
            # self.sub_break_ack = rospy.Subscriber(FDBK_BREAK_TOPIC, String, self.break_callback) if not self.simulator_flag else None

            #set control callbacks
            # rospy.Timer(rospy.Duration(STEER_UPDATE_TIME), self.update_angle_callback)
            # rospy.Timer(rospy.Duration(SPEED_UPDATE_TIME), self.update_speed_callback)

        if trig_cam:
            # camera stuff
            self.sub_cam = rospy.Subscriber("/automobile/image_raw", Image, self.camera_callback)
        if trig_gps:
            # position stuff
            self.sub_pos = rospy.Subscriber("/automobile/localisation", localisation, self.position_callback)
        if trig_bno:
            # imu stuff
            imu_topic = "/automobile/imu" if not self.simulator_flag else "/automobile/IMU"
            self.sub_imu = rospy.Subscriber(imu_topic, IMU, self.imu_callback)
        if trig_estimation:
            # estimation stuff
            self.ekf = EKFCar(x0 = EST_X0, WB = WB)
            rospy.Timer(rospy.Duration(EST_EKF_STATE_TS), self.update_estimated_state)
        if trig_enc:
            # encoder stuff
            self.activate_encoder(True)
            caller(subscribing=True,code='ENPB',topic=FDBK_ENC_TOPIC)
            self.sub_enc = rospy.Subscriber(FDBK_ENC_TOPIC, String, self.encoder_callback)
            # activate relative positioning
            self.reset_rel_pose()
            rospy.Timer(rospy.Duration(EST_REL_POS_TS), self.update_rel_pose)
        if trig_sonar:
            self.sub_son = rospy.Subscriber("/automobile/sonar", Float32, self.sonar_callback)

   
    # -------------------- COMMAND CALLBACKS --------------------
    def steer_callback(self, data):  
        """Callback when a steer acknowledge is received

        :param data: acknowledge message
        :type data: dict
        """              
        self.steer_ack = True
    
    def break_callback(self, data):
        """Callback when a break acknowledge is received

        :param data: acknowledge message
        :type data: dict
        """        
        self.break_ack = True

    def speed_callback(self, data):
        """Callback when a speed acknowledge is received

        :param data: acknowledge message
        :type data: dict
        """        
        self.speed_ack = True

    # -------------------- DATA CALLBACKS --------------------
    def camera_callback(self, data):
        """Receive and store camera frame

        :param data: sensor_msg array containing the image from the camera
        :type data: object
        """        
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def sonar_callback(self, data):
        self.obstacle_ahead = data.data if data.data>0 else self.obstacle_ahead
        self.obstacle_ahead_buffer.append(self.obstacle_ahead)
        self.obstacle_ahead_median = np.median(np.array(list(self.obstacle_ahead_buffer)))

    def position_callback(self, data):
        """Receive and store global coordinates from GPS

        :param data: contains x and y coordinates in the world rf
        :type data: object
        """        
        self.x = data.posA
        self.y = data.posB 

    def imu_callback(self, data):
        """Receive and store rotation from IMU 

        :param data: a geometry_msgs Twist message
        :type data: object
        """        
        #imu gives true coordinates in the simulator
        if self.simulator_flag:
            self.roll = float(data.roll)
            self.roll_deg = np.rad2deg(self.roll)
            self.pitch = float(data.pitch)
            self.pitch_deg = np.rad2deg(self.pitch)
            self.yaw = float(data.yaw)
            self.yaw_deg = np.rad2deg(self.yaw)
            #true position, for training purposes
            self.x_true = float(data.posx)
            self.y_true = float(data.posy)
            self.time_stamp = float(data.timestamp)
        else:
            self.roll_deg = float(data.roll)
            self.pitch_deg = float(data.pitch)
            self.yaw_deg = float(data.yaw)

            self.roll = np.deg2rad(data.roll)
            self.pitch = np.deg2rad(data.pitch)
            self.yaw = np.deg2rad(data.yaw) if data.yaw>=0 else self.yaw

            self.accel_x = data.accelx
            self.accel_y = data.accely
            self.accel_z = data.accelz

            self.gyrox = data.gyrox        
            self.gyroy = data.gyroy   
            self.gyroz = data.gyroz

    def encoder_callback(self, data):
        """Callback when an encoder speed message is received

        :param data: speed from encoder message
        :type data: dict
        """        
        data = data.data
        # print(f'data from encoder: {data}')
        motor_speed = data[6:-2]# extract string value
        motor_speed = float(motor_speed)        # [rps] speed at motor side
        #wheel_speed = motor_speed/GBOX_RATIO    # [rps] speed at wheel side

        #self.speed_meas = 2*pi*wheel_speed * WHEEL_LEN  # [m/s] car speed
        self.speed_meas = motor_speed /154.0# see Embedded Platform code

        self.speed_meas_buffer.append(self.speed_meas)
        self.speed_meas_median = np.median(np.array(list(self.speed_meas_buffer)))

    '''
    def update_angle_callback(self, event):
        """Callback to update the angle of the car

        :param data: angle from IMU message
        :type data: object
        """ 

        if np.isclose(self.target_steer, self.prev_steer):
            return
        else:
            if self.target_steer > self.prev_steer:
                set_steer = min(self.target_steer, self.prev_steer+STEER_INCREMENT)
            else:
                set_steer = max(self.target_steer, self.prev_steer-STEER_INCREMENT)

            if self.simulator_flag:
                # steer command
                data = {}
                data['action']        =  '2'
                data['steerAngle']    =  float(set_steer)*STEER_COMPENSATION
                reference = json.dumps(data)

                self.steer_ack = False
                cnt = 0
                while not self.steer_ack and cnt < ROS_REPEAT:
                    self.pub.publish(reference)
                    cnt += 1
                    #print(f'waiting for ack:{cnt*ROS_PAUSE} seconds')
                    sleep(ROS_PAUSE)

                    if self.simulator_flag: break
                if cnt >= ROS_REPEAT:
                    raise Exception('steer command not acknowledged')
            else: # If we are in the real car, control the servo directly from the pi
                val = deg2pwm(set_steer)
                # assert -0.58 < val < 0.8, f'ANGLE IS TOO BIG: {val}'
                self.servo.value = val

            self.prev_steer = set_steer
        

    def update_speed_callback(self, event):
        """Callback to update the speed of the car

        :param data: speed from encoder message
        :type data: object
        """ 
        if np.isclose(self.target_speed, self.prev_speed):
            if not self.simulator_flag:
                if np.abs(self.speed_meas-self.target_speed) > 0.05:
                    print(f'Car didnt move, resending the speed command meas = {self.speed_meas} target = {self.target_speed}')
                    self.publish_speed(self.target_speed)
                else:
                    # print(f'speed is close, meas = {self.speed_meas} target = {self.target_speed}')
                    # print(np.abs(self.speed_meas-self.target_speed))
                    pass
                return
        else:
            #reset dist from last call
            self.dist_from_last_call = 0.0
            self.time_last_call = time()

            if self.target_speed > self.prev_speed:
                set_speed = min(self.target_speed, self.prev_speed+SPEED_INCREMENT)
            else:
                set_speed = max(self.target_speed, self.prev_speed-SPEED_INCREMENT)

            self.speed = set_speed
            self.publish_speed(set_speed)

            self.prev_speed = set_speed
    

    def publish_speed(self, set_speed):
        # Create and publish a SPEED command until speed_ack is received
        data = {}
        data['action']        =  '1'
        data['speed']         =  SPEED_COMPENSATION * float(set_speed)
        reference = json.dumps(data)

        self.speed_ack = False
        cnt = 0
        while not self.speed_ack and cnt < ROS_REPEAT:
            self.pub.publish(reference)
            cnt += 1 
            sleep(ROS_PAUSE)
            if self.simulator_flag: break
        if cnt >= ROS_REPEAT:
            #raise Exception('speed command not acknowledged')
            print('speed command not acknowledged')
    '''

    
    # -------------------- ACTIVATE ACTIONS --------------------
    def activate_encoder(self, encoder_enable=True):
        """Activate/Deactivate encoder feedback

        :param encoder_enable: self explanatory, defaults to True
        :type encoder_enable: bool, optional
        """        
        data = {}
        data['action']      =  'ENPB'
        data['activate']    =  encoder_enable
        reference = json.dumps(data)
        self.pub.publish(reference)
        sleep(ROS_PAUSE)

    def activate_PID(self, pid_enable=True):
        """Activate/Deactivate speed PID

        :param pid_enable: self explanatory, defaults to True
        :type pid_enable: bool, optional
        """            
        data = {}
        data['action']      =  'PIDA'
        data['activate']    =  pid_enable
        reference = json.dumps(data)
        self.pub.publish(reference)
        sleep(ROS_PAUSE)

    # -------------------- COMMAND ACTIONS --------------------
    def drive_speed(self, speed=0.0):
        """Publish the SPEED command to the command topic

        :param speed: [m/s] desired speed, defaults to 0.0
        :type speed: float, optional
        :raises Exception: if the command is not acknowledged
        """                
        speed = Automobile_Data.normalizeSpeed(speed)   # normalize speed
        #self.target_speed = speed
        data = {}
        data['action']        =  'SPED'
        data['speed']         =  SPEED_COMPENSATION * float(speed)
        reference = json.dumps(data)
        self.pub.publish(reference)
        sleep(ROS_PAUSE)


    def drive_angle(self, angle=0.0, direct=False):
        """Publish the STEER command to the command topic

        :param angle: [rad] desired angle, defaults to 0.0
        :type angle: float, optional
        :raises Exception: if the command is not acknowledged
        """        
        angle = Automobile_Data.normalizeSteer(angle)   # normalize steer
        if self.simulator_flag:
            data = {}
            data['action']        =  '2'
            data['steerAngle']    =  float(angle)*STEER_COMPENSATION
            reference = json.dumps(data)
        
            self.steer_ack = False
            cnt = 0
            while not self.steer_ack and cnt < ROS_REPEAT:
                self.pub.publish(reference)
                cnt += 1
                #print(f'waiting for ack:{cnt*ROS_PAUSE} seconds')
                if self.simulator_flag: break
                sleep(ROS_PAUSE)

            if cnt >= ROS_REPEAT:
                # raise Exception('steer command not acknowledged')
                print('steer command not acknowledged')
                pass
        else: #direct in the real car
            data = {}
            data['action']        =  'STER'
            data['steerAngle']    =  float(angle)*STEER_COMPENSATION
            reference = json.dumps(data)
            self.pub.publish(reference)
            sleep(ROS_PAUSE)


    def drive(self, speed=0.0, angle=0.0):
        """Publish the SPEED and STEER command to the command topic

        :param speed: [m/s] desired speed, defaults to 0.0
        :type speed: float, optional
        :param angle: [rad] desired angle, defaults to 0.0
        :type angle: float, optional
        :raises Exception: if the command is not acknowledged
        """        
        self.drive_speed(speed)
        self.drive_angle(angle)
    
    def stop(self, angle=0.0):
        """Publish the BREAK command to the command topic

        :param angle: [deg] stop angle, defaults to 0.0
        :type angle: float, optional
        :raises Exception: if the command is not acknowledged
        """
        # self.drive_speed(0.0)
        # self.drive_angle(angle)

        angle = Automobile_Data.normalizeSteer(angle)   # normalize steer
        self.speed = 0.0
        self.steer = angle

        data = {}
        if not self.simulator_flag:
            data['action']        =  'BRAK'
            data['brake (steerAngle)']    =  float(angle)*STEER_COMPENSATION
        else:
            data['action']        =  '3'
            data['steerAngle']    =  float(angle)*STEER_COMPENSATION
        
        reference = json.dumps(data)
        self.pub.publish(reference)
        sleep(ROS_PAUSE)

        # self.break_ack = False
        # cnt = 0
        # while not self.break_ack and cnt < ROS_REPEAT:
        #     self.pub.publish(reference)
        #     cnt += 1
        #     sleep(ROS_PAUSE)
        #     if self.simulator_flag: break
        # if cnt >= ROS_REPEAT:
        #     raise Exception('break command not acknowledged')

    # -------------------- ESTIMATION --------------------
    def reset_rel_pose(self):
        """Set origin of the local frame to the actual pose
        """        
        self.xLoc = 0.0
        self.yLoc = 0.0
        self.yawLoc_o = self.yaw
        self.yawLoc = 0.0
        self.distLoc = 0.0
    
    def reset_rel_dist(self):
        self.distLoc = 0.0
    
    def update_rel_pose(self,event):
        """Update relative pose of the car

        :param event: event parameter passed by the rospy.Timer
        :type event: dict
        """ 
        if event.last_real is not None:
            dt = event.current_real - event.last_real
            dt = dt.to_sec() #floating point
        else:
            dt = EST_REL_POS_TS      
        
        self.yawLoc = diff_angle(self.yaw, self.yawLoc_o)
        #rint(f'yaw: {self.yaw:.2f}, Loc: {self.yawLoc:.2f}, origin: {self.yawLoc_o:.2f}')
        self.xLoc += self.speed_meas * sin(self.yawLoc) * dt
        self.yLoc += self.speed_meas * cos(self.yawLoc) * dt

        dist_increment = abs(self.speed_meas) * dt

        self.distLoc += dist_increment
        self.dist_from_last_call += dist_increment

    def update_estimated_state(self,event):
        """Update estimated state according to EKF
        """        
        # Sampling time of the estimator
        if event.last_real is not None:
            dt = event.current_real - event.last_real
            dt = dt.to_sec() #floating point
        else:
            dt = EST_EKF_STATE_TS
        
        # callback_time = rospy.get_time()

        # if self.fist_stateEst_callback:
        #     self.fist_stateEst_callback = False
        #     self.prev_est_time = callback_time
        #     return
        # else:
        #     DT = rospy.get_time() - self.prev_est_time
        #     self.prev_est_time = callback_time
            
        if dt > 0:            
            # Input: [SPEED, STEER]
            u0 = self.speed
            u1 = self.steer
            u = np.array([u0, u1]).reshape(-1,1)

            # Output: [GPS:x, GPS:y, IMU:yaw]
            zx = self.x
            zy = 15.0 - self.y# change of coordinates to bottom-left reference frame
            zth = self.yaw
            z = np.array([zx, zy, zth]).reshape(-1,1)

            # Predict step
            self.ekf.predict(u=u, dt=dt)
            # Update STEP
            self.ekf.update(z=z, HJacobian=Automobile_Data.HJacobian, Hx=Automobile_Data.Hx, residual=Automobile_Data.residual)

            # Store estimated state
            xxEst = np.copy(self.ekf.x)
            # x: CoM
            self.xEst = xxEst[0,0]
            # y: CoM
            self.yEst = 15-xxEst[1,0]# change of coordinates to top-left reference frame
            # Yaw
            self.yawEst = xxEst[2,0]

            #self.rear_xEst = self.xEst - ((WB / 2) * np.cos(self.yawEst))
            #self.rear_yEst = self.yEst + ((WB / 2) * np.sin(self.yawEst))
        else:
            print(f'Estimator sampling time not positive:  {dt} s')  


    # STATIC METHODS
    @staticmethod
    def normalizeSpeed(val):
        """Clamp speed value

        :param val: speed to clamp
        :type val: double
        :return: clamped speed value
        :rtype: double
        """        
        if val < MIN_SPEED:
            val = MIN_SPEED
        elif val > MAX_SPEED:
            val = MAX_SPEED
        return val

    @staticmethod
    def normalizeSteer(val):
        """Clamp steer value

        :param val: steer to clamp
        :type val: double
        :return: clamped steer value
        :rtype: double
        """        
        if val < -MAX_STEER/STEER_COMPENSATION:
            val = -MAX_STEER/STEER_COMPENSATION
        elif val > MAX_STEER/STEER_COMPENSATION:
            val = MAX_STEER/STEER_COMPENSATION
        return val

    @staticmethod
    def residual(a,b):
        """Compute residual between two output vectors and normalize yaw angle. Used inside the EKF

        :param a: first output vector
        :type a: ndarray
        :param b: second output vector
        :type b: ndarray
        :return: normalized residual of two measurements a-b
        :rtype: ndarray
        """          
        y = a - b
        if y[2] > np.pi:
            y[2] -= 2*np.pi
        if y[2] < -np.pi:
            y[2] += 2*np.pi
        return y

    @staticmethod
    def HJacobian(x):
        """Compute Jacobian of the output map J_h(x)

        :param x: state
        :type x: ndarray
        :return: output map jacobian J_h(x)
        :rtype: ndarray
        """        
        H = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
        return H

    @staticmethod
    def Hx(x):
        """Compute output map h(x)

        :param x: state
        :type x: ndarray
        :return: respective output z=h(x)
        :rtype: ndarray
        """        
        H = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
        z = H @ x
        return z
    

def deg2pwm(f_angle):
    # pwm_nucleo = 0.0009505 * f_angle + 0.07525# deg-> pwm in 0-1
    # pwm_nucleo = pwm_nucleo * 10.0#57.29577951/10.0
    f_angle += 3.0 #offset
    # pwm_raspi = 0.033925 * f_angle  ##02313
    pwm_raspi = 0.050634/2.0 * f_angle# AFTER CALIBRATION WITH CIRCLE PATH
    # pwm_raspi = (pwm_nucleo * 2.0) - 1.0
    return pwm_raspi










