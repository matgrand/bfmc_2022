#!/usr/bin/python3
from time import sleep, time
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
import json
# messages for communication
from std_msgs.msg import String
from utils.msg import IMU
from utils.msg import localisation
from utils.srv import subscribing
from sensor_msgs.msg import Image
import os
# state estimation
#from estimation import EKFCar
from math import cos,sin,pi, radians
from helper_functions import *

# Vehicle driving parameters
MIN_SPEED = -0.3                    # minimum speed [m/s]
MAX_SPEED = 0.5                    # maximum speed [m/s]
MAX_ACCEL = 0.5                     # maximum accel [m/ss]
MAX_STEER = np.deg2rad(22.8)        # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(40.0)       # maximum steering speed [rad/s]

# Vehicle parameters
LENGTH = 0.45  			            # car body length [m]
WIDTH = 0.18   			            # car body width [m]
BACKTOWHEEL = 0.10  		        # distance of the wheel and the car body [m]
WHEEL_LEN = 0.03  			        # wheel raduis [m]
WHEEL_WIDTH = 0.03  		        # wheel thickness [m]
TREAD = 0.09  			            # horizantal distance between the imaginary line through the center of the car to where the wheel is on the body (usually width/2) [m]
WB = 0.26  			                # distance between the two wheels centers on one side (front and back wheel) [m]
GBOX_RATIO = 28

#Camera position and orientation wrt the car frame
CAM_X, CAM_Y, CAM_Z, CAM_ROLL, CAM_PITCH, CAM_YAW = 0, 0, 0.2, 0, 0.2617, 0 #[m, rad]
CAM_FOV = 1.085594795 #[rad]
CAM_F = 1.0 # focal length
CAM_Sx, CAM_Sy, CAM_Ox, CAM_Oy = 10,10,10,10#100, 100, 240, 320 #scaling factors [m]->[pixels], and offsets [pixel]

class Automobile_Data():
    def __init__(self, trig_control=True, trig_cam=False, trig_gps=False, trig_bno=False, trig_stateEst=False):
        """

        :param command_topic_name: drive and stop commands are sent on this topic
        :param response_topic_name: "ack" message is received on this topic
        """
        # State of the car
        # sampling time
        self.Ts = 0.1# [s] sampling time

        self.ros_pause = 0.001
        self.ros_repeat = 50

        # position
        self.x = 0.0
        self.y = 0.0

        #true position
        self.x_true = 0.0
        self.y_true = 0.0

        #simulator time_stamp, prob not nocessary
        self.time_stamp = 0.0

        # EKF estimation
        self.trig_stateEst = trig_stateEst
        if self.trig_stateEst:
            x0=np.array([1.5, 15-6.0,0.9*pi]).reshape(-1,1) # initial state
            self.ekf = EKFCar(x0 = x0, WB = WB)             # EKF declaration
            
            self.fist_stateEst_callback = True                  # bool to check first call of estimator
            self.last_stateEst_time = 0.0               # used to compute sampling time of estimator
            self.xEst = 0.0                             # estimated x [m]
            self.yEst = 0.0                             # estimated y [m]
            self.yawEst = 0.0                           # estimated yaw [rad]
            self.rear_xEst = 0.0                        # estimated rear axis x [m]
            self.rear_yEst = 0.0                        # estimated rear axis y [m]

        # orientation
        self.roll = 0.0
        self.roll_deg = 0.0
        self.pitch = 0.0
        self.pitch_deg = 0.0
        self.yaw = 0.0
        self.yaw_deg = 0.0
        self.delta_yaw = 0.0
        self.yaw_old = 0.0
        self.yaw_accumulator = 0.0

        # car parameters
        self.rear_x = 0.0
        self.rear_y = 0.0

        # PID activation
        self.PID_active = True

        # camera
        self.bridge = CvBridge()
        self.cv_image = np.zeros((640, 480))
        self.cam_x = CAM_X
        self.cam_y = CAM_Y
        self.cam_z = CAM_Z
        self.cam_roll = CAM_ROLL
        self.cam_pitch = CAM_PITCH
        self.cam_yaw = CAM_YAW
        self.cam_fov = CAM_FOV
        self.cam_K = np.array([[CAM_F*CAM_Sx, 0, CAM_Ox], [ 0, CAM_F*CAM_Sy, CAM_Oy], [ 0, 0, 1.0]])
        assert self.cam_K.shape == (3,3) # make sure the matrix is 3x3

        # encoder
        self.speed_meas = 0.0       # [m/s] car speed measure from encoder
        self.last_enc_time = 0.0    # [s] used to compute encoder sampling time
        self.xEst_rel = 0.0
        self.yEst_rel = 0.0

        # Publisher node : Send command to the car
        rospy.init_node('automobile_data', anonymous=False)  
        if trig_control:
            # control stuff
            self.pub = rospy.Publisher('/automobile/command', String, queue_size=1)
            self.activate_PID(True)
        if trig_cam:
            # camera stuff
            self.sub_cam = rospy.Subscriber("/automobile/image_raw", Image, self.camera_callback)
        if trig_gps:
            # position stuff
            self.sub_pos = rospy.Subscriber("/automobile/localisation", localisation, self.position_callback)
        if trig_bno:
            # imu stuff
            self.sub_imu = rospy.Subscriber('/automobile/imu', IMU, self.imu_callback)

        # Create publishers for command acknowledge
        self.steer_feedback_topic = '/steer_feedback'
        self.stop_feedback_topic = '/break_feedback'
        self.speed_feedback_topic = '/speed_feedback'
        self.encoder_feedback_topic = '/encoder_feedback'
        self.steer_ack = False
        self.speed_ack = False
        self.stop_ack = False

        # os.system('rosservice call /command_feedback_en \"subscribing: true\ncode:\'2:ac\'\ntopic:\''+self.steer_feedback_topic+'\'\"')
        caller = rospy.ServiceProxy('/command_feedback_en', subscribing)
        caller(subscribing=True,code='2:ac',topic=self.steer_feedback_topic)
        caller(subscribing=True,code='3:ac',topic=self.stop_feedback_topic)
        caller(subscribing=True,code='1:ac',topic=self.speed_feedback_topic)
        caller(subscribing=True,code='ENPB',topic=self.encoder_feedback_topic)

        # Create subscribers for drive and stop commands
        self.steer_command_sub = rospy.Subscriber(self.steer_feedback_topic, String, self.steer_command_callback)
        self.stop_command_sub = rospy.Subscriber(self.stop_feedback_topic, String, self.stop_command_callback)
        self.speed_command_sub = rospy.Subscriber(self.speed_feedback_topic, String, self.speed_command_callback)
        self.encoder_command_sub = rospy.Subscriber(self.encoder_feedback_topic, String, self.encoder_command_callback)

        # control action
        self.speed = 0.0
        self.steer = 0.0
        #rospy.spin() # spin() simply keeps python from exiting until this node is stopped

        # Wait for publisher to register to roscore -
        # This is a temporary fix. Problem with timing
        rospy.sleep(1)

    #command feeback callbacks
    def steer_command_callback(self, data): 
        self.steer_ack = True
        # print("steer ack received", data)
    
    def stop_command_callback(self, data):
        self.stop_ack = True
        # print("stop ack received", data)

    def speed_command_callback(self, data):
        self.speed_ack = True
        # print("speed ack received", data)

    def encoder_command_callback(self, data):
        data = data.data
        motor_speed = data[6:-2]
        motor_speed = float(motor_speed)# motor angular speed [rps]

        self.speed_meas = motor_speed * 2*pi/GBOX_RATIO * WHEEL_LEN # car speed [m/s]

        curr_enc_time = time()
        dt = curr_enc_time - self.last_enc_time
        self.delta_yaw = diff_angle(self.yaw,self.yaw_old)
        self.yaw_accumulator += self.delta_yaw

        self.xEst_rel -= self.speed_meas * sin(self.delta_yaw) * dt
        self.yEst_rel += self.speed_meas * cos(self.delta_yaw) * dt

        self.last_enc_time = curr_enc_time
        self.yaw_old = self.yaw

    def reset_relative_position(self):
        self.xEst_rel = 0.0
        self.yEst_rel = 0.0
        

    def activate_PID(self, activate):
        data = {}
        data['action']        =  '4'
        data['activate']    =  activate
        reference = json.dumps(data)
        for i in range(self.ros_repeat):
            self.pub.publish(reference)
            sleep(self.ros_pause)


    def drive_speed(self, speed=0.0):
        """Transmite the command to the remotecontrol receiver. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe. 
        """
        data = {}
        
        # normalize speed and angle, basically clip them
        speed = Automobile_Data.normalizeSpeed(speed)

        # self.update_control_action(speed, self.yaw)
        self.speed = speed

        # # speed command
        data = {}
        data['action']        =  '1'
        data['speed']         =  float(speed)

        # publish
        reference = json.dumps(data)
        self.speed_ack = False
        cnt = 0
        while not self.speed_ack and cnt < 200:
            self.pub.publish(reference)
            cnt+=1 
            sleep(self.ros_pause)
            if cnt > 150:
                raise Exception('speed command not acknowledged')
        # print(f"speed command sent in {cnt} iterations")



    def drive_angle(self, angle=0.0):
        """Transmite the command to the remotecontrol receiver. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe. 
        """
        data = {}
        
        # normalize speed and angle, basically clip them
        angle = Automobile_Data.normalizeSteer(angle)

        # self.update_control_action(speed, self.yaw)

        # steer command
        data['action']        =  '2'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.steer_ack = False
        cnt = 0
        print(f'angle command sent {angle}')
        while not self.steer_ack and cnt < 200:
            self.pub.publish(reference)
            sleep(self.ros_pause)
            cnt += 1
            if cnt > 150:
                raise Exception('steer command not acknowledged')
    
    def stop(self, angle=0.0):
        """
        This simulates the braking behaviour from the Nucleo
        :param angle:
        :return:
        """
        data = {}

        # normalize angle
        angle = Automobile_Data.normalizeSteer(angle)

        # update control action
        self.speed = 0.0

        # brake command
        # data['action']        =  '3'
        # data['steerAngle']    =  angle
        data['action']        =  '3'
        data['brake (steerAngle)']    =  0.0
        # publish
        reference = json.dumps(data)
        self.stop_ack = False
        cnt = 0
        while not self.stop_ack and cnt < 200:
            self.pub.publish(reference)
            sleep(self.ros_pause)
            cnt += 1
            if cnt > 150:
                raise Exception('stop command not acknowledged')


    def activate_encoder(self, angle=0.0):
        """

        """
        data = {}
        data['action']        =  '5'
        data['activate']    =  True
        # publish
        reference = json.dumps(data)
        for i in range(self.ros_repeat):
            self.pub.publish(reference)
            sleep(self.ros_pause)

    def camera_callback(self, data):
        """
        :param data: sensor_msg array containing the image in the Gazsbo format
        :return: nothing but sets [cv_image] to the usefull image that can be use in opencv (numpy array)
        """
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # cv2.imshow("Frame preview", self.cv_image)
        # key = cv2.waitKey(1)

    def position_callback(self, data):
        self.x = data.posA
        self.y = data.posB 

        if self.trig_stateEst:
            self.update_estimated_state()

    def imu_callback(self, data):
        """
        :param msg: a geometry_msgs Twist message
        :return: nothing, but sets the oz, wx, wy, wz class members
        """
        self.roll = np.deg2rad(data.roll)
        self.pitch = np.deg2rad(data.pitch)
        self.yaw = np.deg2rad(data.yaw)
        self.roll_deg = data.roll
        self.pitch_deg = data.pitch
        self.yaw_deg = data.yaw
        
    
    def update_estimated_state(self):
        ''' Updates estimated state according to EKF '''
        # Sampling time of the estimator
        callback_time = rospy.get_time()

        if self.fist_stateEst_callback:
            self.fist_stateEst_callback = False
            self.last_stateEst_time = callback_time
            return
        else:
            DT = rospy.get_time() - self.last_stateEst_time
            self.last_stateEst_time = callback_time
            
        if DT>0:            
            # INPUT: [SPEED, STEER]
            u0 = self.speed
            u1 = self.steer
            u = np.array([u0, u1]).reshape(-1,1)

            # OUTPUT: [GPS:x, GPS:y, IMU:yaw]
            zx = self.x
            zy = 15 - self.y# change of coordinates to bottom-left reference frame
            zth = self.yaw
            z = np.array([zx, zy, zth]).reshape(-1,1)
            #print('z:  ',z)

            # *******************************************
            # PREDICT STEP
            self.ekf.predict(u=u, dt=DT)
            # UPDATE STEP
            self.ekf.update(z=z, HJacobian=self.HJacobian, Hx=self.Hx, residual=self.residual)

            # *******************************************
            # ESTIMATED STATE
            xxEst = np.copy(self.ekf.x)
            # x: CoM
            self.xEst = xxEst[0,0]
            # y: CoM
            self.yEst = 15-xxEst[1,0]# change of coordinates to top-left reference frame
            # Yaw
            self.yawEst = xxEst[2,0]

            self.rear_xEst = self.xEst - ((WB / 2) * np.cos(self.yawEst))
            self.rear_yEst = self.yEst + ((WB / 2) * np.sin(self.yawEst)) 
    
    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return np.hypot(dx, dy)

    @staticmethod
    def normalizeSpeed(val):
        """
        :return: normalized value
        """
        if val < MIN_SPEED:
            val = MIN_SPEED
        elif val > MAX_SPEED:
            val = MAX_SPEED
        return val

    @staticmethod
    def normalizeSteer(val):
        """
        :return: normalized value
        """
        if val < -np.rad2deg(MAX_STEER):
            val = -np.rad2deg(MAX_STEER)
        elif val > np.rad2deg(MAX_STEER):
            val = np.rad2deg(MAX_STEER)
        return val

    @staticmethod
    def residual(a,b):
        """compute residual between two output measurements and normalize yaw angle
        Args:
            a (ndarray): first measurement
            b (ndarray): second measurement

        Returns:
            ndarray: normalized residual of two measurements, i.e. a-b
        """    
        y = a - b
        if y[2] > np.pi:
            y[2] -= 2*np.pi
        if y[2] < -np.pi:
            y[2] += 2*np.pi
        return y

    @staticmethod
    def HJacobian(x):
        '''
        Compute Jacobian of the output map h(x)
        '''
        H = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
        return H

    @staticmethod
    def Hx(x):
        '''
        Compute the output, given a state
        '''
        H = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,1]])
        z = H @ x
        return z











