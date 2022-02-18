#!/usr/bin/python3

# license removed for brevity
from tkinter import Frame
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
import json
# messages for communication
from std_msgs.msg import String
from utils.msg import IMU
from utils.msg import localisation
from sensor_msgs.msg import Image

# Estimation parameter of EKF
Q = np.diag([0.01, 0.01])**2  # Observation x,y position covariance
#R = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2  # predict state covariance
R = np.diag([0.1, 0.1, 0, 1.0])**2  # predict state covariance


# Vehicle driving parameters
MIN_SPEED = -0.3                    # minimum speed [m/s]
MAX_SPEED = 0.5                    # maximum speed [m/s]
MAX_ACCEL = 0.5                     # maximum accel [m/ss]
MAX_STEER = np.deg2rad(30.0)        # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(40.0)       # maximum steering speed [rad/s]

# Vehicle parameters
LENGTH = 0.45  			            # car body length [m]
WIDTH = 0.18   			            # car body width [m]
BACKTOWHEEL = 0.10  		        # distance of the wheel and the car body [m]
WHEEL_LEN = 0.03  			        # wheel raduis [m]
WHEEL_WIDTH = 0.03  		        # wheel thickness [m]
TREAD = 0.09  			            # horizantal distance between the imaginary line through the center of the car to where the wheel is on the body (usually width/2) [m]
#WB = 0.26  			                # distance between the two wheels centers on one side (front and back wheel) [m]
WB = 0.304

#Camera position and orientation wrt the car frame
CAM_X, CAM_Y, CAM_Z, CAM_ROLL, CAM_PITCH, CAM_YAW = 0, 0, 0.2, 0, 0.2617, 0 #[m, rad]
CAM_FOV = 1.085594795 #[rad]
CAM_F = 1.0 # focal length
CAM_Sx, CAM_Sy, CAM_Ox, CAM_Oy = 10,10,10,10#100, 100, 240, 320 #scaling factors [m]->[pixels], and offsets [pixel]

#  Simulation parameter
#Qsim = np.diag([0.5, 0.5])**2
#Rsim = np.diag([1.0, np.deg2rad(30.0)])**2

#response_topic_name="/automobile/feedback"
class Automobile_Data():
    def __init__(self, trig_control=True, trig_cam=False, trig_gps=False, trig_bno=False):
        """

        :param command_topic_name: drive and stop commands are sent on this topic
        :param response_topic_name: "ack" message is received on this topic
        """
        # State of the car
        # sampling time
        self.Ts = 0.1# [s] sampling time

        # position
        self.x = 0.0
        self.y = 0.0

        #true position
        self.x_true = 0.0
        self.y_true = 0.0

        #simulator time_stamp, prob not nocessary
        self.time_stamp = 0.0

        # EKF estimation of position
        self.xEst = np.zeros((4, 1))
        self.xTrue = np.zeros((4, 1))
        self.PEst = np.eye(4)

        # orientation
        self.roll = 0.0
        self.roll_deg = 0.0
        self.pitch = 0.0
        self.pitch_deg = 0.0
        self.yaw = 0.0
        self.yaw_deg = 0.0

        # car parameters
        self.rear_x = 0.0
        self.rear_y = 0.0

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

        # Publisher node : Send command to the car
        rospy.init_node('automobile_data', anonymous=False)  
        if trig_control:
            # control stuff
            self.pub = rospy.Publisher('/automobile/command', String, queue_size=1)
        if trig_cam:
            # camera stuff
            self.sub_cam = rospy.Subscriber("/automobile/image_raw", Image, self.camera_callback)
        if trig_gps:
            # position stuff
            self.sub_pos = rospy.Subscriber("/automobile/localisation", localisation, self.position_callback)
        if trig_bno:
            # imu stuff
            self.sub_imu = rospy.Subscriber('/automobile/IMU', IMU, self.imu_callback)
            
        # control action
        self.speed = 0.0
        self.steer = 0.0
        self.yawRate = 0.0
        #rospy.spin() # spin() simply keeps python from exiting until this node is stopped

        # Wait for publisher to register to roscore -
        # This is a temporary fix. Problem with timing
        rospy.sleep(1)



    def drive(self, speed=0.0, angle=0.0):
        """Transmite the command to the remotecontrol receiver. 
        
        Parameters
        ----------
        inP : Pipe
            Input pipe. 
        """
        data = {}
        
        # normalize speed and angle, basically clip them
        angle = Automobile_Data.normalizeSteer(angle)
        speed = Automobile_Data.normalizeSpeed(speed)

        # self.update_control_action(speed, self.yaw)
        self.speed = speed
        self.yawRate = 0.0

        '''
        # steer command
        data['action']        =  '2'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)

        # speed command
        data = {}
        data['action']        =  '1'
        data['speed']         =  float(speed/100.0)
        '''
        # move command
        data['action']        =  '4'
        data['speed']         =  speed
        data['steerAngle']    =  angle

        # publish
        reference = json.dumps(data)
        self.pub.publish(reference)
    
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
        self.yawRate = 0.0

        # brake command
        data['action']        =  '3'
        data['steerAngle']    =  angle

        # publish
        reference = json.dumps(data)
        self.pub.publish(reference)

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

        self.update_estimated_state()
        self.update_car_parameters()

    def imu_callback(self, data):
        """
        :param msg: a geometry_msgs Twist message
        :return: nothing, but sets the oz, wx, wy, wz class members
        """
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

        self.update_estimated_state()
        self.update_car_parameters()
    
    def update_estimated_state(self):
        ''' Updates estimated state according to EKF '''
        # noised input
        # ud1 = speed
        # ud2 = yawrate
        ud1 = self.speed#u[0, 0] + np.random.randn() * Rsim[0, 0]
        ud2 = self.yawRate#u[1, 0] + np.random.randn() * Rsim[1, 1]
        ud = np.array([[ud1, ud2]]).T

        # output measurements
        zx = self.x#xTrue[0, 0] + np.random.randn() * Qsim[0, 0]
        zy = self.y#xTrue[1, 0] + np.random.randn() * Qsim[1, 1]
        z = np.array([[zx, zy]])

        self.xEst, self.PEst = ekf_estimation(self.xEst, self.PEst, z, ud, self.Ts)

    def update_car_parameters(self):
        self.rear_x = self.x - ((WB / 2) * np.cos(self.yaw))
        self.rear_y = self.y + ((WB / 2) * np.sin(self.yaw))
    
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

def jacobF(x, u, DT):
        """
        Jacobian of Motion Model
        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
        """
        yaw = x[2, 0]
        v = u[0, 0]
        jF = np.array([
            [1.0, 0.0, -DT * v * np.sin(yaw), DT * np.cos(yaw)],
            [0.0, 1.0, DT * v * np.cos(yaw), DT * np.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return jF


def jacobH(x):
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH

def observation_model(x):
        ''' Returns 2-dim nparray with [x,y] '''
        #  Observation Model
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = H.dot(x) 
        return z

def motion_model(x, u, DT):

    F = np.array([[1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 0]])

    B = np.array([[DT * np.cos(x[2, 0]), 0],
                [DT * np.sin(x[2, 0]), 0],
                [0.0, DT],
                [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)

    return x

def ekf_estimation(xEst, PEst, z, u, DT):

    #  Predict
    xPred = motion_model(xEst, u, DT)
    jF = jacobF(xPred, u, DT)
    PPred = jF.dot(PEst).dot(jF.T) + R

    #  Update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z.T - zPred
    S = jH.dot(PPred).dot(jH.T) + Q
    K = PPred.dot(jH.T).dot(np.linalg.inv(S))
    xEst = xPred + K.dot(y)
    PEst = (np.eye(len(xEst)) - K.dot(jH)).dot(PPred)

    return xEst, PEst









