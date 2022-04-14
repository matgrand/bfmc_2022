#!/usr/bin/python3
from cmath import cos
from automobile_data_interface import Automobile_Data
from std_msgs.msg import String
from utils.msg import IMU,localisation
from sensor_msgs.msg import Image, Range
import rospy, json, collections
from cv_bridge import CvBridge
import numpy as np
from time import time,sleep
from helper_functions import *

ENCODER_TIMER = 0.01 #frequency of encoder reading

class AutomobileDataSimulator(Automobile_Data):
    def __init__(self,
                trig_control=True,
                trig_bno=False,
                trig_enc=False,
                trig_sonar=False,
                trig_cam=False,
                trig_gps=False,
                trig_estimation=False, 
                ) -> None:
        #initialize the parent class
        super().__init__()

        # ADDITIONAL VARIABLES
        self.sonar_distance_buffer = collections.deque(maxlen=20)
        self.timestamp = 0.0
        self.prev_x_true = self.x_true
        self.prev_y_true = self.y_true
        self.prev_timestamp = 0.0
        self.velocity_buffer = collections.deque(maxlen=20)

        # PUBLISHERS AND SUBSCRIBERS
        if trig_control:
            self.pub = rospy.Publisher('/automobile/command', String, queue_size=1)
        if trig_bno:
            self.sub_imu = rospy.Subscriber('/automobile/IMU', IMU, self.imu_callback)
        if trig_enc:
            self.reset_rel_pose()
            rospy.Timer(rospy.Duration(ENCODER_TIMER), self.encoder_distance_callback) #the callback will also do velocity
        if trig_sonar:
            self.sub_son = rospy.Subscriber('/automobile/sonar1', Range, self.sonar_callback)
        if trig_cam:
            self.bridge = CvBridge()
            self.sub_cam = rospy.Subscriber("/automobile/image_raw", Image, self.camera_callback)
        if trig_gps:
            self.sub_pos = rospy.Subscriber("/automobile/localisation", localisation, self.position_callback)
        if trig_estimation:
            raise NotImplementedError("estimation not implemented yet")

    def camera_callback(self, data) -> None:
        """Receive and store camera frame
        :acts on: self.frame
        """        
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def sonar_callback(self, data) -> None:
        """Receive and store distance of an obstacle ahead in 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        self.sonar_distance = data.range 
        self.sonar_distance_buffer.append(self.sonar_distance)
        self.filtered_sonar_distance = np.median(self.sonar_distance_buffer)

    def position_callback(self, data) -> None:
        """Receive and store global coordinates from GPS
        :acts on: self.x, self.y
        """        
        pL = np.array([data.posA, data.posB])
        pR = mL2mR(pL)
        self.x = pR[0]# + self.WB/2*np.cos(self.yaw)
        self.y = pR[1]# - self.WB/2*np.sin(self.yaw)
        self.update_estimated_state()

    def imu_callback(self, data) -> None:
        """Receive and store rotation from IMU 
        :acts on: self.roll, self.pitch, self.yaw, self.roll_deg, self.pitch_deg, self.yaw_deg
        :acts on: self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz
        """        
        self.roll = float(data.roll)
        self.roll_deg = np.rad2deg(self.roll)    
        self.pitch = float(data.pitch)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw = float(data.yaw)
        self.yaw_deg = np.rad2deg(self.yaw)
        #true position, not in real car

        x_true = float(data.posx)
        y_true = -float(data.posy) + 15.0
        # center x_true on the rear axis
        self.x_true = x_true - self.WB/2*np.cos(self.yaw)  
        self.y_true = y_true - self.WB/2*np.sin(self.yaw)

        self.timestamp = float(data.timestamp)
        #NOTE: in the simulator we don't have neither acceleromter or gyroscope (yet)

    def encoder_distance_callback(self, data) -> None:
        """Callback when an encoder distance message is received
        :acts on: self.encoder_distance
        :needs to: call update_rel_position
        """   
        curr_x = self.x_true
        curr_y = self.y_true
        prev_x = self.prev_x_true
        prev_y = self.prev_y_true
        curr_time = self.timestamp
        # curr_time = time()
        prev_time = self.prev_timestamp
        delta = np.hypot(curr_x - prev_x, curr_y - prev_y)
        #get the direction of the movement: + or -
        motion_yaw = + np.arctan2(curr_y - prev_y, curr_x - prev_x)
        abs_yaw_diff = np.abs(diff_angle(motion_yaw, self.yaw))
        sign = 1.0 if abs_yaw_diff < np.pi/2 else -1.0
        dt = curr_time - prev_time
        if dt > 0.0:
            velocity = (delta * sign) / dt
            self.encoder_velocity_callback(data=velocity) 
            self.encoder_distance += sign*delta
            self.prev_x_true = curr_x
            self.prev_y_true = curr_y
            self.prev_timestamp = curr_time
            self.update_rel_position()

    def encoder_velocity_callback(self, data) -> None:
        """Callback when an encoder velocity message is received
        :acts on: self.encoder_velocity
        """    
        self.encoder_velocity = data
        self.velocity_buffer.append(self.encoder_velocity)
        self.filtered_encoder_velocity = np.median(self.velocity_buffer)

    # COMMAND ACTIONS
    def drive_speed(self, speed=0.0) -> None:
        """Set the speed of the car
        :acts on: self.speed 
        :param speed: speed of the car [m/s], defaults to 0.0
        """                
        speed = Automobile_Data.normalizeSpeed(speed)   # normalize speed
        self.speed = speed
        data = {}
        data['action']        =  '1'
        data['speed']         =  float(speed)
        reference = json.dumps(data)
        self.pub.publish(reference)

    def drive_angle(self, angle=0.0) -> None:
        """Set the steering angle of the car
        :acts on: self.steer
        :param angle: [deg] desired angle, defaults to 0.0
        """        
        angle = Automobile_Data.normalizeSteer(angle)   # normalize steer
        data = {}
        data['action']        =  '2'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)

    def stop(self, angle=0.0) -> None:
        """Hard/Emergency stop the car
        :acts on: self.speed, self.steer
        :param angle: [deg] stop angle, defaults to 0.0
        """
        self.speed = 0.0
        data = {}
        data['action']        =  '3'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)