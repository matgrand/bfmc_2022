#!/usr/bin/python3
from control.automobile_data_interface import Automobile_Data
from control.helper_functions import *
from std_msgs.msg import Float32, Bool
from utils.msg import IMU,localisation
import rospy, collections
import numpy as np
from time import time,sleep

SONAR_THRESHOLD = 5

SONAR_DEQUE_LENGTH = 5

class AutomobileDataPi(Automobile_Data):
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
        self.lateral_sonar_distance_buffer = collections.deque(maxlen=SONAR_DEQUE_LENGTH)


        self.center_sonar_distance = 3.0
        self.center_sonar_distance_buffer = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.filtered_center_sonar_distance = 3.0

        self.left_sonar_distance = 3.0
        self.left_sonar_distance_buffer = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.filtered_left_sonar_distance = 3.0

        self.right_sonar_distance = 3.0
        self.right_sonar_distance_buffer = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.filtered_right_sonar_distance = 3.0

        self.encoder_velocity_buffer = collections.deque(maxlen=SONAR_DEQUE_LENGTH)
        self.reachedPosition = False

        self.is_position_reliable = True
        self.estimation_last_encoder_distance = 0.0
        self.estimation_last_yaw_est = 0.0

        self.x_buffer = collections.deque(maxlen=3)
        self.y_buffer = collections.deque(maxlen=3)

        # PUBLISHERS AND SUBSCRIBERS
        if trig_control:
            self.pub_speed = rospy.Publisher('/automobile/command/speed', Float32, queue_size=1)
            self.pub_steer = rospy.Publisher('/automobile/command/steer', Float32, queue_size=1)
            self.pub_stop = rospy.Publisher('/automobile/command/stop', Float32, queue_size=1)
            self.pub_position = rospy.Publisher('/automobile/command/position', Float32, queue_size=1)
            self.sub_position = rospy.Subscriber("/automobile/feedback/position", Bool, self.feedback_position_callback)
        if trig_bno:
            self.sub_imu = rospy.Subscriber('/automobile/imu', IMU, self.imu_callback)
        if trig_enc:
            self.sub_encSpeed = rospy.Subscriber('/automobile/encoder/speed', Float32, self.encoder_velocity_callback)
            self.sub_encDist = rospy.Subscriber('/automobile/encoder/distance', Float32, self.encoder_distance_callback)
            self.reset_rel_pose()
        if trig_sonar:
            self.sub_son_ahead_center = rospy.Subscriber('/automobile/sonar/ahead/center', Float32, self.center_sonar_callback)
            self.sub_son_ahead_left = rospy.Subscriber('/automobile/sonar/ahead/left', Float32, self.left_sonar_callback)
            self.sub_son_ahead_right = rospy.Subscriber('/automobile/sonar/ahead/right', Float32, self.right_sonar_callback)

            self.sub_lateral = rospy.Subscriber('/automobile/sonar/lateral', Float32, self.lateral_sonar_callback)
        if trig_cam:
            raise NotImplementedError("cam not implemented yet")
        if trig_gps:
            self.sub_pos = rospy.Subscriber("/automobile/localisation", localisation, self.position_callback)
        if trig_estimation:
            self.trig_estimation = trig_estimation
            print("ESTIMATION ENABLED")

    # def camera_callback(self, data) -> None:
    #     """Receive and store camera frame
    #     :acts on: self.frame
    #     """        
    #     self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def center_sonar_callback(self, data) -> None:
        """Receive and store distance of an obstacle ahead in 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        self.center_sonar_distance = data.data if data.data>0 else self.center_sonar_distance
        self.center_sonar_distance_buffer.append(self.center_sonar_distance)
        self.filtered_center_sonar_distance = np.median(self.center_sonar_distance_buffer)
        self.update_sonar_distance()

    def left_sonar_callback(self, data) -> None:
        """Receive and store distance of an obstacle ahead in 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        self.left_sonar_distance = data.data if data.data>0 else self.left_sonar_distance
        self.left_sonar_distance_buffer.append(self.left_sonar_distance)
        self.filtered_left_sonar_distance = np.median(self.left_sonar_distance_buffer)
        self.update_sonar_distance()

    def right_sonar_callback(self, data) -> None:
        """Receive and store distance of an obstacle ahead in 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        self.right_sonar_distance = data.data if data.data>0 else self.right_sonar_distance
        self.right_sonar_distance_buffer.append(self.right_sonar_distance)
        self.filtered_right_sonar_distance = np.median(self.right_sonar_distance_buffer)
        self.update_sonar_distance()

    def update_sonar_distance(self):
        if self.steer < -SONAR_THRESHOLD:
            self.sonar_distance = min(self.left_sonar_distance, self.center_sonar_distance)
            self.filtered_sonar_distance = min(self.filtered_left_sonar_distance, self.filtered_center_sonar_distance)
        elif -SONAR_THRESHOLD <= self.steer < SONAR_THRESHOLD:
            self.sonar_distance = self.center_sonar_distance
            self.filtered_sonar_distance = self.filtered_center_sonar_distance
        elif self.steer >= SONAR_THRESHOLD:
            self.sonar_distance = min(self.right_sonar_distance, self.center_sonar_distance)
            self.filtered_sonar_distance = min(self.filtered_right_sonar_distance, self.filtered_center_sonar_distance)
        # UNCOMMENT AND REMOVE LEFT AND RIGHT SONARS
        # self.sonar_distance = self.center_sonar_distance
        # self.filtered_sonar_distance = self.filtered_center_sonar_distance

    def lateral_sonar_callback(self, data) -> None:
        """Receive and store distance of an obstacle ahead in 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        self.lateral_sonar_distance = data.data if data.data>0 else self.lateral_sonar_distance
        self.lateral_sonar_distance_buffer.append(self.lateral_sonar_distance)
        self.filtered_lateral_sonar_distance = np.median(self.lateral_sonar_distance_buffer)

    def position_callback(self, data) -> None:
        """Receive and store global coordinates from GPS
        :acts on: self.x, self.y
        """        
        pL = np.array([data.posA, data.posB])
        pR = mL2mR(pL)
        # print(f'PL = {pL}')
        # print(f'PR = {pR}')
        # self.x = pR[0]
        # self.y = pR[1]
        tmp_x = pR[0] - self.WB/2*np.cos(self.yaw)
        tmp_y = pR[1] - self.WB/2*np.sin(self.yaw)
        self.x_buffer.append(tmp_x)
        self.y_buffer.append(tmp_y)
        self.x = np.mean(self.x_buffer)
        self.y = np.mean(self.y_buffer)
        self.x_est = self.x
        self.y_est = self.y
        # self.update_estimated_state()


    def imu_callback(self, data) -> None:
        """Receive and store rotation from IMU 
        :acts on: self.roll, self.pitch, self.yaw, self.roll_deg, self.pitch_deg, self.yaw_deg
        :acts on: self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz
        """        
        self.roll = np.deg2rad(data.roll)
        self.pitch = np.deg2rad(data.pitch)
        self.yaw = diff_angle(np.deg2rad(data.yaw) + self.yaw_offset, 0.0)

        # self.yaw = diff_angle(self.yaw, self.yaw_offset)

        self.roll_deg = np.rad2deg(self.roll)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_deg = np.rad2deg(self.yaw)


        self.accel_x = data.accelx
        self.accel_y = data.accely
        self.accel_z = data.accelz

        self.gyrox = data.gyrox        
        self.gyroy = data.gyroy   
        self.gyroz = data.gyroz


    def encoder_distance_callback(self, data) -> None:
        """Callback when an encoder distance message is received
        :acts on: self.encoder_distance
        :needs to: call update_rel_position
        """   
        self.encoder_distance= data.data
        self.update_rel_position()

    def encoder_velocity_callback(self, data) -> None:
        """Callback when an encoder velocity message is received
        :acts on: self.encoder_velocity
        """    
        self.encoder_velocity = data.data
        self.encoder_velocity_buffer.append(self.encoder_velocity)
        self.filtered_encoder_velocity = np.median(self.encoder_velocity_buffer)


    # COMMAND ACTIONS
    def drive_speed(self, speed=0.0) -> None:
        """Set the speed of the car
        :acts on: self.speed 
        :param speed: speed of the car [m/s], defaults to 0.0
        """                
        speed = Automobile_Data.normalizeSpeed(speed)   # normalize speed
        self.pub_speed.publish(speed)

    def drive_angle(self, angle=0.0) -> None:
        """Set the steering angle of the car
        :acts on: self.steer
        :param angle: [deg] desired angle, defaults to 0.0
        """        
        angle = Automobile_Data.normalizeSteer(angle)   # normalize steer
        self.steer = angle
        self.pub_steer.publish(angle)

    def stop(self, angle=0.0) -> None:
        """Hard/Emergency stop the car
        :acts on: self.speed, self.steer
        :param angle: [deg] stop angle, defaults to 0.0
        """
        angle = Automobile_Data.normalizeSteer(angle)   # normalize steer
        self.steer = angle
        self.pub_stop.publish(angle)
    
    # ADDITIONAL METHODS
    def drive_distance(self, dist=0.0):
        self.reachedPosition = False
        self.pub_position.publish(dist)

    def feedback_position_callback(self, data):
        self.reachedPosition = data.data
