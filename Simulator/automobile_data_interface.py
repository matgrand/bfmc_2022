#!/usr/bin/python3
# Functional libraries
import rospy
import numpy as np
from helper_functions import *
import os
# from estimation import EKFCar

START_X = 0.2
START_Y = 14.8

# Vehicle driving parameters
MIN_SPEED = -0.3                    # [m/s]     minimum speed
MAX_SPEED = 2.5                     # [m/s]     maximum speed
MAX_ACCEL = 5.5                     # [m/ss]    maximum accel
MAX_STEER = 27.0                    # [deg]     maximum steering angle

# Vehicle parameters
LENGTH = 0.45  			            # [m]       car body length
WIDTH = 0.18   			            # [m]       car body width
BACKTOWHEEL = 0.10  		        # [m]       distance of the wheel and the car body
WHEEL_LEN = 0.03  			        # [m]       wheel raduis
WHEEL_WIDTH = 0.03  		        # [m]       wheel thickness
WB = 0.26  			                # [m]       wheelbase

# Camera parameters
FRAME_WIDTH = 320#640           # [pix]     frame width
FRAME_HEIGHT = 240#480          # [pix]     frame height
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

class Automobile_Data():
    def __init__(self,
                trig_control=True,
                trig_bno=False,
                trig_enc=False,
                trig_sonar=False,
                trig_cam=False,
                trig_gps=False,
                trig_estimation=False, 
                ) -> None:
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

        # CAR POSITION
        self.x_true = START_X                   # [m]       true:x coordinate (used in simulation and SPARCS)
        self.y_true = START_Y                   # [m]       true:y coordinate (used in simulation and SPARCS)
        self.x = 0.0                            # [m]       GPS:x global coordinate
        self.y = 0.0                            # [m]       GPS:y global coordinate
        # IMU           
        self.roll = 0.0                         # [rad]     IMU:roll angle of the car
        self.roll_deg = 0.0                     # [deg]     IMU:roll angle of the car
        self.pitch = 0.0                        # [rad]     IMU:pitch angle of the car
        self.pitch_deg = 0.0                    # [deg]     IMU:pitch angle of the car
        self.yaw = 0.0                          # [rad]     IMU:yaw angle of the car
        self.yaw_deg = 0.0                      # [deg]     IMU:yaw angle of the car
        self.accel_x = 0.0                      # [m/ss]    IMU:accelx angle of the car
        self.accel_y = 0.0                      # [m/ss]    IMU:accely angle of the car
        self.accel_z = 0.0                      # [m/ss]    IMU:accelz angle of the car
        self.gyrox = 0.0                        # [rad/s]   IMU:gyrox angular vel of the car
        self.gyroy = 0.0                        # [rad/s]   IMU:gyroy angular vel of the car
        self.gyroz = 0.0                        # [rad/s]   IMU:gyroz angular vel of the car
        # ENCODER                           
        self.encoder_velocity = 0.0             # [m/s]     ENC:speed measure of the car from encoder
        self.filtered_encoder_velocity = 0.0    # [m/s]     ENC:filtered speed measure of the car from encoder
        self.encoder_distance = 0.0             # [m]       total absolute distance measured by the encoder, it never get reset
        # CAR POSE ESTIMATION
        self.x_est = 0.0                        # [m]       EST:x EKF estimated global coordinate
        self.y_est = 0.0                        # [m]       EST:y EKF estimated global coordinate
        self.yaw_est = 0.0                      # [rad]     EST:yaw EKF estimated
        # LOCAL POSITION
        self.x_loc = 0.0                        # [m]       local:x local coordinate
        self.y_loc = 0.0                        # [m]       local:y local coordinate
        self.yaw_loc = 0.0                      # [rad]     local:yaw local
        self.yaw_loc_o = 0.0                    # [rad]     local:yaw origin wrt to global yaw from IMU
        self.dist_loc = 0.0                     # [m]       local:absolute distance, length of local trajectory
        self.dist_loc_o = 0.0                   # [m]       local:absolute distance origin, wrt global encoder distance
        # SONAR
        self.sonar_distance = 0.0               # [m]       SONAR: unfiltered distance from the sonar
        self.filtered_sonar_distance = 3.0      # [m]       SONAR: filtered distance from the sonar
        # CAMERA
        self.frame = np.zeros((FRAME_WIDTH, FRAME_HEIGHT)) # [ndarray] CAM:image of the camera
        # CONTROL ACTION
        self.speed = 0.0            # [m/s]     MOTOR:speed
        self.steer = 0.0            # [rad]     SERVO:steering angle
        # CONSTANT PARAMETERS
        self.MIN_SPEED = MIN_SPEED              # [m/s]     minimum speed of the car
        self.MAX_SPEED = MAX_SPEED              # [m/s]     maximum speed of the car
        self.MAX_STEER = MAX_STEER              # [deg]     maximum steering angle of the car
        # Vehicle parameters
        self.LENGTH = LENGTH			        # [m]       car body length
        self.WIDTH = WIDTH  			        # [m]       car body width
        self.BACKTOWHEEL = BACKTOWHEEL		    # [m]       distance of the wheel and the car body
        self.WHEEL_LEN = WHEEL_LEN  			# [m]       wheel raduis
        self.WHEEL_WIDTH = WHEEL_WIDTH  		# [m]       wheel thickness
        self.WB = WB  			                # [m]       wheelbase
        # Camera parameters
        self.FRAME_WIDTH = FRAME_WIDTH          # [pix]     frame width
        self.FRAME_HEIGHT = FRAME_HEIGHT        # [pix]     frame height
        self.CAM_X = CAM_X
        self.CAM_Y = CAM_Y
        self.CAM_Z = CAM_Z
        self.CAM_ROLL = CAM_ROLL
        self.CAM_PITCH = CAM_PITCH
        self.CAM_YAW = CAM_YAW
        self.CAM_FOV = CAM_FOV
        self.CAM_K = CAM_K


        # I/O interface
        rospy.init_node('automobile_data', anonymous=False)

        # SUBSCRIBERS AND PUBLISHERS
        # to be implemented in the specific class
        # they need to refer to the specific callbacks
        pass


    # DATA CALLBACKS
    def camera_callback(self, data) -> None:
        """Receive and store camera frame
        :acts on: self.frame
        """        
        pass
    def sonar_callback(self, data) -> None:
        """Receive and store distance of an obstacle ahead in 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        pass

    def position_callback(self, data) -> None:
        """Receive and store global coordinates from GPS
        :acts on: self.x, self.y
        """        
        pass

    def imu_callback(self, data) -> None:
        """Receive and store rotation from IMU 
        :acts on: self.roll, self.pitch, self.yaw, self.roll_deg, self.pitch_deg, self.yaw_deg
        :acts on: self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz
        """        
        pass

    def encoder_distance_callback(self, data) -> None:
        """Callback when an encoder distance message is received
        :acts on: self.encoder_distance
        :needs to: call update_rel_position
        """        
        pass

    def update_rel_position(self) -> None:
        """Update relative pose of the car
        right-hand frame of reference with x aligned with the direction of motion
        """  
        self.yaw_loc = diff_angle(self.yaw, self.yaw_loc_o)
        # dyaw = diff_angle(self.yaw_loc, self.prev_yaw_loc)
        # dyaw = diff_angle(self.yaw_loc - self.prev_yaw_loc)
        # self.yaw_loc = self.yaw - self.yaw_loc_o
        prev_dist = self.dist_loc
        self.dist_loc = np.abs(self.encoder_distance - self.dist_loc_o)
        L = np.abs(self.dist_loc - prev_dist)
        x_increment = L * np.cos(self.yaw_loc)
        y_increment = - L * np.sin(self.yaw_loc)
        self.x_loc += x_increment
        self.y_loc += y_increment

    def encoder_velocity_callback(self, data) -> None:
        """Callback when an encoder velocity message is received
        :acts on: self.encoder_velocity
        """        
        pass

    # COMMAND ACTIONS
    def drive_speed(self, speed=0.0) -> None:
        """Set the speed of the car
        :acts on: self.speed 
        :param speed: speed of the car [m/s], defaults to 0.0
        """       
        raise NotImplementedError()         

    def drive_angle(self, angle=0.0) -> None:
        """Set the steering angle of the car
        :acts on: self.steer
        :param angle: [deg] desired angle, defaults to 0.0
        """    
        raise NotImplementedError()    

    def drive(self, speed=0.0, angle=0.0) -> None:
        """Command a speed and steer angle to the car
        :param speed: [m/s] desired speed, defaults to 0.0
        :param angle: [deg] desired angle, defaults to 0.0
        """        
        self.drive_speed(speed)
        self.drive_angle(angle)
    
    def stop(self, angle=0.0) -> None:
        """Hard/Emergency stop the car
        :acts on: self.speed, self.steer
        :param angle: [deg] stop angle, defaults to 0.0
        """
        raise NotImplementedError()

    def reset_rel_pose(self) -> None:
        """Set origin of the local frame to the actual pose
        """        
        self.x_loc = 0.0
        self.y_loc = 0.0
        self.yaw_loc_o = self.yaw
        self.prev_yaw = self.yaw
        self.yaw_loc = 0.0
        self.prev_yaw_loc = 0.0
        self.dist_loc = 0.0
        self.dist_loc_o = self.encoder_distance

    # STATIC METHODS
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

    def normalizeSteer(val):
        """Clamp steer value

        :param val: steer to clamp
        :type val: double
        :return: clamped steer value
        :rtype: double
        """        
        if val < -MAX_STEER:
            val = -MAX_STEER
        elif val > MAX_STEER:
            val = MAX_STEER
        return val


    










