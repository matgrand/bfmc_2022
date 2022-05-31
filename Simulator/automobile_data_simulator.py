#!/usr/bin/python3
from automobile_data_interface import Automobile_Data
from std_msgs.msg import String
from utils.msg import IMU,localisation
from sensor_msgs.msg import Image, Range
import rospy, json, collections
from cv_bridge import CvBridge
import numpy as np
from time import time
from helper_functions import *

REALISTIC = False

ENCODER_TIMER = 0.01 #frequency of encoder reading
STEER_UPDATE_FREQ = 50.0 if REALISTIC else 150.0 #50#[Hz]
SERVO_DEAD_TIME_DELAY = 0.15 if REALISTIC else 0.0 #0.15 #[s]
MAX_SERVO_ANGULAR_VELOCITY = 2.8 if REALISTIC else 15.0 #3. #[rad/s]
DELTA_ANGLE = np.rad2deg(MAX_SERVO_ANGULAR_VELOCITY) / STEER_UPDATE_FREQ
MAX_STEER_COMMAND_FREQ = 50.0 # [Hz], If steer commands are sent at an higher freq they will be discarded
MAX_STEER_SAMPLES = max(int((2*SERVO_DEAD_TIME_DELAY) * MAX_STEER_COMMAND_FREQ), 10)

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
        self.lateral_sonar_distance_buffer = collections.deque(maxlen=20)
        self.timestamp = 0.0
        self.prev_x_true = self.x_true
        self.prev_y_true = self.y_true
        self.prev_timestamp = 0.0
        self.velocity_buffer = collections.deque(maxlen=20)
        self.target_steer = 0.0
        self.curr_steer = 0.0
        self.steer_deque = collections.deque(maxlen=MAX_STEER_SAMPLES)
        self.time_last_steer_command = time()
        self.target_dist = 0.0
        self.arrived_at_dist = True
        self.yaw_true = 0.0

        self.x_buffer = collections.deque(maxlen=5)
        self.y_buffer = collections.deque(maxlen=5)

        # PUBLISHERS AND SUBSCRIBERS
        if trig_control:
            self.pub = rospy.Publisher('/automobile/command', String, queue_size=1)
            self.steer_updater = rospy.Timer(rospy.Duration(1/STEER_UPDATE_FREQ), self.steer_update_callback)
            self.drive_dist_updater = rospy.Timer(rospy.Duration(ENCODER_TIMER), self.drive_distance_callback)
        if trig_bno:
            self.sub_imu = rospy.Subscriber('/automobile/IMU', IMU, self.imu_callback)
        if trig_enc:
            self.reset_rel_pose()
            rospy.Timer(rospy.Duration(ENCODER_TIMER), self.encoder_distance_callback) #the callback will also do velocity
        if trig_sonar:
            self.sub_son = rospy.Subscriber('/automobile/sonar1', Range, self.sonar_callback)
            self.sub_lateral_son = rospy.Subscriber('/automobile/sonar2', Range, self.lateral_sonar_callback)
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
        """Receive and store distance of an obstacle ahead 
        :acts on: self.sonar_distance, self.filtered_sonar_distance
        """        
        self.sonar_distance = data.range 
        self.sonar_distance_buffer.append(self.sonar_distance)
        self.filtered_sonar_distance = np.median(self.sonar_distance_buffer)

    def lateral_sonar_callback(self, data) -> None:
        """Receive and store distance of a lateral obstacle
        :acts on: self.lateral_sonar_distance, self.filtered_lateral_sonar_distance
        """        
        self.lateral_sonar_distance = data.range 
        self.lateral_sonar_distance_buffer.append(self.lateral_sonar_distance)
        self.filtered_lateral_sonar_distance = np.median(self.lateral_sonar_distance_buffer)

    def position_callback(self, data) -> None:
        """Receive and store global coordinates from GPS
        :acts on: self.x, self.y
        """        
        # pL = np.array([data.posA, data.posB])
        # pR = mL2mR(pL)
        # self.x = pR[0]
        # self.y = pR[1]
        # self.x = pR[0] - self.WB/2*np.cos(self.yaw)
        # self.y = pR[1] - self.WB/2*np.sin(self.yaw)
        # self.update_estimated_state()
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

    def imu_callback(self, data) -> None:
        """Receive and store rotation from IMU 
        :acts on: self.roll, self.pitch, self.yaw, self.roll_deg, self.pitch_deg, self.yaw_deg
        :acts on: self.accel_x, self.accel_y, self.accel_z, self.gyrox, self.gyroy, self.gyroz
        """        
        self.roll = float(data.roll)
        self.roll_deg = np.rad2deg(self.roll)    
        self.pitch = float(data.pitch)
        self.pitch_deg = np.rad2deg(self.pitch)
        self.yaw_true = float(data.yaw)
        self.yaw = float(data.yaw) + self.yaw_offset
        self.yaw_deg = np.rad2deg(self.yaw)
        #true position, not in real car
        true_posL = np.array([data.posx, data.posy])
        true_posR = mL2mR(true_posL)
        # center x_true on the rear axis
        self.x_true = true_posR[0] - self.WB/2*np.cos(self.yaw_true)  
        self.y_true = true_posR[1] - self.WB/2*np.sin(self.yaw_true)

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
        abs_yaw_diff = np.abs(diff_angle(motion_yaw, self.yaw_true))
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

    def steer_update_callback(self, data) -> None:
        # check self.steer_deque is not empty
        if len(self.steer_deque) > 0:
            curr_time = time()
            angle, t = self.steer_deque.popleft()
            if curr_time - t < SERVO_DEAD_TIME_DELAY: #we need to w8, time has not passed yet
                self.steer_deque.appendleft((angle,t))
            else: # enough time is passed, we can update the angle 
                self.target_steer = angle
        diff = self.target_steer - self.curr_steer
        if diff > 0.0:
            incr = min(diff, DELTA_ANGLE)
        elif diff < 0.0:
            incr = max(diff, -DELTA_ANGLE)
        else : return
        self.curr_steer += incr
        self.pub_steer(self.curr_steer)
        # print(F'CURRENT STEER: {self.curr_steer:.1f}')

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
        self.arrived_at_dist = True #ovrride the drive distance        
        self.pub_speed(speed)

    def drive_angle(self, angle=0.0, direct=False) -> None:
        """Set the steering angle of the car
        :acts on: self.steer
        :param angle: [deg] desired angle, defaults to 0.0
        """        
        angle = Automobile_Data.normalizeSteer(angle)   # normalize steer
        curr_time = time()
        if curr_time - self.time_last_steer_command > 1/MAX_STEER_COMMAND_FREQ: #cannot receive commands too fast
            self.time_last_steer_command = curr_time
            self.steer_deque.append((angle, curr_time))
        else:
            print('Missed steer command...')
        # self.target_steer = angle #self.steer is updated in the callback

    def drive_distance(self, dist=0.0):
        """Drive the car a given distance forward or backward
        from the point it has been called and stop there, 
        it uses control in position and not in velocity, 
        used for precise movements
        :param dist: distance to drive, defaults to 0.0
        """
        self.target_dist = self.encoder_distance + dist
        self.arrived_at_dist = False

    def drive_distance_callback(self, data) -> None:
        Kp = 0.5
        max_speed = 0.2
        if not self.arrived_at_dist:
            dist_error = self.target_dist - self.encoder_distance
            # print(F'DISTANCE ERROR: {dist_error:.2f}, arrrived at dist: {self.arrived_at_dist}')
            self.pub_speed(min(Kp * dist_error, max_speed))
            if np.abs(dist_error) < 0.01:
                self.arrived_at_dist = True
                self.drive_speed(0.0)
        
    def stop(self, angle=0.0) -> None:
        """Hard/Emergency stop the car
        :acts on: self.speed, self.steer
        :param angle: [deg] stop angle, defaults to 0.0
        """
        self.steer_deque.append((angle, time()))
        self.speed = 0.0
        data = {}
        data['action']        =  '3'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)

    def pub_steer(self, angle):
        data = {}
        data['action']        =  '2'
        data['steerAngle']    =  float(angle)
        reference = json.dumps(data)
        self.pub.publish(reference)

    def pub_speed(self, speed):
        speed = Automobile_Data.normalizeSpeed(speed)   # normalize speed
        self.speed = speed
        data = {}
        data['action']        =  '1'
        data['speed']         =  float(speed)
        reference = json.dumps(data)
        self.pub.publish(reference)