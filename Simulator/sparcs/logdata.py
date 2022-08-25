#!/usr/bin/python3
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion

import rospy, collections
import numpy as np
import time

VICON_TOPIC = '/vicon/bfmc_car/bfmc_car'
YAW_OFFSET = np.deg2rad(-90)

SONAR_DEQUE_LENGTH = 5

class DataLogger():
    def __init__(self) -> None:
        self.x_buffer = collections.deque(maxlen=3)
        self.y_buffer = collections.deque(maxlen=3)

        self.x          = 0
        self.y          = 0
        self.z          = 0
        self.euler      = 0
        self.roll       = 0
        self.pitch      = 0
        self.yaw        = 0

        self.isCalibrated = False
        self.x0, self.y0, self.z0, self.yaw0 = None, None, None, None

        self.sub_vicon = rospy.Subscriber(VICON_TOPIC, TransformStamped, self.vicon_callback)
        rospy.init_node('logdata_node', anonymous=False)


    def vicon_callback(self, data) -> None:  
        self.euler = euler_from_quaternion([data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w]) 
        self.roll   = self.euler[0]
        self.pitch  = self.euler[1]
        self.yaw    = self.euler[2]

        self.x = data.transform.translation.x
        self.y = data.transform.translation.y
        self.z = data.transform.translation.z

        # calibration
        if not self.isCalibrated:
            self.x0 = self.x
            self.y0 = self.y
            self.z0 = self.z
            self.yaw0 = YAW_OFFSET
            self.isCalibrated = True
        
        self.x, self.y, self.z = self.vicon2world(self.x, self.y, self.z)
        self.yaw = np.arctan2(np.sin(self.yaw-self.yaw0), np.cos(self.yaw-self.yaw0))

        print(f'\nroll: {np.rad2deg(self.roll)}\npitch: {np.rad2deg(self.pitch)}\nyaw: {np.rad2deg(self.yaw)}\n\nx: {self.x}\ny: {self.y}\nz: {self.z}\n')
    
    def vicon2world(self, x_v, y_v, z_v):
        x_w = x_v - self.x0
        y_w = y_v - self.y0
        z_w = z_v - self.z0

        return x_w, y_w, z_w



if __name__ == '__main__':
    try:
        log = DataLogger()
        while not rospy.is_shutdown():
            time.sleep(0.1)
    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass


        