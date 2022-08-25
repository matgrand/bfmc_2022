#!/usr/bin/python3
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion

import rospy, collections, os, signal
from std_msgs.msg import Float32, Bool

import numpy as np
import matplotlib.pyplot as plt
import time

VICON_TOPIC = '/vicon/bfmc_car/bfmc_car'
YAW_OFFSET = np.deg2rad(-90.0)
X0 = -1.3425
Y0 = -2.35

SONAR_DEQUE_LENGTH = 5

SPEED = 0.3
DIST_AHEAD = 0.6

class Vicon():
    def __init__(self) -> None:
        self.time_stamp = None
        self.x          = 0
        self.y          = 0
        self.z          = 0
        self.euler      = 0
        self.roll       = 0
        self.pitch      = 0
        self.yaw        = 0

        self.x0, self.y0, self.yaw0 = X0, Y0, YAW_OFFSET

        self.sub_vicon = rospy.Subscriber(VICON_TOPIC, TransformStamped, self.vicon_callback)
        # rospy.init_node('logdata_node', anonymous=False)


    def vicon_callback(self, data) -> None:  
        self.euler = euler_from_quaternion([data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w]) 
        self.roll   = self.euler[0]
        self.pitch  = self.euler[1]
        self.yaw    = self.euler[2]

        x = data.transform.translation.x
        y = data.transform.translation.y
        z = data.transform.translation.z

        self.time_stamp = data.header.stamp
        
        self.x, self.y, self.z = self.vicon2world(x, y, z)
        self.yaw = np.arctan2(np.sin(self.yaw-self.yaw0), np.cos(self.yaw-self.yaw0))

        # print(f'\ntime: {self.time_stamp}\nyaw: {np.rad2deg(self.yaw)}\n\nx: {self.x}\ny: {self.y}\nz: {self.z}\n')
    
    def vicon2world(self, x_v, y_v, z_v):
        x_w = x_v - self.x0
        y_w = y_v - self.y0
        z_w = z_v
        return x_w, y_w, z_w


class SparcsTracker():
    def __init__(self, path, noise_deg, noise_time) -> None:
        self.path = path.T

        #reverse path 
        # self.path = np.flip(self.path, axis = 0)

        #init node

        self.pub_speed = rospy.Publisher('/automobile/command/speed', Float32, queue_size=1)
        self.pub_steer = rospy.Publisher('/automobile/command/steer', Float32, queue_size=1)
        self.pub_stop = rospy.Publisher('/automobile/command/stop', Float32, queue_size=1)
        
        time.sleep(0.3)
        self.pub_speed.publish(SPEED)

        



    def update(self,x,y,yaw) -> None:
        print(f'x: {x} \ny: {y} \nyaw: {np.rad2deg(yaw)}\n')

        p = np.array([x,y]).T

        #find closest point on path
        diff = self.path - p
        min_index = np.argmin(np.linalg.norm(self.path-p,axis=1))

        index_point_ahead = (min_index + int(100*DIST_AHEAD)) % len(self.path) 
        p_ahead = self.path[index_point_ahead]


        yaw_ref = np.arctan2(self.path[index_point_ahead,1]-y,self.path[index_point_ahead,0]-x)

        he = np.arctan2(np.sin(yaw_ref-yaw), np.cos(yaw_ref-yaw))

        print(f'x_a: {p_ahead[0]} \ny_a: {p_ahead[1]} \nhe: {np.rad2deg(he)}\n')

        delta = -1.0*np.arctan((2*0.26*np.sin(he))/DIST_AHEAD)

        delta_deg = np.rad2deg(delta)
        delta_deg = np.clip(delta_deg, -28.0, 28.0)

        self.pub_steer.publish(delta_deg)

    def stop(self):
        self.pub_stop.publish(0)



if __name__ == '__main__':
    try:
        path = np.load('sparcs_path.npy')
        # plt.plot(path[0],path[1])
        # plt.axis('equal')
        # plt.show()

        rospy.init_node('pc_controller', anonymous=False)
        vicon = Vicon() #intialize the rosnode and topic

        st = SparcsTracker(path=path, noise_deg=None, noise_time=None)

        def handler(signum, frame):
            print("Exiting ...")
            st.stop()
            time.sleep(.99)
            exit()
        signal.signal(signal.SIGINT, handler)

        while not rospy.is_shutdown():
            time.sleep(0.01)
            st.update(vicon.x, vicon.y, vicon.yaw)




    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass


        