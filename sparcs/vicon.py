import rospy
#!/usr/bin/python3
from geometry_msgs.msg import TransformStamped
from vicon_bridge.msg import Markers
from tf.transformations import euler_from_quaternion
import numpy as np
from time import sleep
from std_msgs.msg import Float32

# VICON settings
VICON_CAR_TOPIC = '/vicon/bfmc_car/bfmc_car'
VICON_MARKERS_TOPIC = '/vicon/markers'
YAW_OFFSET = np.deg2rad(-90.0)
X0 = -1.3425
Y0 = -2.35

class Vicon():
    def __init__(self, debug=False) -> None:
        self.time_stamp = None
        self.x          = 0
        self.y          = 0
        self.z          = 0
        self.euler      = 0
        self.roll       = 0
        self.pitch      = 0
        self.yaw        = 0
        self.x0, self.y0, self.yaw0 = X0, Y0, YAW_OFFSET
        # create publishers and subscribers to control the car
        self.sub_vicon_car = rospy.Subscriber(VICON_CAR_TOPIC, TransformStamped, self.vicon_car_callback)
        self.sub_vicon_markers = rospy.Subscriber(VICON_MARKERS_TOPIC, Markers, self.vicon_markers_callback)
        self.pub_speed = rospy.Publisher('/automobile/command/speed', Float32, queue_size=1)
        self.pub_steer = rospy.Publisher('/automobile/command/steer', Float32, queue_size=1)
        self.pub_stop = rospy.Publisher('/automobile/command/stop', Float32, queue_size=1)
        sleep(0.3) # wait for publishers to initialize
        self.prev_speed = 0.0
        # self.debug=debug

    def vicon_car_callback(self, data) -> None:  
        self.euler = euler_from_quaternion([data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w]) 
        self.roll   = self.euler[0]
        self.pitch  = self.euler[1]
        self.yaw    = self.euler[2]

        # x = data.transform.translation.x
        # y = data.transform.translation.y
        # z = data.transform.translation.z
        # self.x, self.y, self.z = self.vicon2world(x, y, z)

        self.time_stamp = data.header.stamp
        
        self.yaw = np.arctan2(np.sin(self.yaw-self.yaw0), np.cos(self.yaw-self.yaw0))
        # print(f'\nroll: {np.rad2deg(self.roll)}\npitch: {np.rad2deg(self.pitch)}\nyaw: {np.rad2deg(self.yaw)}\n\nx: {self.x}\ny: {self.y}\nz: {self.z}\n')

      
    def vicon_markers_callback(self, data) -> None:  
        x = (data.markers[3].translation.x)*1e-3
        y = (data.markers[3].translation.y)*1e-3
        z = (data.markers[3].translation.z)*1e-3
        
        self.x, self.y, self.z = self.vicon2world(x, y, z)
        # if self.debug:
            # print(f'marker: {data.markers[3].marker_name}') 
        # print(f'\nx_rear: {self.x_rear}\ny_rear: {self.y_rear}\nz_rear: {self.z_rear}\n')


    def vicon2world(self, x_v, y_v, z_v):
        x_w = x_v - self.x0
        y_w = y_v - self.y0
        z_w = z_v
        return x_w, y_w, z_w

    def stop(self):
        self.pub_stop.publish(0)

    def drive(self, speed, steer):
        if speed != self.prev_speed:
            self.pub_speed.publish(speed)
            self.prev_speed = speed
        steer = np.rad2deg(steer)
        steer = np.clip(steer, -28.0, 28.0) #limit steer between -28 and 28 deg
        self.pub_steer.publish(steer)

if __name__ == '__main__':
    try:
        vicon = Vicon()
        rospy.init_node('vicon_debug_node', anonymous=False)
        while not rospy.is_shutdown():
            print(f'\nx_rear: {vicon.x:.3f}\ny_rear: {vicon.y:.3f}\nyaw: {np.rad2deg(vicon.yaw):.1f}\n')
            sleep(0.1)
    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass