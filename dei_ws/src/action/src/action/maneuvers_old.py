#!/usr/bin/python3
import numpy as np
from mimetypes import init
from pickle import FALSE
import cv2 as cv
import rospy
from time import sleep
import os

from control.helper_functions import *

DELAY = 1.0

class Maneuvers():
    def __init__(self):
        # Global delay for maneuvers
        self.delay = DELAY # [s]
        self.park_wait_time = 3.0# [s]

        # PARALLEL PARKING
        # parking spot dimensions
        self.park_pl_length = 0.7

        # midpoint coords in the local reference frame
        # self.park_pl_midpoint_dist = 0.4106
        self.park_pl_midpoint_dist = 0.45
        # self.park_pl_midpoint_x = 0.169# [m]
        # self.park_pl_midpoint_y = -0.40# [m]
        # self.park_pl_midpoint_yaw = np.deg2rad(-40.0)# [rad]
        
        # control constants
        self.park_pl_turn_right = 25.0# [deg]
        self.park_pl_turn_left = -25.0# [deg]
        self.park_pl_forward = 0.1# [m/s]
        self.park_pl_backward = -0.1# [m/s]

        # PERPENDICULAR PARKING
        # endpoint coords in the local reference frame
        self.park_pp_endpoint_dist = 0.8# [m]
        self.park_pp_endpoint_yaw = -np.deg2rad(90)

        # control constants
        self.park_pp_turn_right = 25.0# [deg]
        self.park_pp_turn_left = -25.0# [deg]
        self.park_pp_forward = 0.2# [m/s]
        self.park_pp_backward = -0.2# [m/s]

        # INTERSECTION NAVIGATION
        self.int_straight_forward = 0.2# [m/s]
        self.int_right_forward = 0.2# [m/s]
        self.int_left_forward = 0.2# [m/s]
        self.int_turn_right = 15.0# [deg] right turn has radius 0.665 m
        self.int_turn_left = -20.0# [deg] left turn has radius 1.035 m

    def parallel_parking(self, car):
        car.reset_rel_pose()

        # FIRST PART
        print('FIRST PART - START - turn right and go backwards')
        car.drive_angle(self.park_pl_turn_right)
        # sleep(self.delay)
        car.drive_speed(self.park_pl_backward)
        #while car.xLoc<self.park_pl_midpoint_x or car.yawLoc>self.park_pl_midpoint_yaw:
        while car.distLoc < self.park_pl_midpoint_dist:
            #print(f'current x and yaw: {car.xLoc, np.rad2deg(car.yawLoc)}')
            sleep(0.001)# keep going
        # car.drive_speed(0.0)
        # car.stop(self.park_pl_turn_left)
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('FIRST PART - END')

        # sleep(self.delay)
        car.reset_rel_pose()
        
        # SECOND PART
        print('SECOND PART - START - turn left and go backwards')
        car.drive_angle(self.park_pl_turn_left)
        # sleep(self.delay)
        car.drive_speed(self.park_pl_backward)
        #while car.xLoc<self.park_pl_midpoint_x*2.0 or car.yawLoc<0.0:
        while car.distLoc < self.park_pl_midpoint_dist:
            sleep(0.001)# keep going
        # car.drive_speed(0.0)
        car.stop()
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('SECOND PART - END')

        sleep(self.delay)
        car.reset_rel_pose()
        #pspot_back = car.yLoc

        # THIRD PART
        print('THIRD PART - START - go to the center of the parking spot')
        car.drive_angle(0.0)
        car.drive_speed(self.park_pl_forward)
        #while car.yLoc - pspot_back < 0.3*self.park_pl_length:
        while car.distLoc < 0.2:
            sleep(0.001)# keep going
        # car.drive_speed(0.0)
        car.stop()
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('THIRD PART - END')

        print("*** END OF THE PARKING MANEUVER ***")
        sleep(self.park_wait_time)
        car.reset_rel_pose()

        # FOURTH PART
        print('FOURTH PART - START - go back up to the end of the parking spot')
        car.drive_speed(self.park_pl_backward)
        #while car.yLoc - pspot_back > 0.0:
        while car.distLoc < 0.2:
            sleep(0.001)# keep going
        # car.drive_speed(0.0)
        car.stop()
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('FOURTH PART - END')

        sleep(self.delay)
        car.reset_rel_pose()
        
        # FIFTH PART
        print('FIFTH PART - START - turn left and go forwards')
        car.drive_angle(self.park_pl_turn_left)
        # sleep(self.delay)
        car.drive_speed(self.park_pl_forward)
        #while car.xLoc>self.park_pl_midpoint_x or car.yawLoc > self.park_pl_midpoint_yaw:
        while car.distLoc < self.park_pl_midpoint_dist:
            #print(f'current x and yaw: {car.xLoc, np.rad2deg(car.yawLoc)}')
            sleep(0.001)# keep going
        # car.stop(self.park_pl_turn_right)
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('FIFTH PART - END')

        # sleep(self.delay)
        car.reset_rel_pose()

        # SIXTH PART
        print('SIXTH PART - START - turn right and go forwards')
        car.drive_angle(self.park_pl_turn_right)
        # sleep(self.delay)
        car.drive_speed(self.park_pl_forward)
        #while car.xLoc>0.0 or car.yawLoc<0.0:
        while car.distLoc < self.park_pl_midpoint_dist:
            sleep(0.001)# keep going
        car.drive_angle(0.0)
        # car.drive_speed(0.0)
        car.stop()
        #sleep(self.delay)
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('SIXTH PART - END')    
        
    def perpendicular_parking(self, way_exit = 'right'):
        # FIRST PART
        print('FIRST PART - START - turn right and go backwards')
        car.drive_angle(self.park_pp_turn_right)
        sleep(self.delay)
        target_speed = self.park_pp_backward
        while not np.isclose(car.speed_meas_median, target_speed, atol=0.01):
            car.drive_speed(target_speed)            
        while car.distLoc < self.park_pp_endpoint_dist or car.yawLoc>self.park_pp_endpoint_yaw:
            #print(f'current x and yaw: {car.xLoc, np.rad2deg(car.yawLoc)}')
            sleep(0.001)# keep going
        car.drive_speed(0.0)
        sleep(self.delay)
        car.drive_angle(0.0)
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('FIRST PART - END')

        sleep(self.delay)
        car.reset_rel_dist()
        
        # SECOND PART
        print('SECOND PART - START - turn right and go forwards')
        car.drive_angle(self.park_pp_turn_right)
        sleep(self.delay)
        target_speed = self.park_pp_forward
        while not np.isclose(car.speed_meas_median, target_speed, atol=0.01):
            car.drive_speed(target_speed)
        while car.distLoc < self.park_pp_endpoint_dist or car.yawLoc<0.0:
            #print(f'current x and yaw: {car.xLoc, np.rad2deg(car.yawLoc)}')
            sleep(0.001)# keep going
        car.drive_speed(0.0)
        car.drive_angle(0.0)
        sleep(self.delay)
        print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
        print('SECOND PART - END')
    
    def go_straight(self,car,ds=0.1, speed = 0.2):
        car.reset_rel_dist()
        car.drive_angle(0.0)
        car.drive_speed(speed)
        
        while car.distLoc < ds:
            sleep(0.001)# keep going
        return
    
    def turn_right(self,car, ds=0.1):
        car.reset_rel_dist()
        car.drive_angle(self.int_turn_right)
        #sleep(self.delay)
        car.drive_speed(self.int_right_forward)
        # right turn has radius 66.5cm
        # while car.xLoc < 0.665:
        # while car.xLoc < 0.1:
        while car.distLoc < ds:
            print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
            sleep(0.001)# keep going
        print('END OF RIGHT TURN')
        
        return

    def turn_left(self,car, speed = 0.1):
        car.reset_rel_dist()
        # car.drive_angle(self.int_turn_left)
        car.drive_angle(-15.0)
        #sleep(self.delay)
        car.drive_speed(speed)

        # left turn has radius 1.035m
        while car.distLoc < 1.0:
            #print(f'pose: {car.xLoc, car.yLoc, car.yawLoc, car.distLoc}')
            sleep(0.001)# keep going
        print('END OF LEFT TURN')
        return

# if __name__ == '__main__':

#     # init the car data
#     car = Automobile_Data(trig_control=True, trig_enc=True, trig_cam=False, trig_gps=False, trig_bno=True, trig_estimation=False, trig_sonar=True)
#     maneuvers = Maneuvers()

#     os.system('cls' if os.name=='nt' else 'clear')

#     car.stop()
#     car.drive_angle(0.0)
#     sleep(2.0)

#     try:
        
#         #print('PARALLEL PARKING - START - reset relative pose')
#         car.reset_rel_pose()
#         print(f'pose: {car.xLoc, car.yLoc, car.yawLoc}')
        
#         while not rospy.is_shutdown():
#             # car.drive_angle(10)
#             # sleep(0.5)
#             # car.drive_angle(-10)
#             # sleep(0.5)
#             os.system('cls' if os.name=='nt' else 'clear')
#             print(f'the distance of an obstacle ahead is {car.obstacle_ahead} m')
#             sleep(0.01)
#         #print('PARALLEL PARKING - END')
#         # target_speed = 0.2
#         # k=1
#         # while not rospy.is_shutdown():
#         #     target_speed = -target_speed
#         #     j=1
#         #     while not np.isclose(car.speed_meas_median, target_speed, atol=0.01):
#         #         car.drive_speed(target_speed)
#         #         print(f'{k} th maneuver, attempt:{j}, target speed:{target_speed}, measured speed median:{car.speed_meas_median}')
#         #         #print(f'speed buffer is:{list(car.speed_meas_buffer)}')
#         #         sleep(0.001)
#         #         j += 1
#         #     sleep(2)
#         #     k += 1
            
#     except rospy.ROSInterruptException:
#         pass
      
