#!/usr/bin/python3
import cv2 as cv
import rospy
import numpy as np
from time import sleep
from time import time
import os

from automobile_data import Automobile_Data
from helper_functions import *
from controller3 import Controller

# PARAMETERS
sample_time = 0.01 # [s]
max_angle = 30.0    # [deg]
max_speed = 0.5  # [m/s]
desired_speed = 0.25 # [m/s]
path_step_length = 0.01 # [m]
# CONTROLLER
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 3.0 #2.0 perpedndicular error gain
k3 = 1.0 #1.5 yaw error gain #inversely proportional
#dt_ahea  = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

if __name__ == '__main__':

    # init the car data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True)

    # init controller
    controller = Controller(k1=k1, k2=k2, k3=k3, ff=ff_curvature)

    start_time_stamp = 0.0

    car.stop()
    car.drive_angle(0.0)

    print("Starting...")

    sleep(1.5)

    car.drive_speed(speed=desired_speed)

    #tests
    previous_e3 = 0.0

    try:
        going_forward = True
        backward_timer_start = 0.0

        while not rospy.is_shutdown():
            # start timer in milliseconds
            start_time_stamp = time() * 1000.0

            # Get the image from the camera
            frame = car.cv_image.copy()

            #Neural network control
            speed_ref, angle_ref, net_out, point_ahead = controller.get_nn_control(frame, desired_speed)

            dist = net_out[2]

            if dist > 0.3: 
                speed_ref = 0.5 * speed_ref

            if dist > 1.5 and going_forward:
                print('Stopping for intersection ahead ', dist)
                car.stop()
                sleep(2.0)
                car.drive_speed(speed=-speed_ref)
                car.drive_angle(-angle_ref)
                going_forward = False
                backward_timer_start = time()

            if (time() - backward_timer_start) > 3.5 and not going_forward:
                going_forward = True
                car.drive_speed(speed=speed_ref) 

            angle_ref = angle_ref if going_forward else - angle_ref

            car.drive_angle(angle=np.rad2deg(angle_ref)) 

            os.system('cls' if os.name=='nt' else 'clear')
            print(f'Net out:\n {net_out}')

            #project point ahead
            print(f'angle_ref: {np.rad2deg(angle_ref)}')
            print(f"point_ahead: {point_ahead}")
            print(f'Dist: {dist:.3f}')
            frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
            if proj is not None:
                #convert proj to cv2 point
                proj = (int(proj[0]), int(proj[1]))
                #draw line from bottmo half to proj
                cv.line(frame, (320,479), proj, (50, 50, 250), 2)

            print(f'Loop Time: {time() * 1000.0 - start_time_stamp} ms')
            print(f'FPS: {1.0/(time() * 1000.0 - start_time_stamp)*1000.0}')
            print(f'current speed: {car.speed_meas}')
            print(f'yaw: {car.yaw}')
            # cv.imshow("Frame preview", frame)
            # key = cv.waitKey(1)
            # if key == 27:
            #     car.stop()
            #     cv.destroyAllWindows()
            #     break
            # sleep(0.2)
    except rospy.ROSInterruptException:
        pass
      