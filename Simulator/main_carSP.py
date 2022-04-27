#!/usr/bin/python3
from this import d
import cv2 as cv
import rospy
import numpy as np
from time import sleep
from time import time
import os
import signal

from matplotlib import pyplot as plt
import collections  

from automobile_data_simulator import AutomobileDataSimulator     # I/O car manager
from controllerSP import Controller              # Lane Keeping
from helper_functions import *                  # helper functions
from detection import Detection                 # detection

# PARAMETERS
SAMPLE_TIME = 0.01      # [s]
WAIT_TIME_STOP = 3.0    # [s]
DESIRED_SPEED = 1.5    # [m/s]
OBSTACLE_DISTANCE = 0.20 # [m]

SLOWDOWN_DIST_VALUE = 0.2     # [m]
STOP_DIST_VALUE = 0.6         # [m]

PLOT_BUF_SIZE = 100
STEERING_CONTROL_INTERVAL = 0.01    # [s]

SHOW_IMGS = False

# Pure pursuit controller parameters
k1 = 0.0 #4.0 gain error parallel to direction (speed)
k2 = 0.0 #2.0 perpedndicular error gain
k3 = 0.7 #1.5 yaw error gain 
#dt_ahea  = 0.5 # [s] how far into the future the curvature is estimated, feedforwarded to yaw controller
ff_curvature = 0.0 # feedforward gain

# init the car flow of data
car = AutomobileDataSimulator(trig_control=True, trig_cam=True, trig_gps=True, trig_bno=True, trig_enc=True, trig_sonar=True)
# init controller
controller = Controller()
# initialize detector
detect = Detection()

rate = rospy.Rate(100)

speed_ref, angle_ref = 0.0, 0.0    

# Handler for exiting the program with CTRL+C
def handler(signum, frame):
    print("Exiting ...")
    car.stop()
    cv.destroyAllWindows()
    exit(1)
signal.signal(signal.SIGINT, handler)


if __name__ == '__main__':

    if SHOW_IMGS:
        cv.namedWindow('Frame preview', cv.WINDOW_NORMAL)
        cv.resizeWindow('Frame preview', 320, 240)
        # plotting
        time_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        lateralError_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        filteredLateralError_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        headingError_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        filteredHeadingError_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        estimatedDelta_list = collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        speedRef_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)
        angleRef_list=collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)

        fig, axs = plt.subplots(3)
        fig.suptitle('Vertically stacked subplots')
        plt.ion()
        plt.show()
        # cv.namedWindow('lane_detection', cv.WINDOW_NORMAL)
        # cv.resizeWindow('lane_detection', 200, 200)

    start_time_stamp = 0.0      # [s]

    # RESET THE CAR POSITION
    car.stop()
    print("Starting in 1 seconds...")
    sleep(1)
    car.drive_speed(speed=DESIRED_SPEED)

    starting_time = time()
    try:
        while not rospy.is_shutdown():
            # -------------------- BREAKING CHECK ---------------------------
            if car.filtered_sonar_distance < 0.3:
                print('stopping car for obstacle ...')
                print(f'sonar distance: {car.filtered_sonar_distance}')
                car.stop()
            else:
                start_time_stamp = time() * 1000.0# [ms] loop start time

                # Get the image from the camera
                frame = car.frame.copy()
                # ret, frame = cap.read()
                # if not ret:
                #     print("No image from camera")
                #     frame = np.zeros((480, 640, 3), np.uint8)
                #     continue

                # -------------------- LANE KEEPING --------------------
                #Neural network control
                lane_info = detect.detect_lane(frame, show_ROI=False)
                e2, e3, point_ahead = lane_info
                speed_ref, angle_ref = controller.get_control(e2, e3, car.encoder_velocity, point_ahead, DESIRED_SPEED)
                alpha_filt = controller.alpha_filt
                e_filt = controller.e_filt
                estimated_delta = controller.delta

                # dist = detect.detect_stop_line(frame, show_ROI=SHOW_IMGS)

                # #stopping logic
                # if 0.0 < dist < 0.35:
                #     print('Slowing down')
                #     speed_ref = DESIRED_SPEED * 0.5
                #     if 0.0 < dist < 0.15:
                #         print('Stopping')
                #         speed_ref = 0.0


                # -------------------- SIGNS ---------------------------
                # sign = detect.detect_sign(frame, show_ROI=SHOW_IMGS)

                # -------------------- INTERSECTION --------------------

                # -------------------- ACTUATION --------------------
                car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))
                

                # -------------------- LOOP SAMPLING TIME --------------------
                #r.sleep()
                # sleep(0.1)

            

                # -------------------- DEBUG --------------------
                # os.system('cls' if os.name=='nt' else 'clear')
                # print(f'Distance to stop line: {dist:.2f}')
                # print(f'Net out:\n {lane_info}')

                # #project point ahead
                # print(f'angle_ref: {np.rad2deg(angle_ref)}')
                # print(f"point_ahead: {point_ahead}")

                # print(f'Loop Time: {time() * 1000.0 - start_time_stamp} ms')
                # print(f'FPS: {1.0/(time() * 1000.0 - start_time_stamp)*1000.0}')
                # print(f'current speed: {car.encoder_velocity}')
                # print(f'yaw: {car.yaw}')
                # print(f'Lane detection time = {detect.avg_lane_detection_time:.2f} ms')
                # print(f'Sign detection time = {detect.avg_sign_detection_time:.2f} ms')

                if SHOW_IMGS:
                    time_list.append(time()-starting_time)

                    headingError_list.append(np.rad2deg(-e3))
                    filteredHeadingError_list.append(np.rad2deg(alpha_filt))

                    lateralError_list.append(e2)
                    filteredLateralError_list.append(e_filt)

                    estimatedDelta_list.append(np.rad2deg(estimated_delta))

                    speedRef_list.append(speed_ref)
                    angleRef_list.append(np.rad2deg(angle_ref))

                    plt.cla()
                    axs[0].set_ylim(-0.2,0.2)
                    axs[0].plot(time_list, lateralError_list, '-k')
                    axs[0].plot(time_list, filteredLateralError_list, '-b')
                    axs[0].set_ylabel("lateral error [m]")
                    # =================================================
                    axs[1].set_ylim(-20,20)
                    axs[1].plot(time_list, headingError_list, '-k')
                    axs[1].plot(time_list, filteredHeadingError_list, '-b')
                    axs[1].set_ylabel("heading error [deg]")
                    # =================================================
                    # axs[2].set_ylim(-1,1)
                    # axs[2].plot(time_list, speedRef_list, '-k')
                    # axs[2].set_ylabel("speed reference [m/s]")
                    axs[2].set_ylim(-30,30)
                    axs[2].plot(time_list, estimatedDelta_list, '-b')
                    axs[2].plot(time_list, angleRef_list, '-k')
                    axs[2].set_ylabel("estimated delta [deg]")
                    # =================================================
                    #project point ahead
                    # frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
                    frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
                    if proj is not None:
                        #convert proj to cv2 point
                        proj = (int(proj[0]), int(proj[1]))
                        #draw line from bottmo half to proj
                        cv.line(frame, (320,479), proj, (200, 200, 100), 2)

                    cv.imshow('Frame preview', frame)
                    key = cv.waitKey(1)
                    if key == 27:
                        car.stop()
                        sleep(1.5)
                        cv.destroyAllWindows()
                        print("Shutting down...")
                        exit(0)
            rate.sleep()

    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass
      