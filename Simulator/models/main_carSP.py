#!/usr/bin/python3
from pickle import FALSE
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

from control.automobile_data_pi import AutomobileDataPi     # I/O car manager
from controllerSP import ControllerSpeed              # Lane Keeping
from helper_functions import *                  # helper functions
from detection import Detection                 # detection

# FLAGS
SHOW_IMGS = False
SHOW_MATPLOTLIB = False

# PARAMETERS
DESIRED_SPEED = 0.7   # [m/s]
CURVE_SPEED = 0.6

PLOT_BUF_SIZE = 100
NUM_SUBPLOTS = 3
data_plot = {}
data_plot_labels = ['time', 'short_e2', 'short_e3', 'long_e3', 'angle_ref', 'speed_ref']


# init the car flow of data
car = AutomobileDataPi(trig_control=True, trig_cam=False, trig_gps=False, trig_bno=True, trig_enc=True, trig_sonar=True)
# init controller
controller = ControllerSpeed(desired_speed=DESIRED_SPEED, curve_speed=CURVE_SPEED)
# initialize detector
detect = Detection()

#load camera with opencv
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_FPS, 30)

LOOP_FPS = 30.0

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
        if SHOW_MATPLOTLIB:
            for label in data_plot_labels:
                data_plot[label] = collections.deque(np.zeros(PLOT_BUF_SIZE),maxlen=PLOT_BUF_SIZE)

            fig, axs = plt.subplots(NUM_SUBPLOTS)
            fig.suptitle('Vertically stacked subplots')
            plt.ion()
            plt.show()

    # RESET THE CAR POSITION
    car.stop()
    print("Starting in 1 seconds...")
    sleep(1)
    car.drive_speed(speed=DESIRED_SPEED)

    start_time_loop = 0.0
    start_time = rospy.get_time()
    try:
        while not rospy.is_shutdown():
            # -------------------- BREAKING CHECK ---------------------------
            if car.filtered_sonar_distance < 0.3:
                print('stopping car for obstacle ...')
                print(f'sonar distance: {car.filtered_sonar_distance}')
                car.stop()
            else:
                start_time_loop = time()

                # -------------------- CAMERA FRAME --------------------
                # frame = car.frame.copy()
                ret, frame = cap.read()
                if not ret:
                    print("No image from camera")
                    frame = np.zeros((480, 640, 3), np.uint8)
                    continue

                # -------------------- LANE KEEPING --------------------
                #Neural network control
                # short_e2, short_e3, short_point_ahead = detect.detect_lane(frame, show_ROI=False, faster=False)
                long_e3, long_point_ahead = detect.detect_lane_ahead(frame, show_ROI=False, faster=False)

                speed_ref, angle_ref = controller.get_control_speed(0.0, 0.0, long_e3)
                
                # -------------------- STOPPING LOGIC --------------------
                # dist = detect.detect_stop_line(frame, show_ROI=SHOW_IMGS)
                # if 0.0 < dist < 0.35:
                #     print('Slowing down')
                #     speed_ref = DESIRED_SPEED * 0.5
                #     if 0.0 < dist < 0.15:
                #         print('Stopping')
                #         speed_ref = 0.0

                # -------------------- ACTUATION --------------------
                car.drive(speed=speed_ref, angle=np.rad2deg(angle_ref))            

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
                loop_time = time() - start_time_loop
                if loop_time < 1/LOOP_FPS:
                    sleep(1/LOOP_FPS - loop_time)
                
                if SHOW_IMGS:
                    if SHOW_MATPLOTLIB:
                        data_plot['time'].append(rospy.get_time()-start_time)
                        # data_plot['short_e2'].append(short_e2)
                        # data_plot['short_e3'].append(np.rad2deg(short_e3))
                        data_plot['long_e3'].append(np.rad2deg(long_e3))
                        data_plot['angle_ref'].append(np.rad2deg(angle_ref))
                        data_plot['speed_ref'].append(speed_ref)

                        axs[0].cla()
                        axs[0].set_ylim(-20,20)
                        axs[0].set_ylabel('Heading error [deg]')
                        axs[0].set_xlabel('Time [ms]')
                        # axs[0].legend()
                        axs[0].plot(data_plot['time'], data_plot['short_e3'], '-k', label='short_e3')
                        axs[0].plot(data_plot['time'], data_plot['long_e3'], '-b', label='long_e3')
                        
                        axs[1].cla()
                        axs[1].set_ylim(0,DESIRED_SPEED*1.01)
                        axs[1].set_ylabel('Speed_ref [m/s]')
                        axs[1].set_xlabel('Time [ms]')
                        axs[1].plot(data_plot['time'], data_plot['speed_ref'], '-k', label='speed_ref')
                        axs[1].legend(loc='lower right',frameon=False)
                    
                    # =================================================
                    #project point ahead
                    # frame, proj = project_onto_frame(frame, car, point_ahead, False, color=(200, 200, 100))
                    frame, proj = project_onto_frame(frame, car, long_point_ahead, False, color=(200, 100, 200))
                    if proj is not None:
                        #convert proj to cv2 point
                        proj = (int(proj[0]), int(proj[1]))
                        #draw line from bottom half to proj
                        cv.line(frame, (320//2,479//2), proj, (200, 100, 200), 2)
                    # frame, proj = project_onto_frame(frame, car, short_point_ahead, False, color=(200, 200, 100))
                    if proj is not None:
                        #convert proj to cv2 point
                        proj = (int(proj[0]), int(proj[1]))
                        #draw line from bottom half to proj
                        cv.line(frame, (320//2,479//2), proj, (200, 200, 100), 2)

                    cv.imshow('Frame preview', frame)
                    key = cv.waitKey(1)
                    if key == 27:
                        car.stop()
                        sleep(1.5)
                        cv.destroyAllWindows()
                        print("Shutting down...")
                        exit(0)
    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass
      