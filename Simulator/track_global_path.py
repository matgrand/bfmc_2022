#!/usr/bin/python3
import os, signal
import cv2 as cv
import rospy
import numpy as np
from time import sleep, time
os.system('clear')
print('Track local path starting...')
# from automobile_data import Automobile_Data
from automobile_data_simulator import AutomobileDataSimulator
from PathPlanning4 import PathPlanning
from helper_functions import *
from controller3 import Controller
from detection import Detection

map = cv.imread('data/2021_VerySmall.png')

LOOP_DELAY = 0.01
CHECKPOINTS = [86,99,112,337,463,240]
DESIRED_SPEED = 1.2 # m/s

K1 = 0.0
K2 = 0.0
K3 = .9
K3D = 0.0
FF = 0.0
DIST_POINT_AHEAD = 0.5

if __name__ == '__main__':

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('frame',640,480)
    cv.namedWindow('Map', cv.WINDOW_NORMAL)
    cv.resizeWindow('Map', 600, 600)


    # init the car data
    os.system('rosservice call /gazebo/reset_simulation')
    car = AutomobileDataSimulator(trig_cam=True, trig_gps=True, trig_bno=True, 
                               trig_enc=True, trig_control=True, trig_estimation=False, trig_sonar=True)

    
    #stop the car with ctrl+c
    def handler(signum, frame):
        print("Exiting ...")
        car.stop()
        cv.destroyAllWindows()
        sleep(0.1)
        exit()
    signal.signal(signal.SIGINT, handler)

    #initiliaze all the neural networks for detection and lane following
    detect = Detection()

    controller = Controller(k1=K1,k2=K2,k3=K3, k3D=K3D, ff=FF, cm_ahead=int(100*DIST_POINT_AHEAD))

    path_planner = PathPlanning(map)

    path_planner.generate_path_passing_through(CHECKPOINTS)
    path_planner.draw_path()
    path = path_planner.path

    map1 = map
    draw_car(map1, car.x_true, car.y_true, car.yaw)
    cv.imshow('Map', map1)
    cv.waitKey(1)

    try:
        car.stop()
        fps_avg = 0.0
        fps_cnt = 0
        while True:
            if car.trust_gps:
                break
            sleep(0.1)
            print('Waiting for GPS...')

        while not rospy.is_shutdown():
            os.system('cls' if os.name=='nt' else 'clear')

            loop_start_time = time()
            frame = car.frame.copy()


            # stopline_x, stopline_y, stopline_angle = detect.detect_stop_line2(car.frame, show_ROI=True)

            # frame, _ = project_stopline(frame, car, stopline_x, stopline_y, stopline_angle, color=(0,200,0))


            #control
            car_est_pos = np.array([car.x_est, car.y_est])
            idx_car_on_path = np.argmin(np.linalg.norm(path-car_est_pos, axis=1))
            p_ahead = path[min(len(path)-1, idx_car_on_path+int(100*DIST_POINT_AHEAD))]
            e3 = diff_angle(np.arctan2(p_ahead[1]-car_est_pos[1], p_ahead[0]-car_est_pos[0]),car.yaw)
            
            output_speed, output_angle = controller.get_control(e2=0, e3=e3, curv=0, desired_speed=DESIRED_SPEED)

            car.drive(output_speed, np.rad2deg(output_angle))


            ################################################################

            map1 = map.copy()
            draw_car(map1, car.x, car.y, car.yaw, color=(180,0,0))
            color=(255,0,255) if car.trust_gps else (100,0,100)
            draw_car(map1, car.x_true, car.y_true, car.yaw, color=(0,180,0))
            draw_car(map1, car.x_est, car.y_est, car.yaw, color=color)
            cv.imshow('Map', map1)
            cv.waitKey(1)

            ## DEBUG INFO
            print(car)
            print(f'Lane detection time = {detect.avg_lane_detection_time:.1f} [ms]')
            print(f'FPS = {fps_avg:.1f},  loop_cnt = {fps_cnt}')

            cv.imshow('frame', frame)
            if cv.waitKey(1) == 27:
                cv.destroyAllWindows()
                break
            
            loop_time = time() - loop_start_time
            fps_avg = (fps_avg * fps_cnt + 1.0 / loop_time) / (fps_cnt + 1)
            fps_cnt += 1

    except KeyboardInterrupt:
        print("Shutting down")
        car.stop()
        sleep(.5)
        cv.destroyAllWindows()
        exit(0)
    except rospy.ROSInterruptException:
        pass
      