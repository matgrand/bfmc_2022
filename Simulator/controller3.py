#!/usr/bin/python3
import numpy as np
import cv2 as cv
import os

from helper_functions import *
from time import sleep, time

# POINT_AHEAD_CM = 35#60#35 #distance of the point ahead in cm
SEQ_POINTS_INTERVAL = 20 #interval between points in the sequence in cm 
NUM_POINTS = 5 #number of points in the sequence
L = 0.4  #length of the car, matched with lane_detection

NOISE_RESET_MEAN = 10 #avg frames after which the noise is reset
NOISE_RESET_STD = 8 #frames max "deviation"

class Controller():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, k3D=0.08, dist_point_ahead=0.35, ff=1.0, cm_ahead=35, folder='training_imgs',
                    training=False, noise_std=np.deg2rad(20)):
        
        #controller paramters
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k3D = k3D 
        self.dist_point_ahead = dist_point_ahead
        self.ff = ff
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0

        self.prev_delta = 0.0
        self.prev_time = 0.0

        self.training = training

        if training:
            #training
            self.folder = folder
            self.curr_data = []
            self.input_data = []
            self.regression_labels = []
            self.classification_labels = []
            self.cnt = 0
            self.noise = 0.0
            self.data_cnt = 0   
            self.noise_std = noise_std
            self.noise_reset = NOISE_RESET_MEAN
            self.seq_points_ahead = []
            self.seq_yaws_ahead = []
            #clear and create file 
            with open(folder+"/input_data.csv", "w") as f:
                f.write("")
            with open(folder+"/regression_labels.csv", "w") as f:
                f.write("")
            with open(folder+"/classification_labels.csv", "w") as f:
                f.write("")
            #delete all images in the folder
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    os.remove(os.path.join(folder, file))
    
    def get_control(self, e2, e3, curv, desired_speed, gains=None):

        self.e2 = e2 #lateral error [m]
        self.e3 = e3 #yaw error [rad]
        alpha = e3

        # # if not self.training:
        # #     curv = curv*30
        # if curv == 0.0: r = 10000.0
        # else: r = 1.0 / ( curv*6.28 ) #radius of curvature

        # #adaptive controller
        # curv100 = 100 * curv
        # if 0.7 < np.abs(curv100) < 2.0: #big curvature, pure pursuit is too aggressive
        #     # print(f'HIGH CURVATURE: {curv100}')
        #     # k2 = 5.0
        #     # k3 = 5.0
        #     k2 = self.k2
        #     k3 = self.k3
        # else:
        k2 = self.k2
        k3 = self.k3

        if gains is not None:
            k1, k2, k3, k3D = gains

        # yaw error (e3), proportional term
        d = self.dist_point_ahead#POINT_AHEAD_CM/100.0 #distance point ahead, matched with lane_detection
        delta = np.arctan((2*L*np.sin(alpha))/d)
        proportional_term = k3 * delta
        # print(f'proportional term: {np.rad2deg(proportional_term):.2f}')

        #derivative term
        k3D = self.k3D
        curr_time = time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time
        diff3 = (delta - self.prev_delta) / dt
        self.prev_delta = delta
        derivative_term = k3D * diff3
        # print(f'derivative term: {np.rad2deg(derivative_term):.2f}')

        #feedforward term
        k3FF = self.ff #0.2 #higher for high speeds
        # ff_term = k3FF * np.arctan(L/r) #from ackerman geometry
        ff_term = -k3FF * curv
        print(f'Feedforward term: {np.rad2deg(ff_term):2f}')
    
        output_angle = ff_term - proportional_term - k2 * e2 - derivative_term
        output_speed = desired_speed - self.k1 * self.e1

        return output_speed, output_angle
        
    def get_training_control(self, car, path_ahead, vd, curvature_ahead):

        #add e2 (lateral error)
        ex = path_ahead[0][0] - car.x_true #x error
        ey = path_ahead[0][1] - car.y_true #y error
        e2 = ex * np.cos(car.yaw) + ey * np.sin(car.yaw) #y error in the body frame

        #get point ahead
        assert len(path_ahead) > max(NUM_POINTS * SEQ_POINTS_INTERVAL, int(100*self.dist_point_ahead)), f"path_ahead is not long enough, len: {len(path_ahead)}"
        point_ahead = path_ahead[min(int(100*self.dist_point_ahead), len(path_ahead)-1),:]
        #translate to car coordinates
        point_ahead = to_car_frame(point_ahead, car, return_size=2)

        self.seq_points_ahead = [path_ahead[(i)*SEQ_POINTS_INTERVAL,:] for i in range(NUM_POINTS+1)]
        self.seq_points_ahead = [to_car_frame(p, car, return_size=2) for p in self.seq_points_ahead]
        #yaws should be changed: they should be calculated wrt the car's not wrt to each other
        self.seq_yaws_ahead = [np.arctan2(self.seq_points_ahead[i][1], self.seq_points_ahead[i][0]) for i in range(NUM_POINTS)]

        alpha = np.arctan2(point_ahead[1], point_ahead[0]+L/2) 
        output_speed, output_angle = self.get_control(e2, alpha, curvature_ahead, vd)

        output_angle = output_angle + self.get_random_noise(std=self.noise_std) #add noise for training
        return output_speed, output_angle, point_ahead
    
    def pack_input_data(self):
        data = [0,0,0,0]
        xd,yd,yawd,curv,path_ahead,stopline_x,stopline_y,stopline_yaw,(state,next,action,dist,_,_,_) = self.curr_data
        if action == "straight":
            data[0] = 1
        elif action == "left":
            data[1] = 1
        elif action == "right":
            data[2] = 1
        elif action == "continue":
            data[3] = 1
        #append to file 
        with open(self.folder+"/input_data.csv", "a") as f:
            #write single elements in the list separated by comma
            for i in range(len(data)-1):
                f.write(str(data[i])+",")
            f.write(str(data[-1])+"\n")

    def pack_regression_labels(self, bb_const=1000.0):
        xd,yd,yawd,curv,path_ahead,stopline_x,stopline_y,stopline_yaw,(state,next,action,dist,_,_,_) = self.curr_data
        if dist is None:
            dist = 10
        #add errors 
        reg_label = [self.e2, self.e3, curv, dist, stopline_x,stopline_y,stopline_yaw]
        #add sequ of points ahead
        for i in range(NUM_POINTS):
            reg_label.append(self.seq_yaws_ahead[i])
        np_arr = np.array(reg_label)
        #append to file
        with open(self.folder+"/regression_labels.csv", "a") as f:
            for i in range(len(reg_label)-1):
                f.write(str(reg_label[i])+",")
            f.write(str(reg_label[-1])+"\n")

    def pack_classification_labels(self):
        xd,yd,yawd,curv,path_ahead,stopline_x,stopline_y,stopline_yaw,(state,next,action,dist,_,_,_) = self.curr_data
        #4 states, 4 next states, 7 signs
        class_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if state == 'road':
            class_data[0] = 1
        if state == 'intersection':
            class_data[1] = 1
        if state == 'roundabout':
            class_data[2] = 1
        if state == 'junction':
            class_data[3] = 1
        if next == 'road':
            class_data[4] = 1
        if next == 'intersection':
            class_data[5] = 1
        if next == 'roundabout':
            class_data[6] = 1
        if next == 'junction':
            class_data[7] = 1

        #append to file
        with open(self.folder+"/classification_labels.csv", "a") as f:
            for i in range(len(class_data)-1):
                f.write(str(class_data[i])+",")
            f.write(str(class_data[-1])+"\n")

    def get_action_vector(self, action):
        action_vec = [0,0,0,0]
        if action == "straight":
            action_vec[0] = 1
        elif action == "left":
            action_vec[1] = 1
        elif action == "right":
            action_vec[2] = 1
        elif action == "continue":
            action_vec[3] = 1
        return np.array(action_vec).reshape(1,4)

    def get_random_noise(self, std=np.deg2rad(20)):
        if self.cnt == self.noise_reset:
            self.cnt = 0
            self.noise = np.random.normal(0, std)
            self.noise_reset = np.random.randint(NOISE_RESET_MEAN - NOISE_RESET_STD, NOISE_RESET_MEAN + NOISE_RESET_STD)
            # print(f"noise: {self.noise}")
        self.cnt += 1
        return self.noise

    def save_data(self, frame, folder):
        self.data_cnt += 1
        cv.imwrite(folder+f"/img_{self.data_cnt}.png", frame)
        self.pack_input_data()
        self.pack_regression_labels()
        self.pack_classification_labels()
    
