#!/usr/bin/python3

from cgi import print_environ
from importlib.resources import path
import numpy as np
import cv2 as cv
import os

from helper_functions import *

IMG_SIZE = (128,64)

class Controller():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0, cm_ahead=35, folder='training_imgs', 
                    lane_keeper_path="models/lane_keeper_small.onnx",
                    training=True, noise_std=np.deg2rad(20)):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0
        self.ff = ff

        self.cm_ahead = cm_ahead
        self.L = 0.4
        self.d = self.cm_ahead*0.01 #meters ahead

        self.cnt = 0
        self.noise = 0.0
        self.data_cnt = 0

        self.noise_std = noise_std

        #load neural network
        # self.detector =  cv.dnn.readNetFromONNX(detector_path) 
        # self.feature_extractor = cv.dnn.readNetFromONNX(feature_extractor_path)
        self.lane_keeper = cv.dnn.readNetFromONNX(lane_keeper_path)

        #training
        self.folder = folder
        self.curr_data = []
        self.input_data = []
        self.regression_labels = []
        self.classification_labels = []

        if training:
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
        
    def get_control(self, car, path_ahead, vd, curvature_ahead):

        #add e2 (lateral error), not used in the controller
        ex = path_ahead[0][0] - car.x_true #x error
        ey = path_ahead[0][1] - car.y_true #y error
        self.e2 = -ex * np.sin(car.yaw) - ey * np.cos(car.yaw) #y error in the body frame

        #get point ahead
        assert len(path_ahead) > 0, "path_ahead is empty"
        point_ahead = path_ahead[min(self.cm_ahead, len(path_ahead)-1),:]
        #translate to car coordinates
        point_ahead = to_car_frame(point_ahead, car, return_size=2)

        #pure pursuit control 
        alpha = np.arctan2(point_ahead[1], point_ahead[0]+self.L/2) 
        self.e3 = alpha
        delta = np.arctan((2*self.L*np.sin(alpha))/(self.k3*self.d))

        output_angle = delta
        output_speed = vd - self.k1 * self.e1
        output_angle = output_angle + self.get_random_noise(std=self.noise_std) #add noise for training
        return output_speed, output_angle, point_ahead

    def get_nn_control(self, frame, vd, action):
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #keep the bottom 2/3 of the image
        frame = frame[int(frame.shape[0]/3):,:]
        # #paint gray the top section of the image
        # frame[:int(IMG_SIZE[1]/3),:] = 127
        frame = cv.resize(frame, IMG_SIZE)
        #blur
        frame = cv.GaussianBlur(frame, (5,5), 0)
        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        assert blob.shape == (1, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.lane_keeper.setInput(blob)
        output = self.lane_keeper.forward()
        # print(f'output: {output.shape}')
        output = output[0]

        e2,e3,dist,curv = self.unpack_network_output(output)

        self.e2 = e2
        self.e3 = e3
        alpha = e3
        delta = np.arctan((2*self.L*np.sin(alpha))/(self.k3*self.d))

        # output_angle = self.ff*curv - self.k2 * e2 - self.k3 * e3
        output_angle = delta - self.k2 * e2
        output_speed = vd - self.k1 * self.e1

        #calculate estimated of thr point ahead to get visual feedback
        est_point_ahead = np.array([np.cos(alpha)*self.d, np.sin(alpha)*self.d])

        net_out = (e2,e3,dist,curv)
        return output_speed, output_angle, net_out, est_point_ahead
    
    def pack_input_data(self):
        data = [0,0,0,0]
        xd,yd,yawd,curv,path_ahead,(state,next,action,dist,_,_,_) = self.curr_data
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
        xd,yd,yawd,curv,path_ahead,(state,next,action,dist,_,_,_) = self.curr_data
        if dist is None:
            dist = 10
        # print(f'bounding box: {bounding_box}')
        reg_label = [self.e2, self.e3, dist, curv]
        np_arr = np.array(reg_label)
        # print(f'reg_label: {np_arr}')
        #append to file
        with open(self.folder+"/regression_labels.csv", "a") as f:
            for i in range(len(reg_label)-1):
                f.write(str(reg_label[i])+",")
            f.write(str(reg_label[-1])+"\n")
    
    def unpack_network_output(self, output, bb_const=1000.0):
        #first 4 regression + 7 classification
        #4 regression [0:4]: 
        e2 = output[0]
        e3 = output[1]
        dist = output[2]
        curv = output[3]
        # # 7 classification: [4:11]
        # states = ['road', 'intersection', 'roundabout', 'junction']
        # state_vec = output[8:12]
        # state_index = np.argmax(state_vec)
        # state = states[state_index]
        # next_vec = output[12:16]
        # next_index = np.argmax(next_vec)
        # next = states[next_index]
        return e2,e3,dist,curv
    

    def pack_classification_labels(self):
        xd,yd,yawd,curv,path_ahead,(state,next,action,dist,_,_,_) = self.curr_data
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

    def get_random_noise(self, std=np.deg2rad(20), reset=10):
        if self.cnt == reset:
            self.cnt = 0
            self.noise = np.random.normal(0, std)
            # print(f"noise: {self.noise}")
        self.cnt += 1
        return self.noise

    def save_data(self, frame, folder):
        self.data_cnt += 1
        cv.imwrite(folder+f"/img_{self.data_cnt}.png", frame)
        self.pack_input_data()
        self.pack_regression_labels()
        self.pack_classification_labels()
    
