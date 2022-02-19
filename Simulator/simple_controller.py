#!/usr/bin/python3

from dis import dis
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline
import os

from sklearn.semi_supervised import LabelSpreading

IMG_SIZE = (320, 240)


class SimpleController():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0, folder='training_imgs', 
                    detector_path="detector.onnx", feature_extractor_path = "feature_extractor.onnx",
                    training=True, noise_std=np.deg2rad(20)):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0
        self.ff = ff

        self.cnt = 0
        self.noise = 0.0
        self.data_cnt = 0

        self.noise_std = noise_std

        #load neural network
        self.detector =  cv.dnn.readNetFromONNX(detector_path) 
        self.feature_extractor = cv.dnn.readNetFromONNX(feature_extractor_path)

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
            #delet all images in the folder
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    os.remove(os.path.join(folder, file))
        
    def get_control(self, x, y, yaw, xd, yd, yawd, vd, curvature_ahead):
        e_yaw = np.arctan2(np.sin(yawd-yaw), np.cos(yawd-yaw)) # yaw error, in radians, not a simple difference (overflow at +-180)
        ex = xd - x #x error
        ey = yd - y #y error
        #similar to UGV trajectory tracking but frame of reference is left-hand
        self.e1 = -ex * np.cos(yaw) + ey * np.sin(yaw) #x error in the body frame
        self.e2 = -ex * np.sin(yaw) - ey * np.cos(yaw) #y error in the body frame
        self.e3 = e_yaw #yaw error
        output_angle = self.ff*curvature_ahead - self.k2 * self.e2 - self.k3 * self.e3
        output_speed = vd - self.k1 * self.e1
        output_angle = output_angle + self.get_random_noise(std=self.noise_std) #add noise for training
        return output_speed, output_angle

    def get_nn_control(self, frame, vd, action):
        cv.resize(frame, IMG_SIZE)
        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE,(0, 0, 0), swapRB=True, crop=False)
        self.feature_extractor.setInput(blob)
        features = self.feature_extractor.forward()
        action_vec = self.get_action_vector(action)
        # print(f'features: {features.shape}, action_vec: {action_vec.shape}')
        input = np.concatenate((features, action_vec), axis=1)
        self.detector.setInput(input)
        output = self.detector.forward()
        # print(f'output: {output.shape}')
        output = output[0]

        (state,next,sign,dist,curv,bounding_box) = self.unpack_network_output(output)

        net_out = (state,next,sign,dist,curv,bounding_box)

        curvature_ahead = curv ####### <------------------------------------------------------------
        output_angle = self.ff*curvature_ahead - self.k2 * self.e2 - self.k3 * self.e3
        output_speed = vd - self.k1 * self.e1
        return output_speed, output_angle, net_out
    
    def pack_input_data(self):
        data = [0,0,0,0]
        xd,yd,yawd,curv,path_ahead,(state,next,action,dist,_,_,_),coeffs,bounding_box,sign = self.curr_data
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
        
        xd,yd,yawd,curv,path_ahead,(state,next,action,dist,_,_,_),coeffs,bounding_box,sign = self.curr_data
        #reshape coeffs to be a 1D array and convert to list
        # coeffs = coeffs.ravel().tolist()
        # c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16 = coeffs
        if dist is None:
            dist = 10
        # print(f'bounding box: {bounding_box}')
        reg_label = [self.e2, self.e3, dist, curv, bounding_box[0]/bb_const, bounding_box[1]/bb_const, bounding_box[2]/bb_const, bounding_box[3]/bb_const]
        np_arr = np.array(reg_label)
        # print(f'reg_label: {np_arr}')
        #append to file
        with open(self.folder+"/regression_labels.csv", "a") as f:
            for i in range(len(reg_label)-1):
                f.write(str(reg_label[i])+",")
            f.write(str(reg_label[-1])+"\n")
    
    def unpack_network_output(self, output,bb_const=1000.0):
        #first 22 regression + 13 classification
        #22 regression: 
        self.e2 = output[0]
        self.e3 = output[1]
        dist = output[2]
        curv = output[3]
        # coeffs = output[4:20] #round(bb_const*
        bounding_box = (round(bb_const*output[20-16]), round(bb_const*output[21-16]), round(bb_const*output[22-16]), round(bb_const*output[23-16]))
        #13 classification:
        states = ['road', 'intersection', 'roundabout']
        state_vec = output[22-16+2:25-16+2]
        state_index = np.argmax(state_vec)
        state = states[state_index]
        next_vec = output[25-16+2:28-16+2]
        next_index = np.argmax(next_vec)
        next = states[next_index]
        sign_vec = output[28-16+2:]
        signs = ['no_sign', 'cross_walk', 'highway', 'park', 'stop', 'preference_road', 'roundabout']
        sign_index = np.argmax(sign_vec)
        sign = signs[sign_index]
        return (state,next,sign,dist,curv,bounding_box)

    

    def pack_classification_labels(self):
        xd,yd,yawd,curv,path_ahead,(state,next,action,dist,_,_,_),coeffs,bounding_box,sign = self.curr_data
        #3 states, 3 next states, 7 signs
        class_data = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        if state == 'road':
            class_data[0] = 1
        if state == 'intersection':
            class_data[1] = 1
        if state == 'roundabout':
            class_data[2] = 1
        if next == 'road':
            class_data[3] = 1
        if next == 'intersection':
            class_data[4] = 1
        if next == 'roundabout':
            class_data[5] = 1
        #TODO add signs
        signs = ['no_sign', 'cross_walk', 'highway', 'park', 'stop', 'preference_road', 'roundabout']
        if sign == signs[0]:
            class_data[6] = 1
        elif sign == signs[1]:
            class_data[7] = 1
        elif sign == signs[2]:
            class_data[8] = 1
        elif sign == signs[3]:
            class_data[9] = 1
        elif sign == signs[4]:
            class_data[10] = 1
        elif sign == signs[5]:
            class_data[11] = 1
        elif sign == signs[6]:
            class_data[12] = 1
        
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

    def get_random_noise(self, std=np.deg2rad(20), reset=7):
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
    
