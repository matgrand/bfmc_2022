#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline
import os

IMG_SIZE = (320, 240)


class SimpleController():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0, folder='training_imgs', 
                    nn_model_path="model_test.onnx", training=True):
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

        #load neural network
        self.net =  cv.dnn.readNetFromONNX(nn_model_path) 

        
        if training:
            #clear and create file 
            with open(folder+"/labels.csv", "w") as f:
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
        output_angle = output_angle + self.get_random_noise() #add noise for training
        return output_speed, output_angle

    def get_nn_control(self, frame, vd):
        cv.resize(frame, IMG_SIZE)
        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE,(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        self.e2 = preds[0][0]
        self.e3 = preds[0][1]
        curvature_ahead = 0.0 ####### <------------------------------------------------------------
        output_angle = self.ff*curvature_ahead - self.k2 * self.e2 - self.k3 * self.e3
        output_speed = vd - self.k1 * self.e1
        output_angle = output_angle
        return output_speed, output_angle


    def get_random_noise(self, std=np.deg2rad(20), reset=7):
        if self.cnt == reset:
            self.cnt = 0
            self.noise = np.random.normal(0, std)
            print(f"noise: {self.noise}")
        self.cnt += 1
        return self.noise

    def save_data(self, frame, folder):
        self.data_cnt += 1
        cv.imwrite(folder+f"/img_{self.data_cnt}.png", frame)
        #append to file 
        with open(folder+"/labels.csv", "a") as f:
            f.write(f"{self.e2}, {self.e3}\n")
