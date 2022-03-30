#!/usr/bin/python3

from cgi import print_environ
from importlib.resources import path
import numpy as np
import cv2 as cv
import os

from helper_functions import *

IMG_SIZE = (128, 64)

class Controller():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0, cm_ahead=35, 
                    lane_keeper_path="models/lane_keeper_small.onnx"):
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

        #load neural network
        # self.detector =  cv.dnn.readNetFromONNX(detector_path) 
        # self.feature_extractor = cv.dnn.readNetFromONNX(feature_extractor_path)
        self.lane_keeper = cv.dnn.readNetFromONNX(lane_keeper_path)

        #testing
        self.last_e3 = 0.0

        #training
        self.curr_data = []
        self.input_data = []
        self.regression_labels = []
        self.classification_labels = []

    def get_nn_control(self, frame, vd):
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
        output_angle = delta - self.k2 * e2 + 0.0*curv
        output_speed = vd - self.k1 * self.e1

        #calculate estimated of thr point ahead to get visual feedback
        est_point_ahead = np.array([np.cos(alpha)*self.d, np.sin(alpha)*self.d])

        net_out = (e2,e3,dist,curv)
        return output_speed, output_angle, net_out, est_point_ahead
    
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
