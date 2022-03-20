#!/usr/bin/python3

import numpy as np
import cv2 as cv
import os
from Simulator.main_simulator import IMG_SIZE

from helper_functions import *

class Detection:

    #init 
    def __init__(self) -> None:
        #traffic light classifier #trafficlight is abbreviated in tl
        self.tl_classifier =  cv.dnn.readNetFromONNX('models/trafficlight_classifier_small.onnx')
        self.tl_names = ['traffic_light', 'NO_traffic_light']  
        self.prev_tl_conf = 0.0

        #sign classifier
        self.sign_classifier =  cv.dnn.readNetFromONNX('models/sign_classifier_small.onnx')
        self.sign_names = ['park', 'closed_road', 'highway_exit', 'highway_enter', 'stop', 'roundabout', 'priority', 'cross_walk', 'one_way', 'NO_sign']
        self.last_sign_detected = self.sign_names[-1]
        self.last_sign_conf = 0.0

        #test frontal obstacles classifications
        self.obstacle_classifier = cv.dnn.readNetFromONNX('models/pedestrian_classifier_small.onnx')
        self.front_obstacle_names = ['pedestrian', 'roadblock', 'NO_obstacle'] #add cars
        self.last_obstacle_detected = self.front_obstacle_names[-1]
        self.last_obstacle_conf = 0.0

    def classify_traffic_light(self, frame, conf_threshold=0.8, show_ROI=False):
        SIZE = (32, 32)
        ROI = [30,260,-100,640]
        trafficlight_roi = frame[ROI[0]:ROI[1], ROI[2]:ROI[3]]
        trafficlight_roi = cv.cvtColor(trafficlight_roi, cv.COLOR_BGR2GRAY)
        # trafficlight_roi = cv.equalizeHist(trafficlight_roi)
        trafficlight_roi = cv.resize(trafficlight_roi, SIZE)
        trafficlight_roi = cv.blur(trafficlight_roi, (3,3))
        # trafficlight_roi = cv.blur(trafficlight_roi, (5,5))
        if show_ROI:
            cv.imshow('trafficlight_roi', trafficlight_roi)
            cv.waitKey(1)
        blob = cv.dnn.blobFromImage(trafficlight_roi, 1.0, SIZE, 0)
        # print(blob.shape)
        self.tl_classifier.setInput(blob)
        preds = self.tl_classifier.forward()[0]
        # print(f'before softmax: {preds.shape}')
        #softmax preds
        soft_preds = my_softmax(preds)
        trafficlight_index = np.argmax(preds)
        if soft_preds[trafficlight_index] > conf_threshold:
            predicted_trafficlight = self.tl_names[trafficlight_index]
            if predicted_trafficlight != self.tl_names[-1]:     
                #find color of traffic light
                tl_color = self.find_trafficlight_color(trafficlight_roi)
                print(f'{predicted_trafficlight} {tl_color} detected, confidence: {float(soft_preds[trafficlight_index]):.2f}')
        else:
            return None, 0.0
                

    def classify_sign(self, frame, conf_threshold=0.8, show_ROI=False):
        """
        Sign classifiier:
        takes the whole frame as input and returns the sign name and classification
        confidence. If the network is not confident enough, it returns None sign name and 0.0 confidence
        """
        #test sign classifier
        SIZE = (32, 32)
        signs_roi = frame[60:160, -100:640, :]
        # signs_roi = car.cv_image[50:200, -150:, :]
        signs_roi = cv.cvtColor(signs_roi, cv.COLOR_BGR2GRAY)
        # signs_roi = cv.equalizeHist(signs_roi)
        signs_roi = cv.blur(signs_roi, (5,5))
        signs_roi = cv.resize(signs_roi, SIZE)
        if show_ROI:
            cv.imshow('signs_roi', signs_roi)
            cv.waitKey(1)
        # signs_roi = cv.blur(signs_roi, (5,5))
        blob = cv.dnn.blobFromImage(signs_roi, 1.0, SIZE, 0)
        # print(blob.shape)
        self.sign_classifier.setInput(blob)
        preds = self.sign_classifier.forward()[0]
        # print(f'before softmax: {preds.shape}')
        #softmax preds
        soft_preds = my_softmax(preds)
        sign_index = np.argmax(preds)
        sign_conf = soft_preds[sign_index]
        if soft_preds[sign_index] > conf_threshold:
            predicted_sign = self.sign_names[sign_index]
            if predicted_sign != self.sign_names[-1]:
                print(f'Sign: {predicted_sign} detected, confidence: {float(soft_preds[sign_index]):.2f}')
                self.last_sign_conf = sign_conf
                self.last_sign_detected = predicted_sign
                return predicted_sign, sign_conf
        else:
            return None, 0.0

    def classify_frontal_obstacle(self, frame, conf_threshold=0.5, show_ROI=False):
        """
        Obstacle classifier:
        takes the whole frame as input and returns the obstacle name and classification
        confidence. If the network is not confident enough, it returns None obstacle name and 0.0 confidence
        """
        SIZE = (32,32)
        IMG_SIZE = (480, 640)
        ROI = [120, 360, 200, 440] #[a,b,c,d] ==> [a:b, c:d]
        front_obstacle_roi = frame[ROI[0]:ROI[1], ROI[2]:ROI[3], :]
        # signs_roi = car.cv_image[50:200, -150:, :]
        front_obstacle_roi = cv.cvtColor(front_obstacle_roi, cv.COLOR_BGR2GRAY)
        # signs_roi = cv.equalizeHist(signs_roi)
        front_obstacle_roi = cv.resize(front_obstacle_roi, SIZE)
        front_obstacle_roi = cv.blur(front_obstacle_roi, (3,3))
        if show_ROI:
            cv.imshow('front_obstacle_roi', front_obstacle_roi)
            cv.waitKey(1)
        blob = cv.dnn.blobFromImage(front_obstacle_roi, 1.0, SIZE, 0)
        # blob = cv.dnn.blobFromImage(front_obstacle_roi, 1.0, SIZE, 0.0, swapRB=True, crop=False)
        # print(blob.shape)
        self.obstacle_classifier.setInput(blob)
        preds = self.obstacle_classifier.forward()
        print(f'Obstacle preds: {preds}')
        preds = preds[0]
        # print(f'before softmax: {preds.shape}')
        #softmax preds
        soft_preds = my_softmax(preds)
        front_obstacle_index = np.argmax(preds)
        conf = soft_preds[front_obstacle_index]
        if soft_preds[front_obstacle_index] > conf_threshold:
            predicted_obstacle = self.front_obstacle_names[front_obstacle_index]
            if predicted_obstacle != self.front_obstacle_names[-1]:
                print(f'Obstacle: {predicted_obstacle} detected, confidence: {float(soft_preds[front_obstacle_index]):.2f}')
                self.last_obstacle_conf = conf
                self.last_obstacle_detected = predicted_obstacle
                return predicted_obstacle, conf
        else:
            return None, 0.0


    def find_trafficlight_color(self, trafficlight_roi, max_deviation=0.2):
        """
        takes the traffic light ROI as input and returns the color of the traffic light
        """
        #split the image in three stripes
        
        red_stripe = trafficlight_roi[:, :int(trafficlight_roi.shape[1]/3)]
        yellow_stripe = trafficlight_roi[:, int(trafficlight_roi.shape[1]/3):int(trafficlight_roi.shape[1]*2/3)]
        green_stripe = trafficlight_roi[:, int(trafficlight_roi.shape[1]*2/3):]

        #resize to 16x16 each stripe
        red_stripe = cv.resize(red_stripe, (16,16))
        yellow_stripe = cv.resize(yellow_stripe, (16,16))
        green_stripe = cv.resize(green_stripe, (16,16))

        #crop to the center of each stripe
        red_stripe = red_stripe[4:12, 4:12]
        yellow_stripe = yellow_stripe[4:12, 4:12]
        green_stripe = green_stripe[4:12, 4:12]

        #get the average of each stripe
        red_avg = np.average(red_stripe)
        yellow_avg = np.average(yellow_stripe)
        green_avg = np.average(green_stripe)

        avg_array = np.array([red_avg, yellow_avg, green_avg])
        colors = ['RED', 'YELLOW', 'GREEN', 'off']

        print(f'Red avg: {red_avg}, yellow avg: {yellow_avg}, green avg: {green_avg}')

        #get brightest color
        brightest_color_index = np.argmax(avg_array)
        brightest_color = colors[brightest_color_index]
        avg_color = np.average(avg_array)

        pixel_deviation = int(avg_color * max_deviation)

        if brightest_color < (avg_color + pixel_deviation):
            # brightest color is too similar to the other ones, tl is off
            return colors[-1]
        else:
            #one color is definitely brighter than the others
            return brightest_color


