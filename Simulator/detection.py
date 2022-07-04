#!/usr/bin/python3

import numpy as np
import cv2 as cv
import pickle , collections
from time import time, sleep
from names_and_constants import *

from helper_functions import *
from stopline import StopLine, detect_angle

LANE_KEEPER_PATH = "Simulator/models/lane_keeper_small.onnx"
DISTANCE_POINT_AHEAD = 0.35
CAR_LENGTH = 0.4

LANE_KEEPER_AHEAD_PATH = "Simulator/models/lane_keeper_ahead.onnx"
DISTANCE_POINT_AHEAD_AHEAD = 0.6

STOP_LINE_ESTIMATOR_PATH = "Simulator/models/stop_line_estimator.onnx"

STOP_LINE_ESTIMATOR_ADV_PATH = "Simulator/models/stop_line_estimator_advanced.onnx"
PREDICTION_OFFSET = -0.08

KERNEL_TYPE_SIGNS = 'linear'
NUM_CLUSTERS_SIGNS = 100
NO_SIGN = 'NO_sign'
signs_dict = {
"0": "stop", 
"1": "closed_road", 
"2": "park", 
"3": "cross_walk", 
"4": "one_way", 
"5": "hw_enter", 
"6": "hw_exit", 
"7": "priority", 
"8": "roundabout",
"9": NO_SIGN}
MAP_DICT_NAMES = [4,1,0,7,8,3,2,6,5,9]


#Obstacle classifier
NUM_CLUSTERS_OBS = 1200
KERNEL_TYPE_OBS = 'linear'
obstacles_dict = {
    "0": "car",
    "1": "pedestrian",
    "2": "roadblock",
}

distance_dict = {
    '20': [(94, 40),(227, 140)],
    '30': [(103, 25),(217, 111)],
    '40': [(116, 23),(205, 90)],
    '50': [(120, 20),(200, 80)]
}

class Detection:

    #init 
    def __init__(self) -> None:

        #lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0

        #speed challenge
        self.lane_keeper_ahead = cv.dnn.readNetFromONNX(LANE_KEEPER_AHEAD_PATH)
        self.lane_ahead_cnt = 0
        self.avg_lane_ahead_detection_time = 0

        #stop line detection
        self.stop_line_estimator = cv.dnn.readNetFromONNX(STOP_LINE_ESTIMATOR_PATH)
        self.est_dist_to_stop_line = 1.0
        self.avg_stop_line_detection_time = 0
        self.stop_line_cnt = 0

        #stop line detection advanced
        self.stop_line_estimator_adv = cv.dnn.readNetFromONNX(STOP_LINE_ESTIMATOR_ADV_PATH)
        self.est_dist_to_stop_line_adv = 1.0
        self.avg_stop_line_detection_adv_time = 0
        self.stop_line_adv_cnt = 0
 

    def detect_lane(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2), 
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32,32) #match with trainer
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):,:] #/3
        #keep the bottom 2/3 of the image
        #blur
        # frame = cv.blur(frame, (15,15), 0) #worse than blur after 11,11 #with 15,15 is 1ms
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)


        # # add noise 1.5 ms 
        # std = 50
        # # std = np.random.randint(1, std)
        # noisem = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.subtract(frame, noisem)
        # noisep = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.add(frame, noisep)
        
        
        images = frame

        if faster:
            blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False) 
        else:
            frame_flip = cv.flip(frame, 1) 
            #stack the 2 images
            images = np.stack((frame, frame_flip), axis=0) 
            blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False) 
        # assert blob.shape == (2, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.lane_keeper.setInput(blob)
        out = -self.lane_keeper.forward() #### NOTE: MINUS SIGN IF OLD NET
        output = out[0]
        output_flipped = out[1] if not faster else None

        e2 = output[0]
        e3 = output[1]

        if not faster:
            e2_flipped = output_flipped[0]
            e3_flipped = output_flipped[1]

            e2 = (e2 - e2_flipped) / 2
            e3 = (e3 - e3_flipped) / 2

        #calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d+0.2, np.sin(e3)*d])
        print(f"est_point_ahead: {est_point_ahead}")
        
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        if show_ROI:
            #edge
            # frame = cv.Canny(frame, 150, 180)
            cv.imshow('lane_detection', frame)
            cv.waitKey(1)
        return e2, e3, est_point_ahead

    def detect_lane_ahead(self, frame, show_ROI=True, faster=False):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2), 
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32,32) #match with trainer
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):,:] #/3
        #keep the bottom 2/3 of the image
        #blur
        # frame = cv.blur(frame, (15,15), 0) #worse than blur after 11,11 #with 15,15 is 1ms
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        # # add noise 1.5 ms 
        # std = 50
        # # std = np.random.randint(1, std)
        # noisem = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.subtract(frame, noisem)
        # noisep = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.add(frame, noisep)
        
        images = frame

        if faster:
            blob = cv.dnn.blobFromImage(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False) 
        else:
            frame_flip = cv.flip(frame, 1) 
            #stack the 2 images
            images = np.stack((frame, frame_flip), axis=0) 
            blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False) 
        # assert blob.shape == (2, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.lane_keeper_ahead.setInput(blob)
        out = self.lane_keeper_ahead.forward() #### NOTE: MINUS SIGN IF OLD NET
        output = out[0]
        output_flipped = out[1] if not faster else None

        # e2 = output[0]
        e3 = output[0]

        if not faster:
            # e2_flipped = output_flipped[0]
            e3_flipped = output_flipped[0]

            # e2 = (e2 - e2_flipped) / 2.0
            e3 = (e3 - e3_flipped) / 2.0

        #calculate estimated of thr point ahead to get visual feedback
        d = DISTANCE_POINT_AHEAD_AHEAD
        est_point_ahead = np.array([np.cos(e3)*d, np.sin(e3)*d])
        
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        if show_ROI:
            #edge
            # frame = cv.Canny(frame, 150, 180)
            cv.imshow('lane_detection', frame)
            cv.waitKey(1)
        return e3, est_point_ahead

    def detect_stop_line(self, frame, show_ROI=True):
        """
        Estimates the distance to the next stop line
        """
        start_time = time()
        IMG_SIZE = (32,32) #match with trainer
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[0:int(frame.shape[0]*(2/5)):,:]#frame = frame[int(frame.shape[0]*(2/5)):,:]
        #keep the bottom 2/3 of the image
        #blur
        # frame = cv.blur(frame, (15,15), 0) #worse than blur after 11,11 #with 15,15 is 1ms
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (5,5), 0)#frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        # # # add noise 1.5 ms 
        # std = 50
        # # std = np.random.randint(1, std)
        # noisem = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.subtract(frame, noisem)
        # noisep = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.add(frame, noisep)

        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        # assert blob.shape == (1, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.stop_line_estimator.setInput(blob)
        output = self.stop_line_estimator.forward()
        dist = output[0][0]

        self.est_dist_to_stop_line = dist

        # return e2, e3, inv_dist, curv, est_point_ahead
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        stop_line_detection_time = 1000*(time()-start_time)
        self.avg_stop_line_detection_time = (self.avg_stop_line_detection_time*self.lane_cnt + stop_line_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        if show_ROI:
            cv.imshow('stop_line_detection', frame)
            cv.waitKey(1)
        print(f"stop_line_detection dist: {dist:.2f}, in {stop_line_detection_time:.2f} ms")
        return dist

    def detect_stop_line2(self, frame, show_ROI=True):
        """
        Estimates the distance to the next stop line
        """
        start_time = time()
        IMG_SIZE = (32,32) #match with trainer
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]*(2/5)):,:]
        #keep the bottom 2/3 of the image
        #blur
        frame = cv.blur(frame, (9,9), 0) #worse than blur after 11,11 #with 15,15 is 1ms
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        # # # add noise 1.5 ms 
        # std = 50
        # # std = np.random.randint(1, std)
        # noisem = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.subtract(frame, noisem)
        # noisep = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        # frame = cv.add(frame, noisep)

        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        # assert blob.shape == (1, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.stop_line_estimator_adv.setInput(blob)
        output = self.stop_line_estimator_adv.forward()
        stopline_x = dist = output[0][0] + PREDICTION_OFFSET
        stopline_y = output[0][1]
        stopline_angle = output[0][2]
        self.est_dist_to_stop_line = dist

        # return e2, e3, inv_dist, curv, est_point_ahead
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        stop_line_detection_time = 1000*(time()-start_time)
        self.avg_stop_line_detection_time = (self.avg_stop_line_detection_time*self.lane_cnt + stop_line_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        if show_ROI:
            cv.imshow('stop_line_detection', frame)
            cv.imwrite(f'sd/sd_{int(time()*1000)}.png', frame)
            cv.waitKey(1)
        print(f"stop_line_detection dist: {dist:.2f}, in {stop_line_detection_time:.2f} ms")
        return stopline_x, stopline_y, stopline_angle

    #helper functions
    def automatic_brightness_and_contrast(image, clip_hist_percent=1):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        hist = cv.calcHist([gray],[0],None,[256],[0,256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        
        auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result)

    #into detection class
    def ImageHistogram(kmeans, descriptor_list, no_clusters):
        """
        Compute the histogram of occurrences of the visual words in the input image
        """
        im_hist = np.zeros(no_clusters)
        
        # feature is the descriptor of a single keypoint
        for feature in descriptor_list:

            feature = feature.reshape(1, 32)
            idx = kmeans.predict(feature)
            im_hist[idx] += 1
        return im_hist

    #into detection class
    def draw_ROI(frame, TL, BR, show_rect = False, prediction = None, conf= None, show_prediction = False):
        # Blue color in BGR
        if show_rect: 
            image = frame.copy()
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv.rectangle(image, TL, BR, color = (255, 0, 0), thickness = 2)
            cv.imshow("Frame preview", image)
            cv.waitKey(1)
        if show_rect and show_prediction:
            image = frame.copy()
            # Draw a rectangle with blue line borders of thickness of 2 px
            image = cv.rectangle(image, TL, BR, color = (255, 0, 0), thickness = 2)
            cv.putText(img=image, text= prediction + ' ' + str(conf) + '%', org=(TL[0]-100, TL[1]), 
                fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
            cv.imshow("Frame preview", image)
            cv.waitKey(1)

    def detect_yaw_stopline(self, frame, show_ROI=False):
        return detect_angle(original_frame=frame, plot=show_ROI)

