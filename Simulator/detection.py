#!/usr/bin/python3

from turtle import width
import numpy as np
import cv2 as cv
import os
from time import time, sleep

from helper_functions import *

LANE_KEEPER_PATH = "models/lane_keeper_small.onnx"
DISTANCE_POINT_AHEAD = 0.35
CAR_LENGTH = 0.4

STOP_LINE_ESTIMATOR_PATH = "models/stop_line_estimator.onnx"

LOCAL_PATH_ESTIMATOR_PATH = "models/local_path_estimator.onnx"
NUM_POINTS_AHEAD = 5
DISTANCE_BETWEEN_POINTS = 0.2

TRAFFICLIGHT_CLASSIFIER_PATH = 'models/trafficlight_classifier_small.onnx'
TRAFFIC_LIGHT_NAMES = ['traffic_light', 'NO_traffic_light']

SIGN_CLASSIFIER_PATH = 'models/sign_classifier.onnx'
# SIGN_NAMES = ['park', 'closed_road', 'highway_exit', 'highway_enter', 'stop', 'roundabout', 'priority', 'cross_walk', 'one_way', 'NO_sign']
SIGN_NAMES = ['park', 'closed_road', 'highway_exit', 'highway_enter', 'stop', 'roundabout', 'priority', 'cross_walk', 'one_way', 'traffic_light', 'NO_sign']

OBSTACLE_CLASSIFIER_PATH = 'models/pedestrian_classifier_small.onnx'
OBSTACLE_NAMES = ['pedestrian', 'roadblock', 'NO_obstacle'] #add cars

class Detection:

    #init 
    def __init__(self) -> None:

        #lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0

        #stop line detection
        self.stop_line_estimator = cv.dnn.readNetFromONNX(STOP_LINE_ESTIMATOR_PATH)
        self.est_dist_to_stop_line = 1.0
        self.avg_stop_line_detection_time = 0
        self.stop_line_cnt = 0

        #local path estimation
        self.local_path_estimator = cv.dnn.readNetFromONNX(LOCAL_PATH_ESTIMATOR_PATH)
        self.avg_local_path_detection_time = 0

        #traffic light classifier #trafficlight is abbreviated in tl
        self.tl_classifier =  cv.dnn.readNetFromONNX(TRAFFICLIGHT_CLASSIFIER_PATH)
        self.tl_names = TRAFFIC_LIGHT_NAMES
        self.prev_tl_conf = 0.0

        #sign classifier
        self.sign_classifier =  cv.dnn.readNetFromONNX(SIGN_CLASSIFIER_PATH)
        self.sign_names = SIGN_NAMES
        self.last_sign_detected = self.sign_names[-1]
        self.last_sign_conf = 0.0
        self.avg_sign_detection_time = 0.0
        self.sign_detection_count = 0.0

        #test frontal obstacles classifications
        self.obstacle_classifier = cv.dnn.readNetFromONNX(OBSTACLE_CLASSIFIER_PATH)
        self.front_obstacle_names = OBSTACLE_NAMES
        self.last_obstacle_detected = self.front_obstacle_names[-1]
        self.last_obstacle_conf = 0.0

    def detect_lane(self, frame, show_ROI=True, faster=True):
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
        return e2, e3, est_point_ahead

    def detect_stop_line(self, frame, show_ROI=True):
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
        # frame = cv.blur(frame, (15,15), 0) #worse than blur after 11,11 #with 15,15 is 1ms
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

    def estimate_local_path(self, frame):
        """
        Estimates the local path, in particular it estimtes a sequence of points at a fixed
        distance between each other of DISTANCE_BETWEEN_POINTS [m]
        """
        start_time = time()
        IMG_SIZE = (32,32)
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #keep the bottom 2/3 of the image
        frame = frame[int(frame.shape[0]/4):,:] #/3
        #blur
        # frame = cv.blur(frame, (15,15), 0) #worse than blur after 11,11 #with 15,15 is 1ms
        # frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        # frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)
        frame = cv.blur(frame, (2,2), 0)  #7,7 both is best

        # add noise 1.5 ms 
        std = 50
        # std = np.random.randint(1, std)
        noisem = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        frame = cv.subtract(frame, noisem)
        noisep = np.random.randint(0, std, frame.shape, dtype=np.uint8)
        frame = cv.add(frame, noisep)

        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        # assert blob.shape == (1, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.local_path_estimator.setInput(blob)
        output = self.local_path_estimator.forward()[0]

        #get the yaws from the network
        yaws = [output[i] for i in range(NUM_POINTS_AHEAD)]
        #calculate the vectors
        points = np.array([np.array([DISTANCE_BETWEEN_POINTS*np.cos(yaw), DISTANCE_BETWEEN_POINTS*np.sin(yaw)]) for yaw in yaws])
        #create the sequence
        for i in range(NUM_POINTS_AHEAD-1):
            points[i+1] = points[i] + points[i+1]
        
        yaws = np.array(yaws)
        return points, yaws
            
    def classify_traffic_light(self, frame, conf_threshold=0.8, show_ROI=False):
        SIZE = (32, 32)
        # ROI = [30,260,-100,640] #if img size is 480x640
        ROI = [30//2,260//2,-100//2,640//2] #if img size is 240x320
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

    ## SIGN DETECTION
    def tile_image(self, image, x,y,w,h, rows, cols, tile_widths, return_size=(32,32), channels=3):
        assert image.shape[0] >= y + h, f'Image height is {image.shape[0]} but y is {y} and h is {h}'
        assert image.shape[1] >= x + w, f'Image width is {image.shape[1]} but x is {x} and w is {w}'
        assert [ws < w for ws in tile_widths], f'Tile width is {tile_widths} but w is {w}'
        #check x,y,w,h,rows,cols,tile_width are all ints
        assert isinstance(x, int), f'x is {x}'
        assert isinstance(y, int), f'y is {y}'
        assert isinstance(w, int), f'w is {w}'
        assert isinstance(h, int), f'h is {h}'
        assert isinstance(rows, int), f'rows is {rows}'
        assert isinstance(cols, int), f'cols is {cols}'
        num_scales = len(tile_widths)
        idx = 0
        img = image[y:y+h, x:x+w].copy()
        if channels == 3:
            imgs = np.zeros((num_scales*rows*cols, return_size[0], return_size[1], channels), dtype=np.uint8)
        elif channels == 1:
            imgs = np.zeros((num_scales*rows*cols, return_size[0], return_size[1], channels), dtype=np.uint8)
        else: 
            raise ValueError(f'return_size must be 2 or 3, but is {return_size}')
        # centers = np.stack([centers_x, centers_y], axis=1)
        centers = []
        widths_idxs = []
        for s, i_s in enumerate(range(num_scales)):
            centers_x = np.linspace(int(tile_widths[s]/2), w-int(tile_widths[s]/2), cols, dtype=int)
            centers_y = np.linspace(int(tile_widths[s]/2), h-int(tile_widths[s]/2), rows, dtype=int)
            for i in range(rows):
                for j in range(cols):
                    # img = image[y:y+h, x:x+w]
                    im = img[centers_y[i]-tile_widths[s]//2:centers_y[i]+tile_widths[s]//2, centers_x[j]-tile_widths[s]//2:centers_x[j]+tile_widths[s]//2].copy()
                    
                    # im = cv.resize(im, (2*return_size[0], 2*return_size[1]))
                    # im = cv.blur(im, (3,3))
                    # im = cv.Canny(im, 100, 200)
                    im = cv.resize(im, return_size)
                    # im = cv.blur(im, (2,2))
                    #bgr2hsv
                    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

                    imgs[idx] = im
                    centers.append([x+centers_x[j], y+centers_y[i]])
                    widths_idxs.append(i_s)
                    idx += 1
        return imgs, centers, widths_idxs
                
    def detect_sign(self, frame, show_ROI=False):
        """
        Sign classifiier:
        takes the whole frame as input and returns the sign name and classification
        confidence. If the network is not confident enough, it returns None sign name and 0.0 confidence
        """
        start_time = time()
        #test sign classifier
        SIZE = (16, 16)
        CHANNELS = 3
        ROWS = 6 #4
        COLS = 12 #8
        TILE_WIDTHS = [76//2]#[64//2, 76//2, 84//2]#[76//2] #assuming a frame is 640x480//320x240
        ROI_X = 440//2
        ROI_Y = 10//2
        ROI_WIDTH = 200//2
        ROI_HEIGHT = 140//2
        TOT_TILES = ROWS*COLS
        VOTES_MAJORITY = 10 #1
        VOTES_ADVANTAGE_THRESHOLD = 2
        CONFIDENCE_THRESHOLD = 0.9

        # preprocessing
        frame_cp = frame.copy()
        imgs, centers, widths_idxs = self.tile_image(frame_cp, ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, ROWS, COLS, TILE_WIDTHS, return_size=SIZE, channels=CHANNELS)

        if show_ROI:
            canvas = frame.copy()
            canvas = cv.rectangle(canvas, (ROI_X, ROI_Y), (ROI_X+ROI_WIDTH, ROI_Y+ROI_HEIGHT), (0,255,0), 2)

        blob = cv.dnn.blobFromImages(imgs, 1.0, SIZE, 0)
        self.sign_classifier.setInput(blob)
        preds = self.sign_classifier.forward()

        votes = np.zeros(len(SIGN_NAMES))
        width_votes =  np.zeros(len(TILE_WIDTHS))
        box_centers = np.zeros((len(SIGN_NAMES), 2))
        for i in range(TOT_TILES):
            soft_preds = my_softmax(preds[i])
            sign_index = np.argmax(preds[i])
            if soft_preds[sign_index] > CONFIDENCE_THRESHOLD:
                predicted_sign = self.sign_names[sign_index]
                if predicted_sign != self.sign_names[-1]: #avoid no_sign
                    box_centers[sign_index] = (box_centers[sign_index]*votes[sign_index] + centers[i]) / (votes[sign_index] + 1.0)
                    votes[sign_index] += 1
                    width_votes[widths_idxs[i]] += 1

        winner = np.argmax(votes)
        tot_votes = np.sum(votes)
        winning_confidence = float(votes[winner]/max(tot_votes, VOTES_MAJORITY))
        width_winner_idx = np.argmax(width_votes)
        final_width = TILE_WIDTHS[widths_idxs[width_winner_idx]]
        final_box_center = box_centers[winner].astype(int)
        if winning_confidence > 0.8:
            if show_ROI:
                canvas = cv.rectangle(canvas, (final_box_center[0]-final_width//2, final_box_center[1]-final_width//2), (final_box_center[0]+final_width//2, final_box_center[1]+final_width//2), (0,255,0), 3)
                canvas = cv.putText(canvas, self.sign_names[winner], (final_box_center[0]-final_width//2, final_box_center[1]-final_width//2), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
                cv.imshow('Sign detection', canvas)   
                cv.waitKey(1)
            return self.sign_names[winner], winning_confidence, final_box_center, final_width
            
        sign_detection_time = 1000*(time() - start_time)
        self.avg_sign_detection_time = (self.avg_sign_detection_time*self.sign_detection_count + sign_detection_time) / (self.sign_detection_count + 1)
        self.sign_detection_count += 1

        # print(f'{self.sign_names[winner]} detected, conf: {winning_confidence*100:.0f}, votes: {votes[winner]}/{tot_votes}')
        if show_ROI:
            cv.imshow('Sign detection', canvas)   
            cv.waitKey(1)
        return None, winning_confidence, final_box_center, final_width

    def classify_frontal_obstacle(self, frame, conf_threshold=0.5, show_ROI=False):
        """
        Obstacle classifier:
        takes the whole frame as input and returns the obstacle name and classification
        confidence. If the network is not confident enough, it returns None obstacle name and 0.0 confidence
        """
        SIZE = (32,32)
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


