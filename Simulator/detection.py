#!/usr/bin/python3

import numpy as np
import cv2 as cv
import pickle , collections
from time import time, sleep
from names_and_constants import *

from helper_functions import *
from stopline import StopLine, detect_angle

LANE_KEEPER_PATH = "models/lane_keeper_small.onnx"
DISTANCE_POINT_AHEAD = 0.35
CAR_LENGTH = 0.4

LANE_KEEPER_AHEAD_PATH = "models/lane_keeper_ahead.onnx"
DISTANCE_POINT_AHEAD_AHEAD = 0.6

STOP_LINE_ESTIMATOR_PATH = "models/stop_line_estimator.onnx"

STOP_LINE_ESTIMATOR_ADV_PATH = "models/stop_line_estimator_advanced.onnx"
PREDICTION_OFFSET = -0.08

# SIGN_CLASSIFIER_PATH = 'models/sign_classifier.onnx'

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

        #sign classifier
        self.sign_no_clusters = NUM_CLUSTERS_SIGNS 
        self.sign_kernel_type = KERNEL_TYPE_SIGNS 
        self.sign_svm_model = pickle.load(open('models/traffic_signs_models/svm_'+ self.sign_kernel_type + '_' + str(self.sign_no_clusters) + '.pkl', 'rb'))
        self.sign_kmean_model = pickle.load(open('models/traffic_signs_models/kmeans_' + self.sign_kernel_type + '_' +  str(self.sign_no_clusters) + '.pkl', 'rb'))
        self.sign_scale_model = pickle.load(open('models/traffic_signs_models/scale_'+ self.sign_kernel_type + '_' + str(self.sign_no_clusters) + '.pkl', 'rb'))
        self.sign_sift = cv.SIFT_create()
        self.sign_probs_buffer = collections.deque(maxlen=10) 
        self.sign_prediction = NO_SIGN
        self.sign_conf = 1

        #obstacle classifier
        self.no_clusters = NUM_CLUSTERS_OBS #use const
        self.kernel_type = KERNEL_TYPE_OBS #const
        self.svm_model = pickle.load(open('models/obstacle_models/svm_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.kmean_model = pickle.load(open('models/obstacle_models/kmeans_' + self.kernel_type + '_' +  str(self.no_clusters) + '.pkl', 'rb'))
        self.scale_model = pickle.load(open('models/obstacle_models/scale_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        # self.sift = cv.SIFT_create() 
        self.sift = cv.ORB_create(nfeatures=100)
        # self.obstacles_probs_buffer = collections.deque(maxlen=OBSTACLE_DETECTION_DEQUE_LENGTH) 
        self.prediction = None
        self.conf = 0        

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
                    # im = cv.cvtColor(im, cv.COLOR_BGR2HSV)

                    imgs[idx] = im
                    centers.append([x+centers_x[j], y+centers_y[i]])
                    widths_idxs.append(i_s)
                    idx += 1
        return imgs, centers, widths_idxs
                
    def detect_sign(self, frame, show_ROI=True, show_kp=True):
        """
        Sign classifiier:
        takes the whole frame as input and returns the sign name and classification
        confidence. If the network is not confident enough, it returns None sign name and 0.0 confidence
        """
        #ROI
        TL = (240, 10)   # x1, y1
        BR = (320, 70)   # x2, y2
        img = frame.copy()
        img = Detection.automatic_brightness_and_contrast(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        img = img[TL[1]:BR[1],TL[0]:BR[0]]
        #print(img.shape)
        if show_ROI:
            cv.imshow('ROI', img)
            Detection.draw_ROI(frame, TL, BR, show_rect = False, prediction = None, conf= None, show_prediction = False)
            cv.waitKey(1)
        #img = cv.resize(img,(40,30))
        ratio = 1
        img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
        kp, des = self.sign_sift.detectAndCompute(img, None)
        if(des is not None):
            if show_kp:
                cop = img.copy()
                cop = cv.drawKeypoints(cop,kp,cop,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv.imshow('keypoints',cop)
                cv.waitKey(1)   
            if len(des) < 10:
                print('No enougth descriptors')
                # cv.destroyWindow('keypoints')
                probs_array = np.zeros(len(signs_dict))
                # the last element of the list of obstacles is no obstacles
                # probability no obstacle = 1
                probs_array[-1] = 1
            else:
                im_hist = Detection.ImageHistogram(self.sign_kmean_model, des, self.sign_no_clusters)
                im_hist = im_hist.reshape(1,-1)
                im_hist = self.sign_scale_model.transform(im_hist)
                probs_array = self.sign_svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                print(f'single prediction: {SIGN_NAMES[MAP_DICT_NAMES[np.argmax(probs_array)]]}')
        else:
            print('No descriptors')
            # cv.destroyWindow('keypoints')
            probs_array = np.zeros(len(signs_dict))
            # the last element of the list of obstacles is no obstacles
            # probability no obstacle = 1
            probs_array[-1] = 1

        # buffer of classification probabilities
        self.sign_probs_buffer.append(probs_array)        

        # mean of the predicion probabilities in the buffer
        buffer_mean = list(self.sign_probs_buffer)
        buffer_mean = np.asarray(buffer_mean)
        buffer_mean = np.mean(buffer_mean, axis = 0)
        buffer_mean = buffer_mean.reshape(-1)
        pred_idx = np.argmax(buffer_mean)
        # most likely sign_prediction
        new_prediction = SIGN_NAMES[MAP_DICT_NAMES[pred_idx]]#signs_dict[str(pred_idx)]
        new_conf = buffer_mean[pred_idx]
        print('Prediction proposal:',new_prediction, int(100*new_conf))
        # Comparison of the first two maxima of the buffer_mean
        buffer_mean = np.delete(buffer_mean, pred_idx)
        conf_2 = buffer_mean[int(np.argmax(buffer_mean))]
        # To have a good enought sign_prediction the rest of the probabilities have to be spread
        if (new_conf - conf_2) > 0.30:
            self.sign_prediction = new_prediction
            self.sign_conf = new_conf
        print('New sign_prediction', self.sign_prediction, int(100*self.sign_conf))
        if show_ROI:
            Detection.draw_ROI(frame, TL, BR, show_rect = True, prediction = self.sign_prediction, conf= int(100*self.sign_conf), show_prediction = True)      
        return self.sign_prediction, self.sign_conf      

    def classify_frontal_obstacle(self, frames, distances, show_ROI=False, show_kp=False):
        """
        Obstacle classifier:
        takes the whole frames as input and returns the obstacle name and classification
        confidence.
        """
        assert len(frames) == len(distances)
        assert len(frames) > 0

        obstacles_probs_buffer = collections.deque(maxlen=len(frames)) 

        for frame, distance  in zip(frames, distances):
            img = frame.copy()
            img = Detection.automatic_brightness_and_contrast(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            distance = distance*100
            if distance >= 50:
                distance = 50
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1],TL[0]:BR[0]]
            elif distance < 50 and distance >= 40:
                distance = 40
                ratio = 0.9
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1],TL[0]:BR[0]]            
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
                
            elif distance < 40 and distance >= 30:
                distance = 30
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1],TL[0]:BR[0]]            
                ratio = 0.7
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
            #  distance < 30:     
            else:
                distance = 20
                TL = distance_dict[str(distance)][0]
                BR = distance_dict[str(distance)][1]
                img = img[TL[1]:BR[1],TL[0]:BR[0]]            
                ratio = 0.6
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
            if show_ROI:
                Detection.draw_ROI(frame, TL, BR, show_rect = False, prediction = None, conf= None, show_prediction = False)
                cv.imshow('ROI', img)
                cv.imwrite(f'sd/sd_{int(time()*1000)}.png', img)
                cv.waitKey(1)
            img = cv.GaussianBlur(img, (3,3) ,0)
            kp, des = self.sift.detectAndCompute(img, None) #des = descriptors
            if des is not None and len(des) >= 15:
                if show_kp:
                    cop = img.copy()
                    cop = cv.drawKeypoints(cop,kp,cop,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv.imshow('keypoints',cop)
                    cv.waitKey(1)   
                im_hist = Detection.ImageHistogram(self.kmean_model, des, self.no_clusters)
                im_hist = im_hist.reshape(1,-1)
                im_hist = self.scale_model.transform(im_hist)
                probs_array = self.svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                # buffer of classification probabilities
                obstacles_probs_buffer.append(probs_array)        
            else:
                print('No descriptors or not enough descriptors')
                #do not add it to the buffer

        if len(obstacles_probs_buffer) > 0:
            # mean of the predicion probabilities in the buffer
            buffer_mean = list(obstacles_probs_buffer)
            buffer_mean = np.asarray(buffer_mean)
            buffer_mean = np.mean(buffer_mean, axis = 0)
            buffer_mean = buffer_mean.reshape(-1)
            pred_idx = np.argmax(buffer_mean)
            # most likely prediction
            prediction = obstacles_dict[str(pred_idx)]
            self.prediction = prediction
            conf = buffer_mean[pred_idx]
            self.conf = conf
            print(f'Prediction proposal: {prediction} , {conf}')
            if show_ROI:
                Detection.draw_ROI(frame, TL, BR, show_rect = True, prediction = self.prediction, conf= int(100*self.conf), show_prediction = True)      
            return self.prediction, self.conf     
        else:
            print('No predictions')
            return None, None

    def classify_frontal_obstacle2(self, frames, distances, show_ROI=False, show_kp=False):
        """
        Obstacle classifier:
        takes the whole frames as input and returns the obstacle name and classification
        confidence.
        """
        assert len(frames) == len(distances)
        assert len(frames) > 0

        obstacles_probs_buffer = collections.deque(maxlen=len(frames)) 

        for frame, distance  in zip(frames, distances):
            img = frame.copy()
            img = Detection.automatic_brightness_and_contrast(img)
            img = cv.resize(img, (160,120))
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            distance = distance*100
            # if distance >= 50:
            #     distance = 50
            #     TL = distance_dict[str(distance)][0]
            #     BR = distance_dict[str(distance)][1]
            #     img = img[TL[1]:BR[1],TL[0]:BR[0]]
            # elif distance < 50 and distance >= 40:
            #     distance = 40
            #     ratio = 0.9
            #     TL = distance_dict[str(distance)][0]
            #     BR = distance_dict[str(distance)][1]
            #     img = img[TL[1]:BR[1],TL[0]:BR[0]]            
            #     img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
                
            # elif distance < 40 and distance >= 30:
            #     distance = 30
            #     TL = distance_dict[str(distance)][0]
            #     BR = distance_dict[str(distance)][1]
            #     img = img[TL[1]:BR[1],TL[0]:BR[0]]            
            #     ratio = 0.7
            #     img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
            # #  distance < 30:     
            # else:
            #     distance = 20
            #     TL = distance_dict[str(distance)][0]
            #     BR = distance_dict[str(distance)][1]
            #     img = img[TL[1]:BR[1],TL[0]:BR[0]]            
            #     ratio = 0.6
            #     img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
            # if show_ROI:
            #     Detection.draw_ROI(frame, TL, BR, show_rect = False, prediction = None, conf= None, show_prediction = False)
            #     cv.imshow('ROI', img)
            #     cv.imwrite(f'sd_{int(time()*1000)}.png', img)
            #     cv.waitKey(1)

            kp, des = self.sift.detectAndCompute(img, None) #des = descriptors
            if des is not None and len(des) >= 15:
                if show_kp:
                    cop = img.copy()
                    cop = cv.drawKeypoints(cop,kp,cop,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv.imshow('keypoints',cop)
                    cv.waitKey(1)   
                im_hist = Detection.ImageHistogram(self.kmean_model, des, self.no_clusters)
                im_hist = im_hist.reshape(1,-1)
                im_hist = self.scale_model.transform(im_hist)
                probs_array = self.svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                # buffer of classification probabilities
                obstacles_probs_buffer.append(probs_array)        
            else:
                print('No descriptors or not enough descriptors')
                #do not add it to the buffer
        
        if len(obstacles_probs_buffer) > 0:
            # mean of the predicion probabilities in the buffer
            buffer_mean = list(obstacles_probs_buffer)
            buffer_mean = np.asarray(buffer_mean)
            buffer_mean = np.mean(buffer_mean, axis = 0)
            buffer_mean = buffer_mean.reshape(-1)
            pred_idx = np.argmax(buffer_mean)
            # most likely prediction
            prediction = obstacles_dict[str(pred_idx)]
            self.prediction = prediction
            conf = buffer_mean[pred_idx]
            self.conf = conf
            print(f'Prediction proposal: {prediction} , {conf}')      
            return self.prediction, self.conf     
        else:
            print('No predictions')
            return None, None

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

