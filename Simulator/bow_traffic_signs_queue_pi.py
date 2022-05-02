#!/usr/bin/python3

import numpy as np
import cv2 as cv
import os
from time import time, sleep
import pickle , collections

signs_dict = {
"0": "STOP", 
"1": "NO_ENTRY", 
"2": "PARKING", 
"3": "CROSS_WALK", 
"4": "STRAIGHT", 
"5": "HW_ENTER", 
"6": "HW_EXIT", 
"7": "PRIORITY", 
"8": "ROUND_ABOUT",
"9": "No sign"}

### ROI coordinates

# in the pi
tl_corner = (240, 10)   # x1, y1
br_corner = (320, 70)   # x2, y2
# # at home
# tl_corner = (240, 170)   # x1, y1
# br_corner = (320, 230)   # x2, y2   

#same func
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

#same func
def ImageHistogram(kmeans, descriptor_list, no_clusters):
    """
    Compute the histogram of occurrences of the visual words in the input image
    """
    im_hist = np.zeros(no_clusters)
    
    # feature is the descriptor of a single keypoint
    for feature in descriptor_list:

        feature = feature.reshape(1, 128)
        idx = kmeans.predict(feature)
        im_hist[idx] += 1
    return im_hist

#same func
def draw_ROI(frame, tl_corner, br_corner, show_rect = False, prediction = None, conf= None, show_prediction = False):
    # Blue color in BGR
    
    if show_rect: 
        image = frame.copy()
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv.rectangle(image, tl_corner, br_corner, color = (255, 0, 0), thickness = 2)
        cv.imshow("Frame preview", image)
        cv.waitKey(1)
    if show_rect and show_prediction:
        image = frame.copy()
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv.rectangle(image, tl_corner, br_corner, color = (255, 0, 0), thickness = 2)
        cv.putText(img=image, text= prediction + ' ' + str(conf) + '%', org=(tl_corner[0]-100, tl_corner[1]), 
            fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
        cv.imshow("Frame preview", image)
        cv.waitKey(1)

class Detection:

    #init 
    def __init__(self):

        self.no_clusters = 100
        self.kernel_type = 'linear'
        self.svm_model = pickle.load(open('models/svm_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.kmean_model = pickle.load(open('models/kmeans_' + self.kernel_type + '_' +  str(self.no_clusters) + '.pkl', 'rb'))
        self.scale_model = pickle.load(open('models/scale_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.sift = cv.SIFT_create()
        self.obstacles_probs_buffer = collections.deque(maxlen=10) 
        self.prediction = 'No sign'
        self.conf = 1
        
    def classify_sign(self, frame, show_ROI = True, show_kp = True, show_results = True):
        """
        Obstacle classifier:
        takes the whole frame as input and returns the obstacle name and classification
        confidence. If the network is not confident enough, it returns None obstacle name and 0.0 confidence
        """
        img = frame.copy()
        img = automatic_brightness_and_contrast(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        img = img[tl_corner[1]:br_corner[1],tl_corner[0]:br_corner[0]]
        #print(img.shape)
        if show_ROI:
            cv.imshow('ROI', img)
            draw_ROI(frame, tl_corner, br_corner, show_rect = False, prediction = None, conf= None, show_prediction = False)
            cv.waitKey(1)
        #img = cv.resize(img,(40,30))
        ratio = 1
        img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
        kp, des = self.sift.detectAndCompute(img, None)
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
                # Falg to indicate classification was performed
                OK_flag = False
                # return OK_flag, probs_array, 0.0
                len_des = len(des)
            else:
                im_hist = ImageHistogram(self.kmean_model, des, self.no_clusters)
                im_hist = im_hist.reshape(1,-1)
                im_hist = self.scale_model.transform(im_hist)
                probs_array = self.svm_model.predict_proba(im_hist)
                probs_array = probs_array.reshape(-1)
                # inserting the No sing probability
                probs_array = np.concatenate([probs_array, [0]])
                # Falg to indicate classification was performed                
                OK_flag = True
                # return OK_flag, probs_array, len(des)
                len_des = len(des)
        else:
            print('No descriptors')
            # cv.destroyWindow('keypoints')
            probs_array = np.zeros(len(signs_dict))
            # the last element of the list of obstacles is no obstacles
            # probability no obstacle = 1
            probs_array[-1] = 1
            # Falg to indicate classification was performed
            OK_flag = False
            # return OK_flag, probs_array, 0.0
            len_des = 0

        # buffer of classification probabilities
        self.obstacles_probs_buffer.append(probs_array)        

        # mean of the predicion probabilities in the buffer
        buffer_mean = list(self.obstacles_probs_buffer)
        buffer_mean = np.asarray(buffer_mean)
        buffer_mean = np.mean(buffer_mean, axis = 0)
        buffer_mean = buffer_mean.reshape(-1)
        pred_idx = np.argmax(buffer_mean)
        # most likely prediction
        new_prediction = signs_dict[str(pred_idx)]
        new_conf = buffer_mean[pred_idx]
        print('Prediction proposal:',new_prediction, int(100*new_conf))
        # Comparison of the first two maxima of the buffer_mean
        buffer_mean = np.delete(buffer_mean, pred_idx)
        conf_2 = buffer_mean[int(np.argmax(buffer_mean))]
        # To have a good enought prediction the rest of the probabilities have to be spread
        if (new_conf - conf_2) > 0.30:
            self.prediction = new_prediction
            self.conf = new_conf
        if show_results:
            print('New prediction', self.prediction, int(100*self.conf))
            draw_ROI(frame, tl_corner, br_corner, show_rect = True, prediction = self.prediction, conf= int(100*self.conf), show_prediction = True)      
        return self.prediction, self.conf, len_des      


show_frame  = False
if __name__ == '__main__':
    
    try:
        des_len_list = []
        detector = Detection()
        # obstacles_probs_buffer = collections.deque(maxlen=10)
        t_start = time()
        count=0
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv.CAP_PROP_FPS, 30)

        while True:                        
            while time() - t_start < 1:
                ret, frame = cap.read()
                # Try until good frame is captured
                while not ret:
                    ret, frame = cap.read()   
                if show_frame:
                    cv.imshow("Frame preview", frame) 
                    cv.waitKey(1)  
                # Classification of a single frame              
                prediction, conf, descriptors_len = detector.classify_frontal_obstacle(frame)
                des_len_list.append(descriptors_len)
                count +=1

            # list to array
            des_len_list = np.asarray(des_len_list)
            mean_len = np.mean(des_len_list, axis = 0)      
            t_start = time()
            #print(prediction, conf)  
            print('mean num of kp', mean_len)     
            print('FPS', count) 
            des_len_list = []
            count = 0         

    except KeyboardInterrupt:
        print('inside interrupt exeption')
        pass