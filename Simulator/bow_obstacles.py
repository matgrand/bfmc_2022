#!/usr/bin/python3

import numpy as np
import cv2 as cv
from time import time
import pickle , collections

obstacles_dict = {
    "0": "car",
    "1": "pedestrian",
    "2": "roadblock",
}

distance_dict = {
    '20': [(94, 60),(227, 160)],
    '30': [(103, 45),(217, 131)],
    '40': [(116, 43),(205, 110)],
    '50': [(120, 40),(200, 100)]
}

### ROI coordinates

#to helper functions
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

        feature = feature.reshape(1, 128)
        idx = kmeans.predict(feature)
        im_hist[idx] += 1
    return im_hist

#into detection class
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

NUM_CLUSTERS = 1200
KERNEL_TYPE = 'linear'

class Detection:

    #init 
    def __init__(self):
        self.no_clusters = NUM_CLUSTERS #use const
        self.kernel_type = KERNEL_TYPE #const
        self.svm_model = pickle.load(open('models/obstacle_models/svm_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.kmean_model = pickle.load(open('models/obstacle_models/kmeans_' + self.kernel_type + '_' +  str(self.no_clusters) + '.pkl', 'rb'))
        self.scale_model = pickle.load(open('models/obstacle_models/scale_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.sift = cv.SIFT_create()
        # self.obstacles_probs_buffer = collections.deque(maxlen=OBSTACLE_DETECTION_DEQUE_LENGTH) 
        self.prediction = None
        self.conf = 0
        
    def classify_frontal_obstacle(self, frames, distances, show_ROI=True, show_kp=True):
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
            img = automatic_brightness_and_contrast(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            distance = distance*100
            if distance >= 50:
                distance = 50
                tl_corner = distance_dict[str(distance)][0]
                br_corner = distance_dict[str(distance)][1]
                img = img[tl_corner[1]:br_corner[1],tl_corner[0]:br_corner[0]]
            elif distance < 50 and distance >= 40:
                distance = 40
                ratio = 0.9
                tl_corner = distance_dict[str(distance)][0]
                br_corner = distance_dict[str(distance)][1]
                img = img[tl_corner[1]:br_corner[1],tl_corner[0]:br_corner[0]]            
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
                
            elif distance < 40 and distance >= 30:
                distance = 30
                tl_corner = distance_dict[str(distance)][0]
                br_corner = distance_dict[str(distance)][1]
                img = img[tl_corner[1]:br_corner[1],tl_corner[0]:br_corner[0]]            
                ratio = 0.7
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
            #  distance < 30:     
            else:
                distance = 20
                tl_corner = distance_dict[str(distance)][0]
                br_corner = distance_dict[str(distance)][1]
                img = img[tl_corner[1]:br_corner[1],tl_corner[0]:br_corner[0]]            
                ratio = 0.6
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
            if show_ROI:
                draw_ROI(frame, tl_corner, br_corner, show_rect = False, prediction = None, conf= None, show_prediction = False)
                cv.imshow('ROI', img)
                cv.waitKey(1)

            kp, des = self.sift.detectAndCompute(img, None) #des = descriptors
            if des is not None and len(des) >= 15:
                if show_kp:
                    cop = img.copy()
                    cop = cv.drawKeypoints(cop,kp,cop,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv.imshow('keypoints',cop)
                    cv.waitKey(1)   
                im_hist = ImageHistogram(self.kmean_model, des, self.no_clusters)
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
                draw_ROI(frame, tl_corner, br_corner, show_rect = True, prediction = self.prediction, conf= int(100*self.conf), show_prediction = True)      
            return self.prediction, self.conf     
        else:
            print('No predictions')
            return None, None


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
                prediction, conf, descriptors_len = detector.classify_frontal_obstacle(frame, distance = 20)
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
