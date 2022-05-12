#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
        
import io
import numpy as np
import time
import pickle 
import rospy
from sensor_msgs.msg import Image
from utils.msg import traffic_sign_prediction
from cv_bridge import CvBridge
import cv2 as cv

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

class traffic_signs_classifierNODE():
    def __init__(self):
        """The purpose of this nodeis to get the images from the camera with the configured parameters
        and post the image on a ROS topic. 
        It is able also to record videos and save them locally. You can do so by setting the self.RecordMode = True.
        """
        rospy.init_node('traffic_signs_classifierNODE', anonymous=False)
        
        self.sign_classifier_pub = rospy.Publisher("/automobile/traffic_sign", traffic_sign_prediction, queue_size=1)
        self.CAMERA_subscriber = rospy.Subscriber("/automobile/image_raw", Image, self._classify)
    #=============================== CLASSIFIER MODELS ===================================
    def __init_classifier(self):
        self.no_clusters = 100
        self.kernel_type = 'linear'
        self.svm_model = pickle.load(open('models/svm_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.kmean_model = pickle.load(open('models/kmeans_' + self.kernel_type + '_' +  str(self.no_clusters) + '.pkl', 'rb'))
        self.scale_model = pickle.load(open('models/scale_'+ self.kernel_type + '_' + str(self.no_clusters) + '.pkl', 'rb'))
        self.sift = cv.SIFT_create()
        self.obstacles_probs_buffer = collections.deque(maxlen=10) 
        self.prediction = 'No sign'
        self.conf = 1        
        self.bridge = CvBridge()
        self.frame = np.zeros((320, 240))
    #================================ RUN ================================================
    def run(self):
        """Apply the initializing methods and start the thread. 
        """
        rospy.loginfo("starting traffic_signs_classifierNODE")
        self.__init_classifier()
        self._classify()
        time.sleep(2)    

    #================================ CLASSIFIER =========================================
        
    def _get_img(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
    def _classify(self, show_ROI = False, show_kp = False, show_results = False):
        """
        Obstacle classifier:
        takes the whole frame as input and returns the obstacle name and classification
        confidence. If the network is not confident enough, it returns None obstacle name and 0.0 confidence
        """
        while not rospy.is_shutdown():
            try:        
                img = self.frame.copy()
                img = automatic_brightness_and_contrast(img)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
                img = img[tl_corner[1]:br_corner[1],tl_corner[0]:br_corner[0]]
                #print(img.shape)
                if show_ROI:
                    cv.imshow('ROI', img)
                    draw_ROI(self.frame, tl_corner, br_corner, show_rect = False, prediction = None, conf= None, show_prediction = False)
                    cv.waitKey(1)
                #img = cv.resize(img,(40,30))
                ratio = 1
                img = cv.resize(img, None, fx = ratio, fy = ratio, interpolation = cv.INTER_AREA)
                kp, des = self.sift.detectAndCompute(img, None)

                if(des is not None) and len(des) > 10:
                    if show_kp:
                        cop = img.copy()
                        cop = cv.drawKeypoints(cop, kp, cop, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        cv.imshow('keypoints',cop)
                        cv.waitKey(1)   
                    im_hist = ImageHistogram(self.kmean_model, des, self.no_clusters)
                    im_hist = im_hist.reshape(1,-1)
                    im_hist = self.scale_model.transform(im_hist)
                    probs_array = self.svm_model.predict_proba(im_hist)
                    probs_array = probs_array.reshape(-1)
                    # inserting the No sing probability
                    probs_array = np.concatenate([probs_array, [0]])
                    # Falg to indicate classification was performed                
                    # OK_flag = True
                    # return OK_flag, probs_array, len(des)
                    # len_des = len(des)
                else:
                    print('None or not enought descriptors')
                    # cv.destroyWindow('keypoints')
                    probs_array = np.zeros(len(signs_dict))
                    # the last element of the list of traffic signs is no sign
                    # probability no obstacle = 1
                    probs_array[-1] = 1
                    # Falg to indicate classification was performed
                    # OK_flag = False
                    # return OK_flag, probs_array, 0.0
                    # len_des = 0

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
                pred = traffic_sign_prediction()
                pred.prediction = self.prediction
                pref.conf = self.conf
                self.sign_classifier_pub.publish(pred)

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
        return auto_result

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
            
if __name__ == "__main__":
    sing_class_NODE = cameraNODE()
    sing_class_NODE.run()
