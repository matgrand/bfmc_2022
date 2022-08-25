#!/usr/bin/python3
import numpy as np
L = 0.4  #length of the car, matched with lane_detection
class Controller():
    def __init__(self, k=1.0, noise_std=np.deg2rad(20)):
        #controller paramters
        self.k = k
        self.cnt = 0
        self.noise = 0.0

    def get_control(self, alpha, dist_point_ahead):
        d = dist_point_ahead#POINT_AHEAD_CM/100.0 #distance point ahead, matched with lane_detection
        delta = np.arctan((2*L*np.sin(alpha))/d)
        return  - self.k * delta
        



    
