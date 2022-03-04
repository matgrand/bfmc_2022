#!/usr/bin/python3

# HELPER FUNCTIONS
import numpy as np
import cv2 as cv

def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def m2pix(m):
    #const_simple = 196.5
    #const_med = 14164/15.0
    const_verysmall = 3541/15
    return np.int32(m*const_verysmall)

def yaw2world(angle):
        return -(angle + np.pi/2)

def world2yaw(angle):
    return -angle -np.pi/2












