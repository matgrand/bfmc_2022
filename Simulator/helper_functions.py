#!/usr/bin/python3
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#function to map meters to pixels, consts for different resolutions
const_simple = 196.5
const_med = 14164/15.0
const_verysmall = 3541/15.0
def m2pix(m):
    return np.int32(m*const_verysmall)

def yaw2world(angle):
    return -(angle + np.pi/2)

def world2yaw(angle):
    return -angle -np.pi/2

#function to draw the car on the map
def draw_car(map, x, y, angle, color=(0, 255, 0),  draw_body=True):
    car_length = 0.4 #m
    car_width = 0.2 #m
    #match angle with map frame of reference
    angle = yaw2world(angle)
    #find 4 corners not rotated
    corners = np.array([[-car_width/2, car_length/2],
                        [car_width/2, car_length/2],
                        [car_width/2, -car_length/2],
                        [-car_width/2, -car_length/2]])
    #rotate corners
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    corners = np.matmul(rot_matrix, corners.T)
    #add car position
    corners = corners.T + np.array([x,y])
    #draw body
    if draw_body:
        cv.polylines(map, [m2pix(corners)], True, color, 3, cv.LINE_AA)
        
    return map