#!/usr/bin/python3

# HELPER FUNCTIONS
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def m2pix(m):
    #const_simple = 196.5
    #const_med = 14164/15.0
    const_verysmall = 3541/15.0
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

def project_onto_frame(frame, car, points):
    # #debug
    # proj = np.copy(map)
    num_points = points.shape[0]
    #check shape
    if points[0].shape == (2,):
        #add 0 Z component
        points = np.concatenate((points, np.zeros((num_points,1))), axis=1)
    # assert points[0].shape == (3,), "points must be (3,), got {}".format(points[0].shape)
    
    #get points inside the fov:
    #translation
    diff = points - np.array([car.x_true, car.y_true, car.cam_z])
    #angles
    angles = - np.arctan2(diff[:,1], diff[:,0])
    #calculate angle differences
    diff_angles = diff_angle(angles, car.yaw)
    #get points in front of the car
    # front_points = []
    rel_pos_points = []
    for i, point in enumerate(points):
        if np.abs(diff_angles[i]) < car.cam_fov/2:
            # front_points.append(point)
            rel_pos_points.append(diff[i])

    if len(rel_pos_points) == 0:
        # return frame, proj
        return frame, None
    
    # #plot the points in the 2d map
    # for p in front_points:
    #     cv.circle(map, m2pix(p[:2]), 15, (0,255,255), -1)

    #convert to numpy
    # front_points = np.array(front_points)
    rel_pos_points = np.array(rel_pos_points)

    #rotate the points around the z axis
    gamma = car.yaw
    rot_matrix = np.array([[np.cos(gamma), -np.sin(gamma), 0],[np.sin(gamma), np.cos(gamma), 0 ], [0,0,1]])
    rel_pos_points = np.matmul(rot_matrix, rel_pos_points.T).T

    #rotate the points around the relative y axis, pitch
    beta = -car.cam_pitch
    rot_matrix = np.array([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]])
    
    rotated_points = np.matmul(rot_matrix, rel_pos_points.T).T

    # max_x = 100*np.max(rotated_points[:,0])
    # min_x = 100*np.min(rotated_points[:,0])
    # max_y = 100*np.max(rotated_points[:,1])
    # min_y = 100*np.min(rotated_points[:,1])
    # proj = np.zeros((int(max_y-min_y+1), int(max_x-min_x+1), 3), dtype=np.uint8)
    # for p in rotated_points:
    #     x = int(100*p[0]-min_x)
    #     y = int(100*p[1]-min_y)
    #     cv.circle(proj, (x,y), 5, (0,255,255), -1)

    # 3d plot the points
    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter3D(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], c='r', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    #project the points onto the camera frame
    proj_points = np.array([[p[1]/p[0], -p[2]/p[0]] for p in rotated_points])
    #convert to pixel coordinates
    proj_points = 500*proj_points + np.array([320, 240])

    # draw the points
    for p in proj_points:
        cv.circle(frame, (round(p[0]), round(p[1])), 2, (0,255,255), -1)

    return frame, proj_points

