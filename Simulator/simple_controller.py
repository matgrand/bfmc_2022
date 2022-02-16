#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline


class SimpleController():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0
        self.ff = ff

    def get_control(self, x, y, yaw, xd, yd, yawd, vd, curvature_ahead):
        e_yaw = np.arctan2(np.sin(yawd-yaw), np.cos(yawd-yaw)) # yaw error, in radians, not a simple difference (overflow at +-180)
        ex = xd - x #x error
        ey = yd - y #y error
        #similar to UGV trajectory tracking but frame of reference is left-hand
        self.e1 = -ex * np.cos(yaw) + ey * np.sin(yaw) #x error in the body frame
        self.e2 = -ex * np.sin(yaw) - ey * np.cos(yaw) #y error in the body frame
        self.e3 = e_yaw #yaw error
        output_angle = self.ff*curvature_ahead - self.k2 * self.e2 - self.k3 * self.e3
        output_speed = vd - self.k1 * self.e1
        return output_speed, output_angle



