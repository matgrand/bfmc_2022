#!/usr/bin/python3
import numpy as np
import cv2 as cv
import os

from helper_functions import *
import time
import math
from scipy.signal import butter, lfilter, lfilter_zi
import collections
from casadi import *
from pyclothoids import Clothoid

POINT_AHEAD_CM = 35 #distance of the point ahead in cm
SEQ_POINTS_INTERVAL = 20 #interval between points in the sequence in cm 
NUM_POINTS = 5 #number of points in the sequence
L = 0.4  #length of the car, matched with lane_detection
WB = 0.26

# def high_pass(x_new, x_old, dt, y_old, cutoff=0.1):
#     alpha = dt / (dt + 1 / (2 * np.pi * cutoff))
#     y_new = alpha * (y_old + x_new - x_old)
#     return y_new

# def low_pass(x_new, y_old, Ts, cutoff=1000):
#     alpha = math.exp(-2*np.pi*cutoff * Ts)
#     y_new = alpha * y_old + x_new * ( 1 - alpha )
#     return y_new 
filter_ic = None
class Filter():
    def __init__(self, type, normal_cutoff, order=5) -> None:
        self.normal_cutoff = normal_cutoff
        self.order = order
        self.ic = None
        if type == 'low' or type == 'high':
            self.type = type
        else:
            raise Exception('invalid filter type')
        self.update_filter_coefficients()

    def filter(self, data):
        self.ic = lfilter_zi(self.b, self.a) * data[0] if self.ic is None else self.ic
        y , self.ic = lfilter(self.b, self.a, data, zi=self.ic)
        return y
    
    def update_filter_coefficients(self):
        self.b, self.a = butter(self.order, self.normal_cutoff, btype=self.type, analog=False)

    def update_params(self, type=None, normal_cutoff=None, order=None):
        self.normal_cutoff = normal_cutoff if normal_cutoff is not None else self.normal_cutoff
        self.order = order if order is not None else self.order
        if type is not None:
            if type == 'low' or type == 'high':
                self.type = type
            else:
                raise Exception('invalid filter type')
        else:
            pass
        self.update_filter_coefficients()

class Controller():
    def __init__(self):
        self.a = WB/2                                   # [m]: distance of camera from front axis   
        self.L_ad = POINT_AHEAD_CM/100.0 - self.a      # [m]: lookahead distance

        # self.Kp = 0.3
        # self.Ki = 0.2
        # self.Kpp = 0.6
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kpp = 0.7
        self.prev_steering_angle = 0.0
        self.curvature_ahead = 0.0
        self.path_ahead = None

        self.Kstanley_e = 1.0
        self.Kstanley_v = 0.05
        self.Kstanley_ks = 1.0

        self.Kp_vel = 1/np.deg2rad(20)
        self.Kd_vel = 0.5
        self.prev_time = None

        self.alpha = 0.0        # x[i]
        self.alpha_filt = 0.0   # y[i] and y[i-1]
        self.alpha_filter = Filter(type='high', normal_cutoff=0.05, order=5)

        self.e = 0.0
        self.e_filt = 0.0
        self.e_filter = Filter(type='high', normal_cutoff=0.2, order=3)

        self.e_int = 0.0

        # SMITH PREDICTOR
        self.delta_state = MX.sym('d_r',1)
        self.delta_input = MX.sym('d(t-tau_d)',1)

        tau = 0.3       # [s]
        tau_d = 0.15    # [s]

        # ODE integration
        self.rhs = -1/tau * self.delta_state + 1/tau * self.delta_input 
        self.model = {}                         # ODE declaration
        self.model['x']   = self.delta_state    # states
        self.model['p']   = self.delta_input    # inputs (as parameters)
        self.model['ode'] = self.rhs            # right-hand side

        # state
        self.delta = 0.0


    def get_control(self, e2, e3, vel, point_ahead, desired_speed):
        curr_time = time.time()
        if self.prev_time is None:
            self.prev_time = curr_time
            return 0.0,0.0
        else:
            DT = curr_time - self.prev_time

            # INPUT DATA
            self.e = e2      # lateral error
            self.alpha = -e3  # heading error

            # FILTER INPUT DATA
            self.alpha_filt = self.alpha_filter.filter([self.alpha])
            self.e_filt = self.e_filter.filter([self.e])

            # PREDICT CURVATURE
            self.path_ahead = Clothoid.Forward(0,0,0, 2*sin(self.alpha)/(POINT_AHEAD_CM/100.0), point_ahead[0], -point_ahead[1])
            self.curvature_ahead = self.path_ahead.KappaEnd

            # LONGITUDINAL CONTROLLER
            alpha_abs = abs(self.alpha)
            output_speed = ( 1 - self.Kp_vel*alpha_abs ) * desired_speed

            # LATERAL CONTROLLER                     
            # Proportional
            e_la = self.e + (self.a+self.L_ad)*np.sin(self.alpha)
            delta_p = self.Kp * e_la
            # Integral
            self.e_int += self.alpha_filt#self.e_filt
            delta_i = self.Ki * self.e_int
            # Pure Pursuit
            # delta_pp = np.arctan((2*WB*np.sin(self.alpha))/(self.L_ad))
            delta_pp = np.arctan((2*WB*np.sin(self.alpha))/(0.6*vel))
            # ================================================================
            # OUTPUT ANGLE
            output_angle = self.Kpp * delta_pp + delta_p + delta_i - 0.0*self.curvature_ahead

            # PREDICT DELTA
            self.delta = self.predict_delta(u=output_angle, dt=0.15)
            # output_angle = self.delta


            os.system('cls' if os.name=='nt' else 'clear')
            print('alpha_abs:  ',np.rad2deg(alpha_abs))
            print(f'alpha: {self.alpha}')
            print(f'steering angle: {np.rad2deg(output_angle)}')
            print(f'sampling time is: {DT}')
            print(f'curvature ahead: {self.curvature_ahead}')

            self.prev_steering_angle = output_angle
            self.prev_time = curr_time

        return output_speed, output_angle
    
    def get_control_stanley(self, e2, e3, vel, desired_speed):

        #e2: lateral error
        #e3: heading error
        e = e2
        alpha = -e3

        output_speed = ( 1-abs(alpha/np.deg2rad(60)) ) * output_speed

        e_front = e - WB/2*np.sin(alpha)                    
        delta = np.rad2deg(np.arctan( -self.Kstanley_e*e_front / (self.Kstanley_ks + self.Kstanley_v*output_speed) ) - alpha)

        output_angle = delta

        os.system('cls' if os.name=='nt' else 'clear')
        print(f'steering angle: {np.rad2deg(output_angle)} deg')
        print(f'lateral error: {e} m')
        print(f'front lateral error: {e_front} m')
        print(f'heading error: {np.rad2deg(alpha)} deg')
        print(f'output speed: {output_speed} m/s')

        return output_speed, output_angle
    
    def predict_delta(self, u, dt):
        # Integrator over dt
        ode_int = integrator('F','rk',self.model,{'tf':dt})
        # Integrate, starting from current state
        res = ode_int(x0=self.delta, p=u)
        return np.array(res['xf'])[0]
