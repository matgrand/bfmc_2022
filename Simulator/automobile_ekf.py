#!/usr/bin/python3
from casadi import *
from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np
import time

DEBUG = False

# EKF estimation parameters
SIGMA_X         = 0.1                   # [m] deviation of GPS:x measurement
SIGMA_Y         = 0.1                   # [m] deviation of GPS:y measurement
SIGMA_H         =  np.deg2rad(0.01)     # [rad] deviation of GPS:yaw measurement
SIGMA_SPEED     = 0.01                  # [m/s] deviation of SPEED action
SIGMA_STEER     =  np.deg2rad(0.1)      # [rad] deviation of STEER action

class AutomobileEKF(EKF):
    def __init__(self, x0, WB):
        """_summary_

        :param x0: initial state
        :type x0: numpy.ndarray
        :param WB: [m]: car wheelbase length
        :type WB: float
        """        
        dim_xx = 2      # xx = [x, y, yaw]^T
        dim_zz = 2      # zz = [GPS:x, GPS:y, IMU:yaw]^T
        dim_uu = 2      # uu = [speed, steer]^T

        EKF.__init__(self, dim_x=dim_xx, dim_z=dim_zz, dim_u=dim_uu)
        assert np.shape(x0) == (dim_xx,1)
        self.x = np.copy(x0)

        # Measurement uncertainty
        self.R = np.diag([SIGMA_X**2, SIGMA_Y**2])

        # Analytical model
        self.xx = MX.sym('x',dim_xx)
        self.uu = MX.sym('u',dim_uu)
        x = self.xx[0]      # [m]   GPS:x
        y = self.xx[1]      # [m]   GPS:y

        # Input: [speed, steering angle]
        v = self.uu[0]      # [m/s] SPEED
        psi = self.uu[1]    # [rad] YAW

        # *******************************************
        # Useful parameters
        lr = WB/2                       # [m]   dist from rear wheel to CoM
        # *******************************************
        # ODE integration
        self.rhs = vertcat(v*cos(psi), v*sin(psi))
        self.model = {}              # ODE declaration
        self.model['x']   = self.xx  # states
        self.model['p']   = self.uu  # inputs (as parameters)
        self.model['ode'] = self.rhs # right-hand side

        F_x = jacobian(self.rhs,self.xx)
        F_u = jacobian(self.rhs,self.uu)
        self.F_x = Function('Jx', [self.xx, self.uu], [F_x])
        self.F_u = Function('Ju', [self.xx, self.uu], [F_u])

        # Regularization terms
        self.gammaQ = 1.0
        self.gammaP = 1.0

        if DEBUG:
            print('*****************************')
            print('this is the kinematic bicycle model\n',self.rhs)
            print('*****************************')
            print('this is df/dx\n',F_x)
            print('*****************************')

    def predict_x(self, u, dt):
        # Integrator over dt
        self.ode_int = integrator('F','rk',self.model,{'tf':dt})
        # Integrate, starting from current state
        res = self.ode_int(x0=self.x, p=u)
        self.x = np.array(res['xf'])
        
        if DEBUG:
            print(f'predicted state is {self.x}')
            print(f'with shape {self.x.shape}')
            print('*****************************')

    def predict(self, u, dt = 0.1):
        ''' Overwrite the predict method '''
        # *******************************************
        # Jacobians
        # *******************************************
        F = np.array(self.F_x(self.x, u))   # State Jacobian
        V = np.array(self.F_u(self.x, u))   # Input Jacobian

        # *******************************************
        # Error covariance prediction
        # *******************************************
        self.Q = np.diag([0.001, 0.001])
        self.P = self.gammaP * F @ self.P @ F.T + self.Q * self.gammaQ

        # *******************************************
        # State prediction
        # *******************************************
        self.predict_x(u,dt)

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
    
    def estimate_state(self, sampling_time, input, output):
        ''' Updates estimated state according to EKF '''
        DT = sampling_time
        u = input
        z = output
        # *******************************************
        # PREDICT STEP
        # *******************************************
        self.predict(u=u, dt=DT)
        # *******************************************
        # UPDATE STEP
        # *******************************************
        self.update(z=z, HJacobian=HJacobian, Hx=Hx, residual=residual)
        # *******************************************
        # ESTIMATED STATE
        # *******************************************
        xxEst = np.copy(self.x)
        # x: CoM
        x_est = xxEst[0,0]
        # y: CoM
        y_est = xxEst[1,0]

        return x_est, y_est
    
def residual(a,b):
    """ compute residual between two measurement.
    """
    return a - b

def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def HJacobian(x):
    ''' compute Jacobian of the output map h(x) '''
    H = np.array([[1,0],
                [0,1]])
    return H


def Hx(x):
    ''' compute output given current state '''
    H = np.array([[1,0],
                [0,1]])
    z = np.matmul(H,x)
    return z
