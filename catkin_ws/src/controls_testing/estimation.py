# Estimation
from filterpy.kalman import ExtendedKalmanFilter as EKF
#rom casadi import *

from math import sin,cos,tan,radians,pi

# EKF estimation parameters
SIGMA_X = 0.1                       # [m] deviation of GPS:x measurement
SIGMA_Y = 0.1                       # [m] deviation of GPS:y measurement
SIGMA_H =  radians(0.01)            # [rad] deviation of GPS:yaw measurement
SIGMA_SPEED = 0.01                  # [m/s] deviation of SPEED action
SIGMA_STEER =  radians(0.1)         # [rad] deviation of STEER action

class EKFCar(EKF):
    def __init__(self,x0,WB):
        """Extended Kalman Filter for state estimation of BFMC car

        Args:
            x0 (ndarray): initial state of the car of dimension dim_xx
            WB (float): wheelbase of the car [m]
        """
        dim_xx = 3      # xx = [x, y, yaw]
        dim_zz = 3      # zz = [GPS:x, GPS:y, IMU:yaw]
        dim_uu = 2      # uu = [SPEED, STEER]
        EKF.__init__(self, dim_x=dim_xx, dim_z=dim_zz, dim_u=dim_uu)
        assert np.shape(x0) == (dim_xx,1)

        # State estimate
        self.x = np.copy(x0)

        # Output Measurement Covariance
        self.R = np.diag([SIGMA_X**2, SIGMA_Y**2, SIGMA_H**2])

        # Analytical model
        self.xx = MX.sym('x',dim_xx)
        self.uu = MX.sym('u',dim_uu)
        x = self.xx[0]      # GPS:x
        y = self.xx[1]      # GPS:y
        psi = self.xx[2]    # IMU:yaw

        # Input: [steering angle, speed]
        v = self.uu[0]      # SPEED
        delta = self.uu[1]  # STEER

        # Useful parameters
        lr = WB/2       # dist from rear wheel to CoM [m]
        beta = atan(lr/WB*tan(delta))# beta: side slip angle [rad]

        # ODE integration
        self.rhs = vertcat(v*cos(psi+beta), v*sin(psi+beta), v/WB*tan(delta)*cos(beta))# equations
        self.model = {}              # ODE declaration
        self.model['x']   = self.xx  # states
        self.model['p']   = self.uu  # inputs (as parameters)
        self.model['ode'] = self.rhs # right-hand side

        # Jacobian functions
        F_x = jacobian(self.rhs,self.xx)
        self.F_x = Function('Jx', [self.xx, self.uu], [F_x])
        #F_u = jacobian(self.rhs,self.uu)
        #self.F_u = Function('Ju', [self.xx, self.uu], [F_u])

        # Constants for covariance estimate update
        self.gammaQ = 1.0
        self.gammaP = 1.0

    def predict_x(self, u, dt):
        """Predicts the state

        Args:
            u (ndarray): current input vector
            dt (float): sampling time [s]
        """        
        # Function that integrates over dt
        self.ode_int = integrator('F','rk',self.model,{'tf':dt})
        # start from current x
        res = self.ode_int(x0=self.x, p=u)
        # State prediciton
        self.x = np.array(res['xf'])

    def predict(self, u, dt):
        """PREDICTION STEP (overwrites the EKF class method)

        Args:
            u (ndarray): current input vector
            dt (float): sampling time [s]
        """                
        # Compute Jacobians
        F = np.array(self.F_x(self.x, u))
        #V = np.array(self.F_u(self.x, u))

        # Process uncertainty in control space
        #self.M = np.array([[SIGMA_SPEED**2, 0], [0, SIGMA_STEER**2]])

        # State Covariance
        self.Q = np.diag([0.001, 0.001, 0.001])
        #self.Q = V @ self.M @ V.T       

        # STATE ERROR COVARIANCE PREDICTION
        self.P = self.gammaP * F @ self.P @ F.T + self.Q * self.gammaQ

        # STATE PREDICTION
        self.predict_x(u,dt)

        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
