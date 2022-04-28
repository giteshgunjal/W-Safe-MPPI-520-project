
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import rospy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm

# def wrapToPi(phase):
#             x_wrap = np.remainder(phase, 2 * np.pi)
#             while abs(x_wrap) > np.pi:
#                 x_wrap -= 2 * np.pi * np.sign(x_wrap)
#             return x_wrap


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance
    

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u):
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)
        self.u_se2 = u_se2

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        # print(H_prev)
        # print(H_prev[:2,:2],H_prev[:2,2])
        # print(np.cross(H_prev[:2,2], H_prev[:2,:2]))
        # print(np.zeros((2,2)))
        # adjX  = np.hstack((H_prev[0:2, 0:2], np.array([[H_prev[1, 2]], [-H_prev[0, 2]]])))
        # adjX= np.vstack((adjX, np.array([0, 0, 1])))
        adjX = self.Ad(H_prev)

        # print(adjX)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.propagation(u_se2, adjX)

    def propagation(self, u, adjX):
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################
        
        
        # X = self.mu
        self.X_pred =self.mu @ expm(u)
        self.P_pred = self.Sigma + adjX@ self.W@adjX.T
        self.mu  = self.X_pred

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
    def correction(self, Y1, Y2, z, landmarks):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        # print(Y1)
        b1 =np.hstack((landmark1.getPosition(),[1]))
        b2 =np.hstack((landmark2.getPosition(),[1]))
        # print(b1)
        b = np.hstack((b1,b2))
        Y = np.hstack((Y1,Y2))
        # print(Y)
        H1 = np.array([[-1,0,b1[1]],
        [0,-1,-b1[0]]
        ])
        H2 = np.array([[-1,0,b2[1]],
        [0,-1,-b2[0]]
        ])
     
        H = np.vstack((H1,H2))

   
        
        N = self.X_pred@block_diag(self.V, 0)@self.X_pred.T
        N = block_diag(N[0:2, 0:2],N[0:2, 0:2])
        S = H@self.P_pred@H.T +N
        # print(S)
        L = self.P_pred@H.T@np.linalg.inv(S)
        # print("L",L@H)
        
        X_pred = block_diag(self.X_pred,self.X_pred)
        nu =X_pred@Y-b
        # print(nu)
        nu = np.hstack((nu[0:2], nu[3:5]))
        u = L@nu
        uhat = self.wedge(u)
        self.mu = expm(uhat)@self.X_pred
        # print("next X",self.mu)
        X = np.array([self.mu[0,2], self.mu[1,2], wrap2Pi(np.arctan2(self.mu[1,0], self.mu[0,0]))])
        P_corr = (np.eye(3)-L@H )@self.P_pred@(np.eye(3)-L@H ).T  + L@N@L.T
        

        self.Sigma = P_corr
        

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        
        self.state_.setTime(rospy.Time.now())
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
    def Ad(self, X):
        # Adjoint
        AdX = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
        AdX = np.vstack((AdX, np.array([0, 0, 1])))
        return AdX

    def wedge(self, x):
        # wedge operation for se(2) to put an R^3 vector to the Lie algebra basis
        G1 = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])  # omega
        G2 = np.array([[0, 0, 1],
                       [0, 0, 0],
                       [0, 0, 0]])  # v_1
        G3 = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 0, 0]])  # v_2
        xhat = G2 * x[0] + G3 * x[1] + G1 * x[2]
        return xhat

    