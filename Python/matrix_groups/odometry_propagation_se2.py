#!usr/bin/env python

#
# An example of process models on Lie groups and uncertainty propagation
# using SE(2). The process model is simply X_k+1 = X_k * U_k exp(w_k) where
# X_k, U_k are both in SE(2) and w_k is N(0, Q_k) and defined in the Lie
# algebra se(2). We use Monte Carlo methods to propagate samples over a
# path and then compute the sample mean and covariance on Lie group, here
# SE(2). Note that the sample mean and covariance are computed using an
# iterative algorithm which is different than usual Euclidean sample
# statistics. The covariance on Lie algebra is flat as expected but it's
# nonlinear when mapped to the manifold using Lie group.
#
# Author: Fangtong Liu
# Date: 05/14/2020
#

from calendar import c
from httplib2 import ProxiesUnavailableError
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
from scipy.stats import wasserstein_distance
import ot
import pickle
import time

# from Python.matrix_groups.wasscheck import pose_mat

n = 50  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

mu_t = np.array([4, 4])
cov_t = np.array([[1, -.8], [-.8, 1]])

xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)

a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
# print(np.shape(a))

# loss matrix
M = ot.dist(xs, xt, metric = 'euclidean')
G1 = ot.emd2(a, b, M)
# print(M)

def pose_mat( X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                    [np.sin(h),np.cos(h),y],\
                    [0,0,1]])
        return H

# print(G1)
# print(M.max().type())

M /= M.max()
# print(M)


u = np.array([1,2,3,5]).reshape(-1,1)
n = u.shape[0]
# print(n)
# uwt = np.array([1,2])/ np.sum(np.array([1,2]))
uwt = np.ones((n,)) / n
# print(uwt)
v = np.array([5,3,6,8]).reshape(-1,1) 
vwt = np.ones((n,)) / n
# vwt = np.array([1,2,3])/ np.sum(np.array([1,2,3]))
# print(wasserstein_distance(u.reshape((-1,)),v.reshape((-1,)),uwt,vwt))

M = ot.dist(u,v,metric='euclidean')
# print(M)
# b =  np.array(M.max())
# print(b)
# M = M/np.array(M.max())


G0 = ot.emd2(uwt, vwt, M)
# print(G0)
# G0 = ot.emd2(a, b, M)

# c

def adjoint(X):
    # SE(2) Adjoint function
    matrix = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
    matrix = np.vstack((matrix, np.array([0, 0, 1])))
    # Adjoint
        # AdX = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
        # AdX = np.vstack((AdX, np.array([0, 0, 1])))
    return matrix


def pos_def_matrix(A):
    # Check input matrix is positive definite or not
    return np.all(np.linalg.eigvals(A) > 0)

def pose2vec(H):
    state_vector = np.zeros(3)  
    state_vector[0] = H[0,2]
    state_vector[1] = H[1,2]
    state_vector[2] = np.arctan2(H[1,0], H[0,0])
    return state_vector

def propagation(robot, u):
    # SE(2) propagation model; each input is U \in SE(2) plus exp map of the
    # invprop

    r_prev = pose2vec(robot['Xpred'])

    # print(r_prev)
    # print('u',u)

    # print('exp', expm(np.zeros((3,3))))
    
    v = np.sqrt(u[0]**2 + u[1]**2)

    R_prev = robot['Xpred']
    r_prev[0] += 1 * (v * np.cos(u[2]))
    r_prev[1] += 1 * (v * np.sin(u[2]))
    r_prev[2] += 1 * u[2]

    # print('rnext1model', r_prev)

    Ui = np.array([[np.cos(u[2]), -np.sin(u[2]), u[0]],
                       [np.sin(u[2]), np.cos(u[2]), u[1]],
                       [0, 0, 1]])

    # print('U', Ui)
    robot['Xpred'] = robot['Xpred']@Ui
    # print(np.linalg.inv(R_prev)@ robot['Xpred'])
    adjX = adjoint(robot['Xpred'])
    
    r_next = pose2vec(robot['Xpred'])
    # print('rnext 2model', r_next)

    robot['P'] = robot['P'] + adjX@robot['Q']@adjX.T
    # noise defined in Lie algebra
    for i in range(robot['n']):
        #  sample from a zero mean Gaussian
        noise = np.dot(robot['L'], np.random.randn(3, 1))
        N = robot['G1'] * noise[0] + robot['G2'] * noise[1] + robot['G3'] * noise[2]
        
        robot['x'][i] = np.dot(np.dot(robot['x'][i], Ui), expm(N))
    return robot


def Lie_sample_statistics(robot):
    # compute sample mean and covariance on matrix Lie group
    mu0 = robot['x'][0]  # pick a sample as initial guess
    v = np.copy(robot['x'])
    max_iter = 100
    iter = 1
    while iter < max_iter:
        mu = mu0 * 0
        Sigma = np.zeros([3, 3])
        for i in range(robot['n']):
            # left-invariant error: eta^L = X^(-1) * X^hat
            v[i] = logm(np.dot(np.linalg.inv(mu0), robot['x'][i]))
            mu = mu + v[i]
            vec_v = np.array([[v[i][0, 2]],
                              [v[i][1, 2]],
                              [v[i][1, 0]]])
            Sigma = Sigma + np.dot(vec_v, vec_v.T)
        mu = np.dot(mu0, expm(mu / robot['n']))
        Sigma = (1 / (robot['n'] - 1)) * Sigma  # unbiased sample covariance
        # check if we're done here
        temp = np.linalg.norm(logm(np.dot(np.linalg.inv(mu0), mu)))
        if temp < 1e-8:
            return mu, Sigma
        else:
            mu0 = np.copy(mu)
        iter += 1
    print('\033[91mWarning: Not converged! Max iteration reached. The statistic might not be reliable.\033[0m')


if __name__ == "__main__":

    # generate a path
    # dt = 0.6
    # gt = {}
    # gt['x'] = np.arange(0, 4 + dt, dt)
    # print(gt['x'])
    # gt['y'] = 0.2 * np.exp(0.9 * gt['x']) - 0.1

    # print('y',gt['x'])


    dt = 0.6
    gt = {}
    gt['x'] = np.arange(0, 4 + dt, dt)
    gt['y'] = 0.1 * np.exp(0.6 * gt['x']) - 0.1
# # for obs
#     dto = 0.6
#     gto = {}
#     gto['x'] = np.arange( 7 + dt,0, -dt)
#     print('gtox',gto['x'])
#     gto['y'] = np.flip(0.4 * np.exp(0.6 * gto['x']) + 2)
#     print('gtoy',gto['y'])

    # find the headings tangent to the path
    gt['h'] = []
    gt['h'].append(0)
    for i in range(1, len(gt['x'])):
        gt['h'].append(np.arctan2((gt['y'][i] - gt['y'][i - 1]), (gt['x'][i] - gt['x'][i - 1])))
    gt['h'] = np.array(gt['h'])

    print('h',gt['h'])

    # generate noise-free control inputs
    u = np.zeros([3, len(gt['x']) - 1])
    u[0, :] = np.diff(gt['x'])
    u[1, :] = np.diff(gt['y'])
    u[2, :] = np.diff(gt['h'])
    print('ur',u)

    # build a 2D robot
    robot = {}
    robot['dt'] = dt
    robot['n'] = 1000
    robot['x'] = []  # state mean
    robot['Cov_fo'] = np.zeros([3, 3])  # first order covariance propagation around mean
    for i in range(robot['n']):
        robot['x'].append(np.eye(3))
    # motion model noise covariance
    robot['Q'] = np.diag([0.03 ** 2, 0.03 ** 2, 0.1 ** 2])
    robot['Q2'] = np.diag([0.1 ** 2,0.03 ** 2, 0.03 ** 2])
    # Cholesky factor of covariance for sampling
    robot['L'] = np.linalg.cholesky(robot['Q'])
    # se(2) generators; twist = vec(v1, v2, omega)
    robot['G1'] = np.array([[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])
    robot['G2'] = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
    robot['G3'] = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 0]])
    # SE(2) Adjoint
    robot['Ad'] = adjoint
    robot['P'] = np.zeros([3, 3]) 
    robot['Xpred'] = np.eye(3)

    # construct noise free motion trajectory
    path = {}
    path['T'] = np.eye(3)
    path['x'] = []
    path['x'].append(0)
    path['y'] = []
    path['y'].append(0)
    for i in range(u.shape[1]):
        Ui = np.array([[np.cos(u[2, i]), -np.sin(u[2, i]), u[0, i]],
                       [np.sin(u[2, i]), np.cos(u[2, i]), u[1, i]],
                       [0, 0, 1]])
        path['T'] = np.dot(path['T'], Ui)
        path['x'].append(path['T'][0, 2])
        path['y'].append(path['T'][1, 2])
    # create confidence ellipse
    # first create points from a unit circle + angle (third dimension of so(3))
    phi = np.arange(-np.pi, np.pi + 0.1, 0.1).reshape(-1, 1)
    circle = np.array([np.cos(phi), np.sin(phi), np.zeros([len(phi), 1])])
    circle = circle.reshape(3, -1).T
    # print(circle)
    sigphi = np.arange(-np.pi, np.pi, np.pi/2).reshape(-1, 1)
    # print(sigphi)
    sigcircle = np.array([np.cos(sigphi), np.sin(sigphi), np.zeros([len(sigphi), 1])])
    sigcircle = sigcircle.reshape(3, -1).T
    # Chi-squared 3-DOF 95% confidence (0.05) : 7.815
    scale = np.sqrt(7.815)
    # print(sigcircle)

    # incremental visualization
    green = np.array([0.2980, 0.6, 0])
    crimson = np.array([220, 20, 60]) / 255
    darkblue = np.array([0, 0.2, 0.4])
    Darkgrey = np.array([0.25, 0.25, 0.25])
    VermillionRed = np.array([156, 31, 46]) / 255
    DupontGray = np.array([144, 131, 118]) / 255

    fig = plt.figure()
    plt.plot(path['x'], path['y'], '-', color=Darkgrey, linewidth=3, label='Normal path')
    plt.grid(True)
    # plt.axis('equal')
    # plt.xlim([-1, 6])
    # plt.ylim([-1, 5])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    pathb = [path['x'],path['y']]


    # extract propagated particles
    p = np.zeros([2, robot['n']])
    for i in range(robot['n']):
        p[0, i] = robot['x'][i][0, 2]
        p[1, i] = robot['x'][i][1, 2]

    # compute sample statistics
    mu, Sigma = Lie_sample_statistics(robot)

    # plot sample mean and particles
    # plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5, label='Samples')
    # plt.plot(mu[0, 2], mu[1, 2], 'o', color=crimson, markersize=6, label='Sample mean')

    # maim loop; iterate over the control inputs and move the robot
    ELLIPSE = np.zeros([circle.shape[0], 2])# covariance ellipse on manifold (nonlinear)
    ELLIPSE_P = np.zeros([circle.shape[0], 2])  
    ELLIPSE_fo = np.zeros([circle.shape[0], 2])  # first order covariance ellipse on maniford (nonlinear)
    ellipse = np.zeros([circle.shape[0], 2])  # covariance ellipse on Lie algebra
    show_label = True

    # for wasserstein check
    sigma_pts_wts = []
    particles = []
    mus = []
    ellipse_sig = []
    ellipse_fo = []



    for i in range(u.shape[1]):
        # move particles based on the input
        robot = propagation(robot, u[:, i])
        # extract propagated particles
        p = np.zeros([2, robot['n']])
        for j in range(robot['n']):
            p[0, j] = robot['x'][j][0, 2]
            p[1, j] = robot['x'][j][1, 2]

        

        # show particles
        # plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5)
    
        plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5)
        particles.append(p)

        # compute sample statistics
        mu, Sigma = Lie_sample_statistics(robot)

        mus.append(mu)
        # print(mu,"mu")



        # compute first order analytical covariance propagation
        Ui = np.array([[np.cos(u[2, i]), -np.sin(u[2, i]), u[0, i]],
                       [np.sin(u[2, i]), np.cos(u[2, i]), u[1, i]],
                       [0, 0, 1]])

        # left-invariant error: eta^L = X^-1 * X^hat
        # robot['Ad'](np.linalg.inv(Ui)) maps the covariance back to Lie algebra using the
        # incremental motion Ui (hence np.linalg.inv(Ui)). Then the noise covariance that
        # is already defined in Lie algebra can be added to the mapped state covariance
        robot['Cov_fo'] = np.dot(np.dot(robot['Ad'](np.linalg.inv(Ui)), robot['Cov_fo']),
                                 robot['Ad'](np.linalg.inv(Ui)).T) + robot['Q']
        robot['L_fo'] = np.linalg.cholesky(robot['Cov_fo'])

        # def sigma_points(self):
        # sigma points around the reference point
        n= 2
        kappa =2
        # L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(robot['Cov_fo']) # scaled Cholesky factor of P
        L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(np.eye(2))
        x = np.zeros(2).reshape(-1,1)
        # x[0] = robot['Xpred'][0,2]
        # x[1] = robot['Xpred'][1,2]
        # x[2] = np.arctan2(robot['Xpred'][1,0], robot['Xpred'][0,0])
        # print(np.linalg.cholesky(np.eye(2)))
        # x[0] = 0
        # x[1] = 0
        # x[2] = 0
        Y = x.repeat(len(x), axis=1)
        # print(Y,'Y')
        Sigpts = np.hstack((x, Y + L_sig, Y - L_sig))  # 2n+1 sigma points
        Sigpts = np.vstack((Sigpts, np.zeros(Sigpts.shape[1])))
        wts = np.zeros([2 * n + 1, 1])  # 2n+1 sigma points weights
        wts[0] = kappa / (n + kappa)
        wts[1:] = 1 / (2 * (n + kappa))
        # wts = wts.reshape(-1)
        Sigpts = Sigpts.T
        # print(Sigpts,'Sigpts')
        # pts = np.zeros([Sigpts.shape[0], 2])
        # for j in range(Sigpts.shape[0]):
        #     ell_se2_vec = scale * np.dot(robot['L_fo'], sigcircle[j, :])
        #     print(ell_se2_vec, 'Sigmapoints')
        #     temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
        #                                + robot['G3'] * ell_se2_vec[2]))
        #     pts[j,:]= np.array([temp[0, 2], temp[1, 2]])
            
        # print(robot['Xpred'])
        # scale * np.dot(L, circle[j, :])
        pts = np.zeros([Sigpts.shape[0], 2])
        pts2 = np.zeros([Sigpts.shape[0], 2])
        for j in range(Sigpts.shape[0]):
            # ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :].reshape(-1,1))
            # print(ell_se2_vec, 'Sigmapoints')
            ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :])
            print(ell_se2_vec, 'Sigmapoints2')
            temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
                                       + robot['G3'] * ell_se2_vec[2]))
            pts[j,:]= np.array([temp[0, 2], temp[1, 2]])
            temp_Matrix = np.array([[np.cos(ell_se2_vec[2]), -np.sin(ell_se2_vec[2]), ell_se2_vec[0]],
                                        [np.sin(ell_se2_vec[2]), np.cos(ell_se2_vec[2]), ell_se2_vec[1]],
                                        [0, 0, 1]])
            temp2 = np.dot(mu, temp_Matrix)
            pts2[j,:]= np.array([temp2[0, 2], temp2[1, 2]])


        # print(pts)
        # print(wts)
        # if i ==5:
        sigma_pts_wts.append(np.hstack((pts, wts)))
        
        # print(pts2)
        plt.plot(pts[:, 0], pts[:, 1], '.', color='tab:purple', alpha=0.7, markersize=5)
        
        # plt.plot(pts2[:, 0], pts2[:, 1], '.', color=crimson, alpha=0.7, markersize=5)
        # print(Sigma,'Sigma')
        # print(robot['Cov_fo'], 'Covfo')
        # print(robot['L_fo'])
        # print(robot['P'], 'P')
        # create the ellipse using the unit circle
        # if Sigma is positive definite, plot the ellipse
        
        if pos_def_matrix(Sigma):
            L = np.linalg.cholesky(Sigma)
            # Lp = np.linalg.cholesky(robot['P'])
            # print(Sigma, "Sigma")
            # print(robot['P'], "robot['P']")
            for j in range(circle.shape[0]):
                # sample covariance on SE(2）
                ell_se2_vec = scale * np.dot(L, circle[j, :])
                # retract and left-translate the ellipse on Lie algebra to SE(2) using LIe exp map
                temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
                                       + robot['G3'] * ell_se2_vec[2]))
                ELLIPSE[j, :] = np.array([temp[0, 2], temp[1, 2]])

                # sample covariance on SE(2）
                # ell_se2_vec = scale * np.dot(Lp, circle[j, :])
                # # retract and left-translate the ellipse on Lie algebra to SE(2) using LIe exp map
                # temp = np.dot(robot['Xpred'], expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
                #                        + robot['G3'] * ell_se2_vec[2]))
                # ELLIPSE_P[j, :] = np.array([temp[0, 2], temp[1, 2]])

                # gert the ellipse on LIe algebra
                ell_se2_vec = scale * np.dot(robot['L_fo'], circle[j, :])
                temp_Matrix = np.array([[np.cos(ell_se2_vec[2]), -np.sin(ell_se2_vec[2]), ell_se2_vec[0]],
                                        [np.sin(ell_se2_vec[2]), np.cos(ell_se2_vec[2]), ell_se2_vec[1]],
                                        [0, 0, 1]])
                temp = np.dot(mu, temp_Matrix)
                ellipse[j, :] = np.array([temp[0, 2], temp[1, 2]])

                # sample covariance on SE(2)
                ell_se2_vec = scale * np.dot(robot['L_fo'], circle[j, :])
                # print(ell_se2_vec, 'pts ell cov_fo')
                # retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
                temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
                                       + robot['G3'] * ell_se2_vec[2]))
                ELLIPSE_fo[j, :] = np.array([temp[0, 2], temp[1, 2]])

            # if i == 5:
            ellipse_sig.append(ELLIPSE)
            ellipse_fo.append(ELLIPSE_fo)
            if show_label:
                plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2,
                         label='Sample covariance - SE(2)')
                # plt.plot(ELLIPSE_P[:, 0], ELLIPSE_P[:, 1], color=green, alpha=0.7, linewidth=2,
                #          label='inv covariance - SE(2)')        
                plt.plot(ELLIPSE_fo[:, 0], ELLIPSE_fo[:, 1], color=darkblue, alpha=0.7, linewidth=2,
                         label='First-order covariance - SE(2)')
                plt.plot(ellipse[:, 0], ellipse[:, 1], color=DupontGray, alpha=0.7, linewidth=2,
                         label='Covariance - Lie Algebra')
                show_label = False
            else:

                plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2)
                # plt.plot(ELLIPSE_P[:, 0], ELLIPSE_P[:, 1], color=green, alpha=0.7, linewidth=2)
                plt.plot(ELLIPSE_fo[:, 0], ELLIPSE_fo[:, 1], color=darkblue, alpha=0.7, linewidth=2)
                plt.plot(ellipse[:, 0], ellipse[:, 1], color=DupontGray, alpha=0.7, linewidth=2)
        plt.plot(mu[0, 2], mu[1, 2], 'o', color=green, alpha=0.7, markersize=6)
        plt.legend()
        plt.pause(0.05)

#  print(particles,'particles')
    # print(sigma_pts_wts,'sigma_pts_wts')

    datarob = [particles] + [sigma_pts_wts] +[ellipse_sig] +[ellipse_fo] + [mus] +[pathb]

    with open("datarob", "wb") as fp:   #Pickling
      pickle.dump(datarob, fp)



#     dt = 1
#     gt = {}
#     gt['x'] = np.arange( 6 + dt,0, -dt)
#     print('gtx',gt['x'])
#     gt['y'] = np.flip(2 * np.log( 1.6* gt['x']) + 1)
#     print('gty',gt['y'])

#     plt.plot(gt['x'], gt['y'], '-', color='g', linewidth=3, label='Normal path')

#     # find the headings tangent to the path
#     gt['h'] = []
#     gt['h'].append(np.pi/2)
#     for i in range(1, len(gt['x'])):
#         gt['h'].append(np.arctan2(gt['y'][i] - gt['y'][i - 1], gt['x'][i] - gt['x'][i - 1]))
#     gt['h'] = np.array(gt['h'])

#     print('h',gt['h'])

#     # generate noise-free control inputs
#     u = np.zeros([3, len(gt['x']) - 1])
#     u[0, :] = np.diff(gt['x'])
#     u[1, :] = np.diff(gt['y'])
#     u[2, :] = np.diff(gt['h'])
#     print('u',u)


#     # build a 2D robot
#     robot = {}
#     robot['dt'] = dt
#     robot['n'] = 1000
#     robot['x'] = []  # state mean
#     robot['Cov_fo'] = np.zeros([3, 3])  # first order covariance propagation around mean
#     for i in range(robot['n']):
#         robot['x'].append(np.diag([7.6,2.5,1]))
        
#     # motion model noise covariance
#     robot['Q'] = np.diag([0.03 ** 2, 0.03 ** 2, 0.1 ** 2])
#     robot['Q2'] = np.diag([0.1 ** 2,0.03 ** 2, 0.03 ** 2])
#     # Cholesky factor of covariance for sampling
#     robot['L'] = np.linalg.cholesky(robot['Q'])
#     # se(2) generators; twist = vec(v1, v2, omega)
#     robot['G1'] = np.array([[0, 0, 1],
#                             [0, 0, 0],
#                             [0, 0, 0]])
#     robot['G2'] = np.array([[0, 0, 0],
#                             [0, 0, 1],
#                             [0, 0, 0]])
#     robot['G3'] = np.array([[0, -1, 0],
#                             [1, 0, 0],
#                             [0, 0, 0]])
#     # SE(2) Adjoint
#     robot['Ad'] = adjoint
#     robot['P'] = np.zeros([3, 3]) 
#     robot['Xpred'] = np.diag([7.6,2.5,1])

#     # construct noise free motion trajectory
#     path = {}
#     path['T'] = pose_mat(np.array([0,0,np.pi/2]))
#     print('T', path['T'])
#     path['x'] = []
#     path['x'].append(7)
#     path['y'] = []
#     path['y'].append(1.94)
#     for i in range(u.shape[1]):
#         Ui = np.array([[np.cos(u[2, i]), -np.sin(u[2, i]), u[0, i]],
#                        [np.sin(u[2, i]), np.cos(u[2, i]), u[1, i]],
#                        [0, 0, 1]])
#         path['T'] = np.dot(path['T'], Ui)
#         path['x'].append(path['T'][0, 2])
#         path['y'].append(path['T'][1, 2])

#     print(path['x'])

#     # create confidence ellipse
#     # first create points from a unit circle + angle (third dimension of so(3))
#     phi = np.arange(-np.pi, np.pi + 0.1, 0.1).reshape(-1, 1)
#     circle = np.array([np.cos(phi), np.sin(phi), np.zeros([len(phi), 1])])
#     circle = circle.reshape(3, -1).T
#     # print(circle)
#     sigphi = np.arange(-np.pi, np.pi, np.pi/2).reshape(-1, 1)
#     # print(sigphi)
#     sigcircle = np.array([np.cos(sigphi), np.sin(sigphi), np.zeros([len(sigphi), 1])])
#     sigcircle = sigcircle.reshape(3, -1).T
#     # Chi-squared 3-DOF 95% confidence (0.05) : 7.815
#     scale = np.sqrt(7.815)
#     # print(sigcircle)
# # [[4. 2. 5. 7.]
#     # incremental visualization
#     green = np.array([0.2980, 0.6, 0])
#     crimson = np.array([220, 20, 60]) / 255
#     darkblue = np.array([0, 0.2, 0.4])
#     Darkgrey = np.array([0.25, 0.25, 0.25])
#     VermillionRed = np.array([156, 31, 46]) / 255
#     DupontGray = np.array([144, 131, 118]) / 255

#     fig = plt.figure()
#     plt.plot(path['x'], path['y'], '-', color=green, linewidth=3, label='Normal path')
#     plt.grid(True)
#     # plt.axis('equal')
#     # plt.xlim([-1, 6])
#     # plt.ylim([-1, 5])
#     plt.xlabel(r'$x_1$')
#     plt.ylabel(r'$x_2$')

#     # extract propagated particles
#     p = np.zeros([2, robot['n']])
#     for i in range(robot['n']):
#         p[0, i] = robot['x'][i][0, 2]
#         p[1, i] = robot['x'][i][1, 2]

#     # compute sample statistics
#     mu, Sigma = Lie_sample_statistics(robot)

#     # plot sample mean and particles
#     # plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5, label='Samples')
#     # plt.plot(mu[0, 2], mu[1, 2], 'o', color=crimson, markersize=6, label='Sample mean')

#     # maim loop; iterate over the control inputs and move the robot
#     ELLIPSE = np.zeros([circle.shape[0], 2])# covariance ellipse on manifold (nonlinear)
#     ELLIPSE_P = np.zeros([circle.shape[0], 2])  
#     ELLIPSE_fo = np.zeros([circle.shape[0], 2])  # first order covariance ellipse on maniford (nonlinear)
#     ellipse = np.zeros([circle.shape[0], 2])  # covariance ellipse on Lie algebra
#     show_label = True

#     # for wasserstein check
#     sigma_pts_wts = []
#     particles = []


#     for i in range(u.shape[1]):
#         # move particles based on the input
#         robot = propagation(robot, u[:, i])
#         # extract propagated particles
#         p = np.zeros([2, robot['n']])
#         for j in range(robot['n']):
#             p[0, j] = robot['x'][j][0, 2]
#             p[1, j] = robot['x'][j][1, 2]

        

#         # show particles
#         # plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5)
#         particles.append(p)

#         # compute sample statistics
#         mu, Sigma = Lie_sample_statistics(robot)
#         # print(mu,"mu")



#         # compute first order analytical covariance propagation
#         Ui = np.array([[np.cos(u[2, i]), -np.sin(u[2, i]), u[0, i]],
#                        [np.sin(u[2, i]), np.cos(u[2, i]), u[1, i]],
#                        [0, 0, 1]])

#         # left-invariant error: eta^L = X^-1 * X^hat
#         # robot['Ad'](np.linalg.inv(Ui)) maps the covariance back to Lie algebra using the
#         # incremental motion Ui (hence np.linalg.inv(Ui)). Then the noise covariance that
#         # is already defined in Lie algebra can be added to the mapped state covariance
#         robot['Cov_fo'] = np.dot(np.dot(robot['Ad'](np.linalg.inv(Ui)), robot['Cov_fo']),
#                                  robot['Ad'](np.linalg.inv(Ui)).T) + robot['Q']
#         robot['L_fo'] = np.linalg.cholesky(robot['Cov_fo'])

#         # def sigma_points(self):
#         # sigma points around the reference point
#         n= 2
#         kappa =2
#         # L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(robot['Cov_fo']) # scaled Cholesky factor of P
#         L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(np.eye(2))
#         x = np.zeros(2).reshape(-1,1)
#         # x[0] = robot['Xpred'][0,2]
#         # x[1] = robot['Xpred'][1,2]
#         # x[2] = np.arctan2(robot['Xpred'][1,0], robot['Xpred'][0,0])
#         # print(np.linalg.cholesky(np.eye(2)))
#         # x[0] = 0
#         # x[1] = 0
#         # x[2] = 0
#         Y = x.repeat(len(x), axis=1)
#         # print(Y,'Y')
#         Sigpts = np.hstack((x, Y + L_sig, Y - L_sig))  # 2n+1 sigma points
#         Sigpts = np.vstack((Sigpts, np.zeros(Sigpts.shape[1])))
#         wts = np.zeros([2 * n + 1, 1])  # 2n+1 sigma points weights
#         wts[0] = kappa / (n + kappa)
#         wts[1:] = 1 / (2 * (n + kappa))
#         # wts = wts.reshape(-1)
#         Sigpts = Sigpts.T
#         # print(Sigpts,'Sigpts')
#         # pts = np.zeros([Sigpts.shape[0], 2])
#         # for j in range(Sigpts.shape[0]):
#         #     ell_se2_vec = scale * np.dot(robot['L_fo'], sigcircle[j, :])
#         #     print(ell_se2_vec, 'Sigmapoints')
#         #     temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
#         #                                + robot['G3'] * ell_se2_vec[2]))
#         #     pts[j,:]= np.array([temp[0, 2], temp[1, 2]])
            
#         # print(robot['Xpred'])
#         # scale * np.dot(L, circle[j, :])
#         pts = np.zeros([Sigpts.shape[0], 2])
#         pts2 = np.zeros([Sigpts.shape[0], 2])
#         for j in range(Sigpts.shape[0]):
#             # ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :].reshape(-1,1))
#             # print(ell_se2_vec, 'Sigmapoints')
#             ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :])
#             print(ell_se2_vec, 'Sigmapoints2')
#             temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
#                                        + robot['G3'] * ell_se2_vec[2]))
#             pts[j,:]= np.array([temp[0, 2], temp[1, 2]])
#             temp_Matrix = np.array([[np.cos(ell_se2_vec[2]), -np.sin(ell_se2_vec[2]), ell_se2_vec[0]],
#                                         [np.sin(ell_se2_vec[2]), np.cos(ell_se2_vec[2]), ell_se2_vec[1]],
#                                         [0, 0, 1]])
#             temp2 = np.dot(mu, temp_Matrix)
#             pts2[j,:]= np.array([temp2[0, 2], temp2[1, 2]])


#         # print(pts)
#         # print(wts)
#         sigma_pts_wts.append(np.hstack((pts, wts)))
        
#         # print(pts2)
#         plt.plot(pts[:, 0], pts[:, 1], '.', color='tab:orange', alpha=0.7, markersize=5)
        
#         # plt.plot(pts2[:, 0], pts2[:, 1], '.', color=crimson, alpha=0.7, markersize=5)
#         # print(Sigma,'Sigma')
#         # print(robot['Cov_fo'], 'Covfo')
#         # print(robot['L_fo'])
#         # print(robot['P'], 'P')
#         # create the ellipse using the unit circle
#         # if Sigma is positive definite, plot the ellipse
#         if pos_def_matrix(Sigma):
#             L = np.linalg.cholesky(Sigma)
#             # Lp = np.linalg.cholesky(robot['P'])
#             # print(Sigma, "Sigma")
#             # print(robot['P'], "robot['P']")
#             for j in range(circle.shape[0]):
#                 # sample covariance on SE(2）
#                 ell_se2_vec = scale * np.dot(L, circle[j, :])
#                 # retract and left-translate the ellipse on Lie algebra to SE(2) using LIe exp map
#                 temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
#                                        + robot['G3'] * ell_se2_vec[2]))
#                 ELLIPSE[j, :] = np.array([temp[0, 2], temp[1, 2]])

#                 # sample covariance on SE(2）
#                 # ell_se2_vec = scale * np.dot(Lp, circle[j, :])
#                 # # retract and left-translate the ellipse on Lie algebra to SE(2) using LIe exp map
#                 # temp = np.dot(robot['Xpred'], expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
#                 #                        + robot['G3'] * ell_se2_vec[2]))
#                 # ELLIPSE_P[j, :] = np.array([temp[0, 2], temp[1, 2]])

#                 # gert the ellipse on LIe algebra
#                 ell_se2_vec = scale * np.dot(robot['L_fo'], circle[j, :])
#                 temp_Matrix = np.array([[np.cos(ell_se2_vec[2]), -np.sin(ell_se2_vec[2]), ell_se2_vec[0]],
#                                         [np.sin(ell_se2_vec[2]), np.cos(ell_se2_vec[2]), ell_se2_vec[1]],
#                                         [0, 0, 1]])
#                 temp = np.dot(mu, temp_Matrix)
#                 ellipse[j, :] = np.array([temp[0, 2], temp[1, 2]])

#                 # sample covariance on SE(2)
#                 ell_se2_vec = scale * np.dot(robot['L_fo'], circle[j, :])
#                 # print(ell_se2_vec, 'pts ell cov_fo')
#                 # retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
#                 temp = np.dot(mu, expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
#                                        + robot['G3'] * ell_se2_vec[2]))
#                 ELLIPSE_fo[j, :] = np.array([temp[0, 2], temp[1, 2]])
#             if show_label:
#                 # plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2,
#                 #          label='Sample covariance - SE(2)')
#                 # plt.plot(ELLIPSE_P[:, 0], ELLIPSE_P[:, 1], color=green, alpha=0.7, linewidth=2,
#                 #          label='inv covariance - SE(2)')        
#                 plt.plot(ELLIPSE_fo[:, 0], ELLIPSE_fo[:, 1], color=crimson, alpha=0.7, linewidth=2,
#                          label='First-order covariance - SE(2)')
#                 plt.plot(ellipse[:, 0], ellipse[:, 1], color=DupontGray, alpha=0.7, linewidth=2,
#                          label='Covariance - Lie Algebra')
#                 show_label = False
#             else:
#                 # plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2)
#                 # plt.plot(ELLIPSE_P[:, 0], ELLIPSE_P[:, 1], color=green, alpha=0.7, linewidth=2)
#                 plt.plot(ELLIPSE_fo[:, 0], ELLIPSE_fo[:, 1], color=crimson, alpha=0.7, linewidth=2)
#                 plt.plot(ellipse[:, 0], ellipse[:, 1], color=DupontGray, alpha=0.7, linewidth=2)
#         plt.plot(mu[0, 2], mu[1, 2], 'o', color=green, alpha=0.7, markersize=6)
#         plt.legend()
#         plt.pause(0.05)








    
   

    # for i in range(len(sigma_pts_wts)-1):
    #     # print(wasserstein_distance(particles[i],particles[i+1]), 'Wasserstein distance particles', str(i))
    #     # print(wasserstein_distance(sigma_pts_wts[i][:,:2],sigma_pts_wts[i+1][:,:2],sigma_pts_wts[i][:,2],sigma_pts_wts[i+1][:,2]), 'Wasserstein distance sigma pts', str(i))
        
        # sig_time = time.time()
        # # W dis Sigma pts
        # u_sig = np.array(sigma_pts_wts[i][:,:2])
        # print(u_sig)
        # uwt_sig = sigma_pts_wts[i][:,2].reshape((-1,))
        # v_sig = np.array(sigma_pts_wts[i+1][:,:2])
        # vwt_sig = sigma_pts_wts[i+1][:,2].reshape((-1,))

        # print(np.shape(uwt_sig), vwt_sig)

        # M = ot.dist(u_sig,v_sig,metric='euclidean')
        # print(M)
        # G0 = ot.emd2(uwt_sig, vwt_sig, M)
        # print(G0, 'Wasserstein using Sigma', str(i))

        # print("Itr", str(i),"Time with Sigma", time.time()- sig_time )

        # par_time = time.time()
        # u_par = particles[i]
        # # uwt_par = parma_pts_wts[i][:,2].reshape((-1,))
        # v_par = particles[i+1]
        # # vwt_par = sigma_pts_wts[i+1][:,2].reshape((-1,))

        # M = ot.dist(u_par,v_par,metric='euclidean')
        # G0 = ot.emd2([], [], M)
        # print(G0, 'Wasserstein using particles', str(i))
        # print("Itr", str(i),"Time with Particles", time.time()- par_time )


    fig.savefig('banana_is_gaussian_iftf.png')
    plt.show()
