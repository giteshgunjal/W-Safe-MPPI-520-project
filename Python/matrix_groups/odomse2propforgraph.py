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

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
import pickle


def adjoint(X):
    # SE(2) Adjoint function
    matrix = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
    matrix = np.vstack((matrix, np.array([0, 0, 1])))
    return matrix


def pos_def_matrix(A):
    # Check input matrix is positive definite or not
    return np.all(np.linalg.eigvals(A) > 0)


def propagation(robot, u):
    # SE(2) propagation model; each input is U \in SE(2) plus exp map of the
    # noise defined in Lie algebra
    for i in range(robot['n']):
        #  sample from a zero mean Gaussian
        noise = np.dot(robot['L'], np.random.randn(3, 1))
        N = robot['G1'] * noise[0] + robot['G2'] * noise[1] + robot['G3'] * noise[2]
        Ui = np.array([[np.cos(u[2]), -np.sin(u[2]), u[0]],
                       [np.sin(u[2]), np.cos(u[2]), u[1]],
                       [0, 0, 1]])
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


    dt = 0.04
    gt = {}
    gt['x'] = np.arange(0, 1.5 + dt, 1.5*dt)
    print(gt['x'])
    gt['y'] = np.zeros(gt['x'].shape)
    # generate a path
    # # generate a path
    # dt = 0.6
    # gt = {}
    # gt['x'] = np.arange(0, 4 + dt, dt)
    # print(gt['x'])
    # gt['y'] = 0.2 * np.exp(0.9 * gt['x']) - 0.1

    # # print('y',gt['x'])
    # dt = 0.6
    # gt = {}
    # gt['x'] = np.arange(0, 4 + dt, dt)
    # gt['y'] = 0.1 * np.exp(0.6 * gt['x']) - 0.1

    # find the headings tangent to the path
    gt['h'] = []
    gt['h'].append(0)
    for i in range(1, len(gt['x'])):
        gt['h'].append(np.arctan2(gt['y'][i] - gt['y'][i - 1], gt['x'][i] - gt['x'][i - 1]))
    gt['h'] = np.array(gt['h'])

    # generate noise-free control inputs
    u = np.zeros([3, len(gt['x']) - 1])
    u[0, :] = np.diff(gt['x'])
    u[1, :] = np.diff(gt['y'])
    u[2, :] = np.diff(gt['h'])

    # build a 2D robot
    robot = {}
    robot['dt'] = dt
    robot['n'] = 500
    robot['x'] = []  # state mean
    # robot['Cov_fo'] = np.diag([0.06 ** 2, 0.06 ** 2, 0.2 ** 2])  # first order covariance propagation around mean
    robot['Cov_fo'] = np.diag([0.01, 0.01, 0.02])  # first order covariance propagation around mean
    robot['L_fo'] = np.linalg.cholesky(robot['Cov_fo'])
    for i in range(robot['n']):
        robot['x'].append(np.eye(3))
    # motion model noise covariance
    robot['Q'] = np.diag([0.03 ** 2, 0.03 ** 2, 0.1 ** 2])
    # robot['Q'] = np.diag([0.01, 0.01, 0.01]) 
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

    # construct noise free motion trajectory
    path = {}
    path['X']=[]
    path['T'] = np.eye(3)
    path['X'].append(path['T'])
    path['x'] = []
    path['x'].append(0)
    path['y'] = []
    path['y'].append(0)
    for i in range(u.shape[1]):
        Ui = np.array([[np.cos(u[2, i]), -np.sin(u[2, i]), u[0, i]],
                       [np.sin(u[2, i]), np.cos(u[2, i]), u[1, i]],
                       [0, 0, 1]])
        path['T'] = np.dot(path['T'], Ui)
        path['X'].append(path['T'])
        path['x'].append(path['T'][0, 2])
        path['y'].append(path['T'][1, 2])

    del path['x'][-1]
    del path['y'][-1]

    # create confidence ellipse
    # first create points from a unit circle + angle (third dimension of so(3))
    phi = np.arange(-np.pi, np.pi + 0.1, 0.1).reshape(-1, 1)
    circle = np.array([np.cos(phi), np.sin(phi), np.zeros([len(phi), 1])])
    circle = circle.reshape(3, -1).T
    # Chi-squared 3-DOF 95% confidence (0.05) : 7.815
    scale = np.sqrt(7.815)

    sigphi = np.arange(-np.pi, np.pi, np.pi/2).reshape(-1, 1)
    # print(sigphi)
    sigcircle = np.array([np.cos(sigphi), np.sin(sigphi), np.zeros([len(sigphi), 1])])
    sigcircle = sigcircle.reshape(3, -1).T

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
    plt.axis('equal')
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
    print(mu, "mu")

    # plot sample mean and particles
    # plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5, label='Samples')
    # plt.plot(mu[0, 2], mu[1, 2], 'o', color=crimson, markersize=6, label='Sample mean')

    # maim loop; iterate over the control inputs and move the robot
    ELLIPSE = np.zeros([circle.shape[0], 2])  # covariance ellipse on manifold (nonlinear)
    ELLIPSE_fo = np.zeros([circle.shape[0], 2])  # first order covariance ellipse on maniford (nonlinear)
    ellipse = np.zeros([circle.shape[0], 2])  # covariance ellipse on Lie algebra
    show_label = True

    sigma_pts_wts = []
    particles = []
    mus = []
    ellipse_sig = []
    ellipse_fo = []


    for i in range(len(path['X'])-1):
        # move particles based on the input
        # robot = propagation(robot, u[:, i])
        # # extract propagated particles
        # p = np.zeros([2, robot['n']])
        # for j in range(robot['n']):
        #     p[0, j] = robot['x'][j][0, 2]
        #     p[1, j] = robot['x'][j][1, 2]

        # # show particles
        # plt.plot(p[0, :], p[1, :], '.', color=green, alpha=0.5, markersize=5)

        # particles.append(p)

        # # compute sample statistics
        # mu, Sigma = Lie_sample_statistics(robot)

        mus.append(path['X'][i])


        # compute first order analytical covariance propagation
        Ui = np.array([[np.cos(u[2, i]), -np.sin(u[2, i]), u[0, i]],
                       [np.sin(u[2, i]), np.cos(u[2, i]), u[1, i]],
                       [0, 0, 1]])

        

# def sigma_points(self):
        # sigma points around the reference point
        n= 2
        kappa =2
        # L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(robot['Cov_fo']) # scaled Cholesky factor of P
        L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(np.eye(2))
        x = np.zeros(2).reshape(-1,1)
        Y = x.repeat(len(x), axis=1)
        # print(Y,'Y')
        Sigpts = np.hstack((x, Y + L_sig, Y - L_sig))  # 2n+1 sigma points
        Sigpts = np.vstack((Sigpts, np.zeros(Sigpts.shape[1])))
        wts = np.zeros([2 * n + 1, 1])  # 2n+1 sigma points weights
        wts[0] = kappa / (n + kappa)
        wts[1:] = 1 / (2 * (n + kappa))
        # wts = wts.reshape(-1)
        Sigpts = Sigpts.T

        
        # scale * np.dot(L, circle[j, :])
        pts = np.zeros([Sigpts.shape[0], 2])
        pts2 = np.zeros([Sigpts.shape[0], 2])
        for j in range(Sigpts.shape[0]):
            # ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :].reshape(-1,1))
            # print(ell_se2_vec, 'Sigmapoints')
            ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :])
            print(ell_se2_vec, 'Sigmapoints2')
            k =robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1] + robot['G3'] * ell_se2_vec[2]
            temp = np.dot(path['X'][i], expm(k))
            print(expm(k), "expo expa")
            print(np.eye(3) + k  + (k@k)/2 + (k@k@k)/6, "expo app")
            pts[j,:]= np.array([temp[0, 2], temp[1, 2]])
            temp_Matrix = np.array([[np.cos(ell_se2_vec[2]), -np.sin(ell_se2_vec[2]), ell_se2_vec[0]],
                                        [np.sin(ell_se2_vec[2]), np.cos(ell_se2_vec[2]), ell_se2_vec[1]],
                                        [0, 0, 1]])
            temp2 = np.dot(path['X'][i], temp_Matrix)
            pts2[j,:]= np.array([temp2[0, 2], temp2[1, 2]])


        # print(pts)
        # print(wts)
        # if i ==5:
        sigma_pts_wts.append(np.hstack((pts, wts)))
        
        # print(pts2)
        # plt.plot(pts[:, 0], pts[:, 1], '.', color='blue', alpha=0.7, markersize=5)
        
        # plt.plot(pts2[:, 0], pts2[:, 1], '.', color='tab:green', alpha=0.7, markersize=5)
        # create the ellipse using the unit circle
        # if Sigma is positive definite, plot the ellipse
    # if pos_def_matrix(Sigma):
    #     L = np.linalg.cholesky(Sigma)
        for j in range(circle.shape[0]):
            # sample covariance on SE(2）
            # ell_se2_vec = scale * np.dot(L, circle[j, :])
            # # retract and left-translate the ellipse on Lie algebra to SE(2) using LIe exp map
            # temp = np.dot(path['X'][i], expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
            #                         + robot['G3'] * ell_se2_vec[2]))
            # ELLIPSE[j, :] = np.array([temp[0, 2], temp[1, 2]])

            # gert the ellipse on LIe algebra
            temp_Matrix = np.array([[np.cos(ell_se2_vec[2]), -np.sin(ell_se2_vec[2]), ell_se2_vec[0]],
                                    [np.sin(ell_se2_vec[2]), np.cos(ell_se2_vec[2]), ell_se2_vec[1]],
                                    [0, 0, 1]])
            temp = np.dot(path['X'][i], temp_Matrix)
            ellipse[j, :] = np.array([temp[0, 2], temp[1, 2]])

            # lie fo covariance on SE(2)
            ell_se2_vec = scale * np.dot(robot['L_fo'], circle[j, :])
            # retract and left-translate the ellipse on Lie algebra to SE(2) using Lie exp map
            temp = np.dot(path['X'][i], expm(robot['G1'] * ell_se2_vec[0] + robot['G2'] * ell_se2_vec[1]
                                    + robot['G3'] * ell_se2_vec[2]))
            ELLIPSE_fo[j, :] = np.array([temp[0, 2], temp[1, 2]])


        ellipse_sig.append(np.array(ELLIPSE))
        ellipse_fo.append(np.array(ELLIPSE_fo))
        # print(ELLIPSE_fo)    

        
        if show_label:
            # plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2,
            #             label='Sample covariance - SE(2)')
            plt.plot(ELLIPSE_fo[:, 0], ELLIPSE_fo[:, 1], color=darkblue, alpha=0.7, linewidth=2,
                        label='First-order covariance - SE(2)')
            plt.plot(ellipse[:, 0], ellipse[:, 1], color=DupontGray, alpha=0.7, linewidth=2,
                    label='Covariance - Lie Algebra')
            show_label = False
        else:
            # plt.plot(ELLIPSE[:, 0], ELLIPSE[:, 1], color=VermillionRed, alpha=0.7, linewidth=2)
            plt.plot(ELLIPSE_fo[:, 0], ELLIPSE_fo[:, 1], color=darkblue, alpha=0.7, linewidth=2)
            # plt.plot(ellipse[:, 0], ellipse[:, 1], color=DupontGray, alpha=0.7, linewidth=2)
    # plt.plot(path['X'][i][0, 2], path['X'][i][1, 2], 'o', color=crimson, alpha=0.7, markersize=4)

        # left-invariant error: eta^L = X^-1 * X^hat
        # robot['Ad'](np.linalg.inv(Ui)) maps the covariance back to Lie algebra using the
        # incremental motion Ui (hence np.linalg.inv(Ui)). Then the noise covariance that
        # is already defined in Lie algebra can be added to the mapped state covariance
        robot['Cov_fo'] = np.dot(np.dot(robot['Ad'](np.linalg.inv(Ui)), robot['Cov_fo']),
                                 robot['Ad'](np.linalg.inv(Ui)).T) + robot['Q']
        robot['L_fo'] = np.linalg.cholesky(robot['Cov_fo'])
    plt.legend()
    # plt.pause(0.05)
        
        

    # fig.savefig('banana_is_gaussian_particels.png')
    # print(ellipse_fo)
    plt.show()

    dataobs2 = [particles] + [sigma_pts_wts] +[ellipse_sig] +[ellipse_fo] + [mus] +[pathb]

    # with open("dataobs2", "wb") as fp:   #Pickling
    #   pickle.dump(dataobs2, fp)




