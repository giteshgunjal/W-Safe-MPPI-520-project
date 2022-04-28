from calendar import c
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import logm, expm
from scipy.stats import wasserstein_distance
# import ot
import pickle
import time
 



c = np.diag([0.03 ** 2, 0.01 ** 2])
print(c)
L = np.linalg.cholesky(c)
print(L)
with open("/home/gitesh/Desktop/520project/dataobs2", "rb") as fp:   # Unpickling
    dataobs = pickle.load(fp)

with open("/home/gitesh/Desktop/520project/datarob2", "rb") as fp:   # Unpickling
    datarob = pickle.load(fp)

sigma_pts_wts = datarob[1]
particles = datarob[0]

sigma_pts_wtsobs = dataobs[1][4]
particlesobs = dataobs[0][4]
print(sigma_pts_wtsobs)
# print
# print(particlesobs)
sigmaptsobs = [[[0.95999795, 0.99999998],
       [1.04485076, 0.9999998 ],
       [0.95939818, 1.08506181],
       [0.87514513, 1.00000016],
       [0.95939782, 0.91493815]],[[0.98999589, 0.99999989],
       [1.0748487 , 0.99999953],
       [0.9893963 , 1.08506173],
       [0.90514308, 1.00000025],
       [0.98939558, 0.91493807]],[[1.01999384, 0.99999975],
       [1.10484665, 0.9999992 ],
       [1.01939443, 1.08506158],
       [0.93514102, 1.00000029],
       [1.01939334, 0.91493792]],[[1.04999178, 0.99999953],
       [1.13484459, 0.99999881],
       [1.04939255, 1.08506137],
       [0.96513897, 1.00000025],
       [1.04939111, 0.91493771]],[[1.07998973, 0.99999926],
       [1.16484254, 0.99999835],
       [1.07939068, 1.08506109],
       [0.99513691, 1.00000016],
       [1.07938887, 0.91493743]],[[1.10998767, 0.99999892],
       [1.19484048, 0.99999783],
       [1.10938881, 1.08506076],
       [1.02513486, 1.        ],
       [1.10938664, 0.91493709]]]

sigmaptsrob =  [[[0.83626303, 0.48355327],
       [0.88617956, 0.55217149],
       [0.76550079, 0.53295664],
       [0.78640162, 0.414895  ],
       [0.90517257, 0.43159692]],[[0.88435927, 0.54982306],
       [0.93389277, 0.61871745],
       [0.81326615, 0.59894604],
       [0.83481454, 0.48093674],
       [0.95353999, 0.49804154]],[[0.92414086, 0.60569005],
       [0.97236186, 0.67550962],
       [0.85267574, 0.65344171],
       [0.87588877, 0.53589198],
       [0.99404595, 0.55568242]],[[0.97542202, 0.67890691],
       [1.02520743, 0.74762   ],
       [0.90412373, 0.72825295],
       [0.92568014, 0.61016231],
       [1.04462584, 0.62666411]],[[1.02811615, 0.75069444],
       [1.07878932, 0.81875503],
       [0.95750786, 0.80099955],
       [0.97746119, 0.6826203 ],
       [1.09659875, 0.69753179]],[[1.08133742, 0.82171262],
       [1.13319551, 0.88887509],
       [1.01161147, 0.87321238],
       [1.02951732, 0.75452083],
       [1.14889682, 0.76740184]]]



for i in range(1):
    plt.plot(np.array(sigmaptsobs[i])[:, 0], np.array(sigmaptsobs[i])[:, 1], '.', color='blue', alpha=0.7, markersize=5)
    plt.plot(np.array(sigmaptsrob[i])[:, 0], np.array(sigmaptsrob[i])[:, 1], '.', color='blue', alpha=0.7, markersize=5)

plt.show()
c
for i in range(len(sigma_pts_wts)):
    # print(wasserstein_distance(particles[i],particles[i+1]), 'Wasserstein distance particles', str(i))
    # print(wasserstein_distance(sigma_pts_wts[i][:,:2],sigma_pts_wts[i+1][:,:2],sigma_pts_wts[i][:,2],sigma_pts_wts[i+1][:,2]), 'Wasserstein distance sigma pts', str(i))
    
    sig_time = time.time()
    # W dis Sigma pts
    u_sig = np.array(sigma_pts_wts[i][:,:2])
    # print(u_sig)
    uwt_sig = np.array(sigma_pts_wts[i][:,2].reshape((-1,)))
    v_sig = np.array(sigma_pts_wtsobs[:,:2])
    vwt_sig = np.array(sigma_pts_wtsobs[:,2].reshape((-1,)))

    print(np.shape(uwt_sig), vwt_sig)

    M = ot.dist(u_sig,v_sig,metric='sqeuclidean')
    # print(M)
    G0 = ot.emd2(uwt_sig, vwt_sig, M)
    print(G0, 'Wasserstein using Sigma', str(i))

    print("Itr", str(i),"Time with Sigma", time.time()- sig_time )

    par_time = time.time()
    u_par = particles[i].T
    # print(np.shape(u_par))
    # uwt_par = parma_pts_wts[i][:,2].reshape((-1,))
    v_par = particlesobs.T
    # vwt_par = sigma_pts_wts[i+1][:,2].reshape((-1,))

    M = ot.dist(u_par,v_par,metric='sqeuclidean')
    # print(M)
    G0 = ot.emd2([], [], M)
    print(G0, 'Wasserstein using particles', str(i))
    print("Itr", str(i),"Time with Particles", time.time()- par_time )




# ploting
# incremental visualization
green = np.array([0.2980, 0.6, 0])
crimson = np.array([220, 20, 60]) / 255
darkblue = np.array([0, 0.2, 0.4])
Darkgrey = np.array([0.25, 0.25, 0.25])
VermillionRed = np.array([156, 31, 46]) / 255
DupontGray = np.array([144, 131, 118]) / 255

path = datarob[5]
mus = datarob[4]
ELLIPSE_fo = datarob[3]
ELLIPSE= datarob[2]

pathobs = dataobs[5]
musobs = dataobs[4][4]
ELLIPSE_foobs = dataobs[3]
ELLIPSEobs= dataobs[2]

print(ELLIPSE_fo)
# print(path)

print(mus)
plt.plot(path[0], path[1], '-', color=Darkgrey, linewidth=3, label='Normal path')

for i in range(len(mus)):

    plt.plot(mus[i][0, 2], mus[i][1, 2], 'o', color=green, alpha=0.7, markersize=6)
    plt.plot(ELLIPSE_fo[i][:, 0], ELLIPSE_fo[i][:, 1], color=darkblue, alpha=0.7, linewidth=2)
    plt.plot(sigma_pts_wts[i][:, 0], sigma_pts_wts[i][:, 1], '.', color='blue', alpha=0.7, markersize=5)

    x_values = [mus[i][0, 2], musobs[0, 2]]
    y_values = [mus[i][1, 2], musobs[1, 2]]
    plt.plot(x_values, y_values, 'tab:green', linestyle="--")
    text = 'Pair' + str(i+1)
    plt.text(mus[i][0, 2]-0.15, mus[i][1, 2]+0.35, text)
    # plt.text(point2[0]-0.050, point2[1]-0.25, "Point2")
    # plt.plot(ELLIPSE[i][:, 0], ELLIPSE[i][:, 1], color=VermillionRed, alpha=0.7, linewidth=2)

# print(ELLIPSE_foobs[5])
plt.plot(np.array(ELLIPSE_foobs[4])[:, 0], np.array(ELLIPSE_foobs[4])[:, 1], color=VermillionRed, alpha=0.7, linewidth=2)
plt.plot(sigma_pts_wtsobs[:, 0], sigma_pts_wtsobs[:, 1], '.', color='tab:orange', alpha=0.7, markersize=5)

plt.title("Wasserstein Distance")
plt.grid(True)
# plt.axis('equal')
# plt.xlim([-1, 6])
# plt.ylim([-1, 5])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.show()


    # fucnt Wcheck
    #State- Dict with mean and cov
    # State - [mean,cov, Qr]
    #Obs - Dict with ids and mean and cov
    #obs[key=id]= [mu, cov, Qo]
    # u_t_rob = [u and t]
    # u_t_obs - Dict [key =id ]= [u]
def wedge(x):
    # wedge operation for se(2) to put an R^3 vector to the Lie algebra basis ( v,w)
    G1 = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
    G2 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 0]])
    G3 = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 0]])
    xhat = G1 * x[0] + G2 * x[1] + G3 * x[2]
    return xhat


def GeneralSigmapts():
    n= 2
    kappa =2
    L_sig = np.sqrt(n + kappa) * np.linalg.cholesky(np.eye(2))
    x = np.zeros(2).reshape(-1,1)
    Y = x.repeat(len(x), axis=1)
    # print(Y,'Y')
    Sigpts = np.hstack((x, Y + L_sig, Y - L_sig))  # 2n+1 sigma points
    Sigpts = np.vstack((Sigpts, np.zeros(Sigpts.shape[1])))
    wts = np.zeros([2 * n + 1, 1])  # 2n+1 sigma points weights
    wts[0] = kappa / (n + kappa)
    wts[1:] = 1 / (2 * (n + kappa))
    Sigpts = Sigpts.T
    return Sigpts, wts

def adjoint(X):
        # SE(2) Adjoint function
        matrix = np.hstack((X[0:2, 0:2], np.array([[X[1, 2]], [-X[0, 2]]])))
        matrix = np.vstack((matrix, np.array([0, 0, 1])))
        return matrix

def pose_mat( X):
    x = X[0]
    y = X[1]
    h = X[2]
    H = np.array([[np.cos(h),-np.sin(h),x],\
                [np.sin(h),np.cos(h),y],\
                [0,0,1]])
    return H
def pose2vec(H):
    state_vector = np.zeros(3)  
    state_vector[0] = H[0,2]
    state_vector[1] = H[1,2]
    state_vector[2] = np.arctan2(H[1,0], H[0,0])
    return state_vector

#  for SE2
def Wcheck(State,M_r, Obs,M_o, u_t_rob, u_t_obs, theta, sigpts, wts):
    safe = True

    

    def prop_sigmapts(Hpred, cov, Sigpts):
        pts = np.zeros([Sigpts.shape[0], 2])
        for j in range(Sigpts.shape[0]):
            # ell_se2_vec = np.dot(robot['L_fo'], Sigpts[j, :].reshape(-1,1))
            # print(ell_se2_vec, 'Sigmapoints')
            ell_se2_vec = np.dot(np.linalg.cholesky(cov), Sigpts[j, :])
            print(ell_se2_vec, 'Sigmapoints2')
            temp = np.dot(Hpred, expm(wedge(ell_se2_vec)))
            pts[j,:]= np.array([temp[0, 2], temp[1, 2]])
        return pts

    def wdis(u_sig, v_sig, wts):
        M = ot.dist(u_sig,v_sig,metric='euclidean')
        G0 = ot.emd2(wts, wts, M)
        return G0
        

    

    dt = 0.1
    ur = u_t_rob[0]
    tr= u_t_rob[1]
    state_cov = State[1]
    state_mu = State[0]
    Qr = State[2]
    mu_prev = state_mu
    cov_prev = state_cov
    Obs_temp = Obs

    for i in range (tr//dt) :

        # propogate robot
        # t = (i+1)*0.1

        mu_pred = M_r(mu_prev,ur,dt)
        Xprev = pose_mat(mu_prev)
        Xpred = pose_mat(mu_pred)

        Ur  = logm(np.linalg.inv(Xprev) @ Xpred)
        # Ur = expm((G1 * ur[0] + G2 * ur[1]
        #                            + G3 * ur[2])*t)

        cov_pred = adjoint(np.linalg.inv(Ur))@ cov_prev @ adjoint(np.linalg.inv(Ur)).T + Qr


        # prop sigma pts
        Xpts = prop_sigmapts(Xpred, cov_pred, sigpts)

        # propogate Obs

        for j in range(len(Obs_temp.keys())):
            Omu_prev = Obs_temp[j][0]
            Ocov_prev = Obs_temp[j][1]
            Qo = Obs_temp[j][2]

            Oprev = pose_mat(Omu_prev)

            uo = u_t_obs[j][0]
            Omu_pred = M_o(Omu_prev,uo,dt)
            Opred = pose_mat(Omu_pred)

            Uo  = logm(np.linalg.inv(Oprev) @ Opred)
            Ocov_pred = adjoint(np.linalg.inv(Uo))@ Ocov_prev @ adjoint(np.linalg.inv(Uo)).T + Qo

            Opts = prop_sigmapts(Opred, Ocov_pred, sigpts)

            W = wdis(Xpts, Opts, wts)

            if W<theta :
                safe = False
                return safe

            Obs_temp[j][0]= Omu_pred
            Obs_temp[j][1] = Ocov_pred

        mu_prev = mu_pred
        cov_prev = cov_pred
    
    return safe




    def M(mu,u,dt):
        X = pose_mat(mu)
        Xnext = X @expm(wedge(u)*dt)
        return pose2vec(Xnext)






            
                

        



