import numpy as np
from scipy import io
import math
import matplotlib.pyplot as plt
import sys


def getg(q, g):
    inv_q=np.concatenate(([q[0]], -q[1:])) / np.linalg.norm(q) ** 2
    res= quart_multiply(q, quart_multiply(np.concatenate(( [0], g )), inv_q ))[1:]
    return res
def matrix_sqrt(sigma_prev):
    a,b,c=np.linalg.svd(sigma_prev)

    return a.dot(np.diag(np.sqrt(b))).dot(c)

def vel2quat(velocity,t_delta):
    quat_vel=np.zeros([4,velocity.shape[1]])
    mag=np.linalg.norm(velocity,2,0)
    alp=mag*t_delta
    e=velocity/mag
    quat_vel[0,:]=np.cos(alp/2)
    quat_vel[1:4]=np.sin(alp/2)*e
    return quat_vel
def rpy2quat(oritation):
    r=oritation[0,:]/2
    p=oritation[1,:]/2
    y=oritation[2,:]/2
    quat=np.zeros([4,oritation.shape[1]])
    quat[0,:]  =np.cos(r)*np.cos(p)*np.cos(y)+np.sin(r)*np.sin(p)*np.sin(y)
    quat[1, :] =np.sin(r)*np.cos(p)*np.cos(y)-np.cos(r)*np.sin(p)*np.sin(y)
    quat[2, :] =np.cos(r)*np.sin(p)*np.cos(y)+np.sin(r)*np.cos(p)*np.sin(y)
    quat[3, :] =np.cos(r)*np.cos(p)*np.sin(y)-np.sin(r)*np.sin(p)*np.cos(y)
    return quat
def quat2rpy(quat_mul):
    rpy=np.zeros(3)
    w = quat_mul[0]
    x = quat_mul[1]
    y = quat_mul[2]
    z = quat_mul[3]
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(2 * (w * y - x * z))
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    rpy=np.array([r,p,y])
    return rpy


def quart_multiply(quart1, quart2):
    quart3 = np.zeros(4)
    quart3[0]=quart1[0] * quart2[0] - np.dot(quart1[1:], quart2[1:])
    quart3[1:]= quart1[0] * quart2[1:] + quart2[0] * quart1[1:] + np.cross(quart1[1:], quart2[1:])
    return quart3
def multi_quat(quat,quat_vel):
    quat_mul=np.zeros([4,quat.shape[1]])
    quat_mul[0,:]=quat[0,:]*quat_vel[0,:]-np.sum(quat[1:,:]*quat_vel[1:,:],0)
    quat_mul[1:,:]=quat[0,:]*quat_vel[1:,:]+quat_vel[0,:]*quat[1:,:]+np.cross(quat[1:,:].T,quat_vel[1:,:].T).T
    return quat_mul

# this part of code is from Internet
def exp(q):
    normq = np.linalg.norm(q[1:])
    if normq < 0.000000001:
        save = np.zeros(3)
    else:
        save = q[1:] / normq * np.sin(normq)
    return np.exp(q[0]) * np.concatenate(([np.cos(normq)], save))
def log(q):
    normq = np.linalg.norm(q)
    normqq = np.linalg.norm(q[1:])
    if normqq < 1.0e-9:
        save = np.zeros(3)
    else:
        save = q[1:] / normqq * np.arccos( q[0] / normq )
    return np.concatenate((
        [np.log( normq)],save
    ))
def quat_aver(points_mu, w):

    qt = points_mu[:, 0]

    while True:
        es = np.empty((points_mu.shape[1], 3))
        for ind in range(points_mu.shape[1]):
            q = points_mu[:, ind]
            cur=np.concatenate(([qt[0]], -qt[1:])) / np.linalg.norm(qt) ** 2
            qe = quart_multiply(cur, q)
            e = 2 * log(qe)[1:]
            norm_e = np.linalg.norm(e)
            if norm_e < 1.0e-3:
                es[ind, :] = np.zeros((3,))
            else:
                es[ind, :] = (-np.pi + (norm_e + np.pi) % (2 * np.pi)) * e / norm_e
        ev = np.dot(w, es)
        qt_new = quart_multiply(qt, exp(np.concatenate(([0], ev / 2))))
        if np.linalg.norm(ev) < 1.0e-3:
            return qt_new, np.transpose(es)
        qt = qt_new

##

def rpy2acc(rpy):
    acc=np.zeros(rpy.shape)
    for i in range(rpy.shape[1]):
        a = rpy[0][i]
        b = rpy[1][i]
        c = rpy[2][i]
        acc[:,i]=[np.cos(a)*np. sin(b) * np.cos(c) + np.sin(a) * np.sin(c),
        np. sin(a)* np.sin(b) *np. cos(c) -np. cos(a)*np. sin(c),
        np. cos(b) *np. cos(c)]
    return acc
def z_update(new_points_mu):
    z_points = np.zeros(new_points_mu.shape)
    z_points[3:, :] = new_points_mu[3:, :]
    acc=rpy2acc(new_points_mu[:3,:])
    z_points[:3,:]=acc
    return z_points
def ukf(t_delta,mu_prev,sigma_prev,val):
    R = np.diag(np.ones(sigma_prev.shape[0]) )*106
    Q = np.diag(np.ones(mu_prev.shape[0]-1))*1000
    g = np.array([0, 0, -1])
    n=3
    alpha=0
    beta=1
    k=10
    lamda=alpha*alpha*(n+k)-n
    lamda=0
    pred_points = np.zeros([3, 2 * n + 1])
    points_change=np.zeros([3,2*n+1])
    point_mu_change=np.linalg.cholesky(2 * n * (sigma_prev + R))
    points_change[:,1:n+1]=point_mu_change
    points_change[:, n+1:] = - point_mu_change
    points_mu=np.zeros([4,2*n+1])
    for i in range(2*n+1):
        a=exp(np.concatenate(([0], points_change[:,i] / 2)))
        b=exp(np.concatenate(([0], val[3:]*t_delta / 2)))

        points_mu[:,i]=quart_multiply(quart_multiply(mu_prev,a),b)
    #weights
    w_c = np.tile(1/2/(n+lamda),2 * n + 1)
    w_m=np.tile(1/2/(n+lamda),2 * n + 1)
    w_m[0]=lamda/(lamda+n)
    w_c[0]=w_m[0]+1-alpha*alpha+beta

    mu_dot,mu_dist=quat_aver( points_mu,w_m)

    sigma_dot=(w_c*mu_dist).dot(mu_dist.T)

    #correction


    for i in range(2*n+1):
        pred_points[:,i]=getg(points_mu[:,i],g)
    pred_mean=np.sum(pred_points*w_m,1)

    sigma_measure=(w_c*(pred_points-pred_mean.reshape([3,1]))).dot((pred_points-pred_mean.reshape([3,1])).T)
    sigma_cross = (w_c*mu_dist).dot((pred_points-pred_mean.reshape([3,1])).T)


    update_sigma = sigma_measure + Q
    gain = np.dot(sigma_cross, np.linalg.inv(update_sigma))
    mu_t = quart_multiply(mu_dot, exp(np.concatenate(([0], np.dot(gain, pred_mean - val[:3]))) / 2))
    sigma_t = sigma_dot - np.dot(np.dot(gain, sigma_cross), np.transpose(gain))

    return mu_t, sigma_t



def estimate_rot(num=1):
    imu_data=io.loadmat('imu/imuRaw'+ str(num)+'.mat')
    imu_ts = imu_data['ts']
    imu_vals = imu_data['vals'].astype(np.float)
    imu_vals=imu_vals[[0,1,2,4,5,3],:]
    scaler_acc, scaler_omg = 10, 3.4
    scale = np.zeros((6, 1))
    scale[:, 0] = 3300 / 1023 / np.array([-scaler_acc , -scaler_acc , scaler_acc ,scaler_omg / np.pi * 180, scaler_omg / np.pi * 180, scaler_omg / np.pi * 180])
    for i in range(imu_vals.shape[0]):
        imu_vals[i, :] = imu_vals[i, :] - np.mean(imu_vals[i, 0:100])
    imu_vals=imu_vals*scale


    mu_prev=np.array([1,0,0,0])
    sigma_prev = np.diag((1, 1, 1))*0.1
    state=np.zeros([3,imu_vals.shape[1]])
    state[:,0]=[0,0,0]
    for i in range(10,imu_ts.shape[1]):
        print(i)
        t_delta=imu_ts[0][i]-imu_ts[0][i-1]
        val=imu_vals[:,i]
        mu_prev,sigma_prev=ukf(t_delta,mu_prev,sigma_prev,val)
        state[:,i]=quat2rpy(mu_prev)

    return state[0,:],state[1,:],state[2,:]




