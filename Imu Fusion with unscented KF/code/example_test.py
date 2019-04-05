import numpy as np
import matplotlib.pyplot as plt
from estimate_rot_yan import estimate_rot
import scipy.io as sio


def angle2rot(roll, pitch, yaw):
    Rx=np.array([[1,0,0],[0,np.cos(roll),np.sin(roll)],[0,-np.sin(roll),np.cos(roll)]])
    Ry=np.array([[np.cos(pitch),0,-np.sin(pitch)],[0,1,0],[np.sin(pitch),0,np.cos(pitch)]])
    Rz=np.array([[np.cos(yaw),np.sin(yaw),0],[-np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    rot = np.dot(np.dot(Rz,Ry),Rx)
    return rot

def rotationMatrix2Euler(M):
	if M[2, 0] < 0.999 and M[2, 0] > -0.999:
		theta = -np.arcsin(M[2, 0])
		# theta = np.pi - theta1
		psi = np.arctan2(M[2, 1]/np.cos(theta), M[2, 2]/np.cos(theta))
		# psi = np.arctan2(M[2, 1]/np.cos(theta), M[2, 2]/np.cos(theta))
		phi = np.arctan2(M[1, 0]/np.cos(theta), M[0, 0]/np.cos(theta))
		# phi = np.arctan2(M[1, 0]/np.cos(theta), M[0, 0]/np.cos(theta))
	else:
		phi = 0
		if M[2, 0] == -1:
			theta = np.pi/2
			psi = phi + np.arctan2(M[0, 1], M[0, 2])
		else:
			theta = -np.pi/2
			psi = -phi + np.arctan2(M[0, 1], M[0, 2])
	return psi, theta, phi
def loadGroundTruth(data_num=1):
    vicon_contents = sio.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    mat_rots = vicon_contents['rots']
    mat_ts = vicon_contents['ts']
    mat_rots = mat_rots.astype(np.float64)
    roll = np.empty((mat_rots.shape[2], 1))
    pitch = np.empty((mat_rots.shape[2], 1))
    yaw = np.empty((mat_rots.shape[2], 1))
    for ind in range(mat_rots.shape[2]):
        roll[ind, 0], pitch[ind, 0], yaw[ind, 0] = rotationMatrix2Euler(mat_rots[:, :, ind])

    # plt.plot(roll)
    # plt.show()

    # plt.plot(pitch)
    # plt.show()

    # plt.plot(yaw)
    # plt.show()

    return roll, pitch, yaw, np.transpose(mat_ts)
if __name__ == "__main__":

    # roll,pitch,yaw = estimate_rot(3)
    # t = np.arange(0,roll.shape[0])
    # plt.plot(t, roll.T)
    # plt.ylabel('roll')
    # plt.show()
    # plt.plot(t, pitch.T)
    # plt.ylabel('pitch')
    # plt.show()
    # plt.plot(t, yaw.T)
    # plt.ylabel('yaw')
    # plt.show()
    data_num = 3
    roll, pitch, yaw = estimate_rot(data_num)
    truth_roll, truth_pitch, truth_yaw, truth_ts = loadGroundTruth(data_num)
    imu_data = sio.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    imu_ts = imu_data['ts']
    imu_ts = np.transpose(imu_ts)


    plt.subplot(3, 1, 1)
    plt.title('Filter Result and Ground Truth')
    plt.plot(truth_ts, truth_roll, imu_ts, roll)
    plt.legend(['GT', 'FR'], loc='upper right')
    plt.subplot(3, 1, 2)
    plt.plot(truth_ts, truth_pitch, imu_ts, pitch)
    plt.legend(['GT', 'FR'], loc='upper right')
    plt.subplot(3, 1, 3)
    plt.plot(truth_ts, truth_yaw, imu_ts, yaw)
    plt.legend(['GT', 'FR'], loc='upper right')

    plt.show()
    