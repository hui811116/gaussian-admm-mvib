import numpy as np
import sys

# saved matrix calculator
def calcmatPz1(A2,A21,cov_x2z1,cov_z1_inv):
	return A2@cov_x2z1 @ cov_z1_inv + A21

def calcmatKz2(cov_yx2,cov_yz1,inv_z1,cov_z1x2,A2,eps_2_inv):
	return (cov_yx2-cov_yz1@inv_z1@cov_z1x2)@A2 @ eps_2_inv

def calcmatMz1(cov_yz1,inv_z1cx2,cov_yx2,inv_x2cz1,cov_x2z1,inv_z1):
	return cov_yz1 @ inv_z1cx2 - cov_yx2@inv_x2cz1 @ cov_x2z1


def calcAK2(cov_yx2,cov_yz1,inv_z1,cov_z1x2,A,inv_z2cz1):
	nx2 = cov_yx2.shape[1]
	tmp_one = np.zeros((nx2,nx2+inv_z1.shape[0]))
	tmp_one[:nx2,:nx2] = np.eye(nx2)
	return (cov_yx2-cov_yz1@inv_z1@cov_z1x2) @ tmp_one @ A.T@inv_z2cz1

def calcYcZ12(cov_y,cov_yz1,inv_z1,A,inv_z2cz1,cov_yx2):
	con_x2y_z1y = np.concatenate((cov_yx2.T,cov_yz1.T),axis=0)
	return cov_y -cov_yz1 @ inv_z1 @ cov_yz1.T - cov_x2y_z1y.T@ A.T @ inv_z2cz1 @ A @ cov_x2cyz1
def calcGent(cov):
	return 0.5 * (np.log(2*np.pi*np.exp(1)) + np.log(np.absolute(np.linalg.det(cov))) )
def calcGMI(cov1,cov2):
	return 0.5 * np.log(np.absolute(np.linalg.det(cov1)/np.linalg.det(cov2)))