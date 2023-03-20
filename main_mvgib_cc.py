import numpy as np
import sys
import os
import algorithms as alg

# we solve the first view with ADMM-GIB,
# call GaussianADMMIB2 (both types will do in first step, use A case will be easier here)

# then we move on to the second view

# specialized for the two-view problem
# assume all zero means
nx1 = 2
nx2 = 2
ny  = 2

cov_x1 = np.eye(2)
#cov_x1 = np.array([[2.4,0],[0, 1.2]])
cov_x2 = np.eye(2)
#cov_x2 = np.array([[1,0],[0, 1]])
cov_y = np.eye(2)
#cov_y = np.array([[1.2, 0],[0, 0.88]])
cov_x1y = np.array([
	[-0.45, 0],
	[0, .90]])
#cov_x2y = np.array([
#	[0,0.45],
#	[-0.90, 0.0]]
#	)
cov_x2y = np.array([
	[0.10,0],
	[0,.40]])
#cov_x1x2 = np.array([
#	[0.8, 0],
#	[0, 0]])
cov_x1x2 = cov_x1y @ np.linalg.inv(cov_y) @ cov_x2y.T


cov_x1cy = cov_x1 - cov_x1y @ np.linalg.inv(cov_y) @ cov_x1y.T
cov_x2cy = cov_x2 - cov_x2y @ np.linalg.inv(cov_y) @ cov_x2y.T
#cov_ycx1 = cov_y  - cov_x1y.T@np.linalg.inv(cov_x1)@ cov_x1y
#cov_ycx2 = cov_y  - cov_x2y.T@np.linalg.inv(cov_x2)@ cov_x2y

# can we construct joint (y,x1x2) and run BA?
cov_x12 = np.eye(4)
cov_x12[:nx1,nx1:nx1+nx2] = cov_x1x2
cov_x12[nx1:nx1+nx2,:nx1] = cov_x1x2.T
cov_x12y = np.zeros((nx1+nx2,ny))
cov_x12y[:nx1,:ny] = cov_x1y
cov_x12y[nx1:nx1+nx2,:ny] = cov_x2y

cov_all = np.zeros((ny+nx1+nx2,ny+nx1+nx2))
n12 = nx1+nx2
# sanity check
cov_all[:n12,:n12] = cov_x12
cov_all[:n12,n12:n12+ny] = cov_x12y
cov_all[n12:n12+ny,:n12] = cov_x12y.T
cov_all[n12:n12+ny,n12:n12+ny] = cov_y
#print(cov_all)
# sanity check, if cholesky decomposition exists
samp_mat = np.linalg.cholesky(cov_all)
print("sanity check passed.")
# now solve the first view with GIB, can use BA for simplicity

print("sig_x12")
print(cov_x12)

print("sig_y")
print(cov_y)
print("sig_x12y")
print(cov_x12y)

tmp_x12cy = cov_x12 - cov_x12y @ np.linalg.inv(cov_y)@ cov_x12y.T
print("sig_x12cy")
print(tmp_x12cy)

print("diff_sigx1-sigx1|y")
print(cov_x1 - cov_x1y)

print("diff_sigx2-sigx2|y")
print(cov_x2 - cov_x2y)

gamma_1 = 0.08
maxiter = 100000
conv_thres = 1e-5
gamma_2 = 0.08
nc = 2

param_dict = {"penalty":128.0,"ss":1e-3}
# find the common information
#cc_out = alg.GaussianMvIBCc(cov_x1,cov_x2,cov_y,cov_x1x2,cov_x1y,cov_x2y,nc,gamma_1,gamma_2,maxiter,conv_thres,**param_dict)
cc_out = alg.GaussianMvIBCondCc(cov_x1,cov_x2,cov_y,cov_x1x2,cov_x1y,cov_x2y,nc,gamma_1,gamma_2,maxiter,conv_thres,**param_dict)
#cov_x1,cov_x2,cov_y,cov_x12,cov_x1y,cov_x2y,nc,gamma1,gamma2,maxiter,convthres
print(cc_out)
# update the equivalent priors
'''
cc_conv = cc_out['conv']
Ax = cc_out['Ax1']
#Ax = cc_out["Ax"]
cov_z = cc_out["cov_z"]
cov_zcy = cc_out["cov_zcy"]

if cc_conv: 
	# compute the priors
	cov_zxall = Ax @ cov_x12
	cov_zx1 = cov_zxall[:,nx1]
	cov_zx2 = cov_zxall[:,nx1:nx1+nx2]
	cov_zy = Ax @ cov_x12y
	#cov_x2,ba_z1,ba_z1x2.T,cov_y,ba_z1y.T,cov_x2y,gamma_2,maxiter,conv_thres
	gamma_c = 0.04
	ce1_out = alg.GaussianADMMMvIBInc(cov_x1,cov_z,cov_zx1.T,cov_y,cov_zy,cov_x1y,gamma_c,maxiter,conv_thres,**param_dict)
	ce2_out = alg.GaussianADMMMvIBInc(cov_x2,cov_z,cov_zx2.T,cov_y,cov_zy,cov_x2y,gamma_c,maxiter,conv_thres,**param_dict)
	if ce1_out['conv'] and ce2_out['conv']:
		print("all converged")
	else:
		print("no convergence")
else:
	print("no common info constructed, abort")

# two inc steps with 
'''