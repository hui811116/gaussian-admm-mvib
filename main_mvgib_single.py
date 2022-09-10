import numpy as np
import sys
import os
import algorithms as alg
import utils as ut

# we solve the first view with ADMM-GIB,
# call GaussianADMMIB2 (both types will do in first step, use A case will be easier here)

# then we move on to the second view

# specialized for the two-view problem
# assume all zero means
nx1 = 2
nx2 = 2
ny  = 2

cov_x1 = np.eye(2)
cov_x2 = np.eye(2)
cov_y = np.eye(2)
cov_x1y = np.array([
	[-0.45, 0],
	[0, .90]])
cov_x2y = np.array([
	[0.10,0],
	[0,.40]]
	)
#cov_x1x2 = np.array([
#	[0.8, 0],
#	[0, 0]])
cov_x1x2 = cov_x1y @ np.linalg.inv(cov_y) @ cov_x2y.T

cov_x1cy = cov_x1 - cov_x1y @ np.linalg.inv(cov_y) @ cov_x1y.T
cov_x2cy = cov_x2 - cov_x2y @ np.linalg.inv(cov_y) @ cov_x2y.T
cov_ycx1 = cov_y  - cov_x1y.T@np.linalg.inv(cov_x1)@ cov_x1y
cov_ycx2 = cov_y  - cov_x2y.T@np.linalg.inv(cov_x2)@ cov_x2y

# can we construct joint (y,x1x2) and run BA?
cov_x12 = np.eye(4)
cov_x12[:nx1,nx1:nx1+nx2] = cov_x1x2
cov_x12[nx1:nx1+nx2,:nx1] = cov_x1x2.T
cov_x12[:nx1,:nx1] = cov_x1
cov_x12[nx1:nx1+nx2,nx1:nx1+nx2] = cov_x2
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

#gamma_1 = 0.22
maxiter = 50000
conv_thres = 1e-5

beta_range = np.geomspace(1,128,num=256) # collect: beta, izx, izy, niter, conv
# NOTE: because this starts with deterministic initialization, no multiple runs
res_v1 = np.zeros((len(beta_range),5))
res_v2 = np.zeros((len(beta_range),5))

for gidx, beta in enumerate(beta_range):
	# first view
	out_dict = alg.GaussianBA(cov_x1,cov_y,cov_x1y,beta,maxiter,conv_thres)
	itcnt = out_dict['niter']
	conv = int(out_dict['conv'])
	ba_A = out_dict['Ax']
	ba_eps = out_dict['cov_eps']
	ker_z = ba_A @ cov_x1 @ ba_A.T + ba_eps
	ker_zx = ba_eps
	ker_zy = ba_A @ cov_x1cy @ ba_A.T + ba_eps
	mizx = ut.calcGMI(ker_z,ker_zx)
	mizy = ut.calcGMI(ker_z,ker_zy)
	res_v1[gidx,:] = np.array([1/beta,mizx,mizy,itcnt,conv])

	# second view
	out_dict = alg.GaussianBA(cov_x2,cov_y,cov_x2y,beta,maxiter,conv_thres)
	itcnt = out_dict['niter']
	conv = int(out_dict['conv'])
	ba_A = out_dict['Ax']
	ba_eps = out_dict['cov_eps']
	ker_z = ba_A @ cov_x1 @ ba_A.T + ba_eps
	ker_zx = ba_eps
	ker_zy = ba_A @ cov_x1cy @ ba_A.T + ba_eps
	mizx = ut.calcGMI(ker_z,ker_zx)
	mizy = ut.calcGMI(ker_z,ker_zy)
	res_v2[gidx,:] = np.array([1/beta,mizx,mizy,itcnt,conv])

# saving the two as base lines
with open("result_sv_v1_ba.npy",'wb') as fid:
	np.save(fid,res_v1)
with open("result_sv_v2_ba.npy","wb") as fid:
	np.save(fid,res_v2)
