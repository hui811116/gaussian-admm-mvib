import numpy as np
import sys
import os
import algorithms as alg
import utils as ut

# This script is used to generate the consensus representation from merge view observations
# this is treated as the base line
# the metric we examine is the relevance rate that the consensus attains
# the result we expect is that the cc approximation can attain the similar relevance rate when there is sufficient overlap

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

cov_x12cy = cov_x12 - cov_x12y @ np.linalg.inv(cov_y) @ cov_x12y.T

beta_range = np.geomspace(1,100,num=64) # collect: beta, izx, izy, niter, conv
res_merge = np.zeros((len(beta_range),5))

nz = 2
maxiter = 100000
conv_thres= 1e-5
for bidx, beta in enumerate(beta_range):
	#covx,covy,covxy,beta,maxiter,convthres
	out_dict = alg.GaussianBACC(cov_x12,cov_y,cov_x12y,beta,maxiter,conv_thres,**{"nz":nz})
	niter = out_dict["niter"]
	conv = int(out_dict['conv'])
	Ax = out_dict["Ax"]
	cov_eps = out_dict["cov_eps"]
	ker_z = Ax @ cov_x12 @ Ax.T + cov_eps
	ker_zx = cov_eps
	ker_zy = Ax @ cov_x12cy @ Ax.T + cov_eps
	mizx = ut.calcGMI(ker_z,ker_zx)
	mizy = ut.calcGMI(ker_z,ker_zy)
	res_merge[bidx,:]= np.array([1/beta, mizx,mizy, niter, conv])
	print("beta,{:.6f},IZX,{:.6f},IZY,{:.6f},niter,{:},conv,{:}".format(beta,mizx,mizy,niter,conv))
with open("result_merge_bacc.npy","wb") as fid:
	np.save(fid,res_merge)
