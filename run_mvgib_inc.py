import numpy as np
import sys
import os
import algorithms as alg
import utils as ut
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--penalty", type=float, default=32.0, help="penalty coefficient")
parser.add_argument("--ss", type=float, default=4e-3, help="step size for gradient descent")
parser.add_argument("--maxiter", type=int, default=100000, help="maximum number of iterations")
parser.add_argument("--convthres",type=float, default=1e-5, help="convergence threshold")
parser.add_argument("--save_name",type=str, default="result_inc",help="save file name")
parser.add_argument("--gamma_min",type=float,default=0.01, help="minimum gamma value (second step)")
parser.add_argument("--gamma_num",type=int, default=16, help="number of gamma on the second stage")
parser.add_argument("--beta_max",type=float, default=32, help="maximum beta value")
parser.add_argument("--beta_num",type=int, default=16, help="number of beta to search")
parser.add_argument("--first_view",type=int, default=1, help="which view starts first")

args = parser.parse_args()
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
#samp_mat = np.linalg.cholesky(cov_all)
#print("sanity check passed.")
# now solve the first view with GIB, can use BA for simplicity


beta_range = np.geomspace(1,args.beta_max,num=args.beta_num) # collect: beta, gamma, iz1x1, iz1y, iz2x1cz1, iz2cz1y, niter1, niter2, conv2
gamma_range = np.geomspace(args.gamma_min,1,num=args.gamma_num)
maxiter = args.maxiter
conv_thres = args.convthres

res_inc_x1x2 = np.zeros((len(beta_range)*len(gamma_range),9))
ncnt = 0

param_dict = {"penalty":args.penalty,"ss":args.ss}
print(param_dict)

print(",".join(["beta","gamma","miz1x1","miz1y","miz2x1cz1","miz2ycz1","niter1","niter2","conv2"]))
if args.first_view == 1:
	for bidx,beta in enumerate(beta_range):
		out_dict = alg.GaussianBA(cov_x1,cov_y,cov_x1y,beta,maxiter,conv_thres)
		ba_niter = out_dict['niter']
		conv_flag = int(out_dict['conv']) # BA must converge
		ba_A = out_dict['Ax']
		ba_eps = out_dict['cov_eps']
		#print("Output of BA for first view:",out_dict)
		# precomputing the prior matrices
		ba_z1 = ba_A@cov_x1 @ ba_A.T + ba_eps
		ba_z1cy = ba_A @ cov_x1cy @ ba_A.T + ba_eps
		##
		ba_mizx = ut.calcGMI(ba_z1,ba_eps)
		ba_mizy = ut.calcGMI(ba_z1,ba_z1cy)
		##
		ba_z1y = ba_A@cov_x1y
		ba_z1x2 = ba_A@cov_x1x2
		ba_z1cy = ba_A@cov_x1cy @ ba_A.T + ba_eps
		prior_z1cx2 = ba_z1 - ba_z1x2 @ np.linalg.inv(cov_x2) @ ba_z1x2.T
		prior_x2cz1 = cov_x2 - ba_z1x2.T @ np.linalg.inv(ba_z1) @ ba_z1x2

		for gidx, gamma in enumerate(gamma_range):
			inc_dict = alg.GaussianADMMMvIBInc(cov_x2,ba_z1,ba_z1x2.T,cov_y,ba_z1y.T,cov_x2y,gamma,maxiter,conv_thres,**param_dict)
			inc_niter = inc_dict["niter"]
			inc_conv = int(inc_dict["conv"])
			if inc_conv:
				# great, converged
				inc_niter = inc_dict["niter"]
				miczx = inc_dict["mixcz"]
				miczy = inc_dict["miycz"]
			else:
				miczx = 0
				miczy = 0
			tmp_output = [beta,gamma,ba_mizx,ba_mizy,miczx,miczy,ba_niter,inc_niter,inc_conv]
			res_inc_x1x2[ncnt,:] = np.array([beta,gamma,ba_mizx,ba_mizy,miczx,miczy,ba_niter,inc_niter,inc_conv])
			print(",".join(["{:.4f}".format(item) for item in tmp_output]))
			ncnt+=1
elif args.first_view==2:
	for bidx,beta in enumerate(beta_range):
		out_dict = alg.GaussianBA(cov_x2,cov_y,cov_x2y,beta,maxiter,conv_thres)
		ba_niter = out_dict['niter']
		conv_flag = int(out_dict['conv']) # BA must converge
		ba_A = out_dict['Ax']
		ba_eps = out_dict['cov_eps']
		#print("Output of BA for first view:",out_dict)
		# precomputing the prior matrices
		ba_z1 = ba_A@cov_x2 @ ba_A.T + ba_eps
		ba_z1cy = ba_A @ cov_x2cy @ ba_A.T + ba_eps
		##
		ba_mizx = ut.calcGMI(ba_z1,ba_eps)
		ba_mizy = ut.calcGMI(ba_z1,ba_z1cy)
		##
		ba_z1y = ba_A@cov_x2y
		ba_z1x1 = ba_A@cov_x1x2.T # this is cross correlation
		ba_z1cy = ba_A@cov_x1cy @ ba_A.T + ba_eps
		prior_z1cx2 = ba_z1 - ba_z1x1 @ np.linalg.inv(cov_x1) @ ba_z1x1.T
		prior_x2cz1 = cov_x2 - ba_z1x1.T @ np.linalg.inv(ba_z1) @ ba_z1x1

		for gidx, gamma in enumerate(gamma_range):
			inc_dict = alg.GaussianADMMMvIBInc(cov_x1,ba_z1,ba_z1x1.T,cov_y,ba_z1y.T,cov_x1y,gamma,maxiter,conv_thres,**param_dict)
			inc_niter = inc_dict["niter"]
			inc_conv = int(inc_dict["conv"])
			if inc_conv:
				# great, converged
				inc_niter = inc_dict["niter"]
				miczx = inc_dict["mixcz"]
				miczy = inc_dict["miycz"]
			else:
				miczx = 0
				miczy = 0
			tmp_output = [beta,gamma,ba_mizx,ba_mizy,miczx,miczy,ba_niter,inc_niter,inc_conv]
			res_inc_x1x2[ncnt,:] = np.array([beta,gamma,ba_mizx,ba_mizy,miczx,miczy,ba_niter,inc_niter,inc_conv])
			print(",".join(["{:.4f}".format(item) for item in tmp_output]))
			ncnt+=1

with open(args.save_name+".npy","wb") as fid:
	np.save(fid,res_inc_x1x2)