import numpy as np
import sys
import os
import algorithms as alg
import utils as ut
import argparse
import pickle
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--penalty", type=float, default=64.0, help="penalty coefficient")
parser.add_argument("--ss", type=float, default=2e-3, help="step size for gradient descent")
parser.add_argument("--maxiter", type=int, default=100000, help="maximum number of iterations")
parser.add_argument("--convthres",type=float, default=1e-5, help="convergence threshold")
#parser.add_argument("--save_name",type=str, default="result_cc",help="save file name")
parser.add_argument("--gamma_min",type=float,default=0.001, help="minimum gamma value (second step)")
parser.add_argument("--gamma_num",type=int, default=32, help="number of gamma on the second stage")
parser.add_argument("--nrun",type=int,default=10,help="number of trials per parameter set")
#parser.add_argument("--beta_max",type=float, default=256, help="maximum beta value")
#parser.add_argument("--beta_num",type=int, default=8, help="number of beta to search")
#parser.add_argument("--first_view",type=int, default=1, help="which view starts first")

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

#beta_range = np.geomspace(1,args.beta_max,num=args.beta_num) # collect: beta, gamma, iz1x1, iz1y, iz2x1cz1, iz2cz1y, niter1, niter2, conv2
gamma_range = np.geomspace(args.gamma_min,1,num=args.gamma_num)

#gamma_1 = 0.04
maxiter = args.maxiter
conv_thres = args.convthres
#gamma_2 = 0.04
nc = 2

param_dict = {"penalty":args.penalty,"ss":args.ss}
# find the common information
result_pkl = []
'''
for gx1, gamma_1 in enumerate(gamma_range):
	for gx2, gamma_2 in enumerate(gamma_range):
		cc_out = alg.GaussianMvIBCc(cov_x1,cov_x2,cov_y,cov_x1x2,cov_x1y,cov_x2y,nc,gamma_1,gamma_2,maxiter,conv_thres,**param_dict)
		#cc_out = alg.GaussianMvIBCondCc(cov_x1,cov_x2,cov_y,cov_x1x2,cov_x1y,cov_x2y,nc,gamma_1,gamma_2,maxiter,conv_thres,**param_dict)
		#cov_x1,cov_x2,cov_y,cov_x12,cov_x1y,cov_x2y,nc,gamma1,gamma2,maxiter,convthres
		# update the equivalent priors
		cc_conv = int(cc_out['conv'])
		print("progress, gamma1={:.5f}, gamma2={:.5f}, convergence={:}".format(gamma_1,gamma_2,cc_conv))
		Ax = cc_out["Ax"]
		cov_z = cc_out["cov_z"]
		cov_zcy = cc_out["cov_zcy"]
		# precalculate the mizx and mizy here

		tmp_dict = {"conv":cc_conv,"micx":cc_out["micx"],"micy":cc_out["micy"],"niter":cc_out["niter"],
					"gamma":np.array([gamma_1,gamma_2]),"metrics_v1":[],"metrics_v2":[]}
		if cc_conv: 
			# compute the priors
			cov_zxall = Ax @ cov_x12
			cov_zx1 = cov_zxall[:,nx1]
			cov_zx2 = cov_zxall[:,nx1:nx1+nx2]
			cov_zy = Ax @ cov_x12y
			#cov_x2,ba_z1,ba_z1x2.T,cov_y,ba_z1y.T,cov_x2y,gamma_2,maxiter,conv_thres
			#gamma_c = 0.04
			for bidx, beta in enumerate(beta_range):
				ce1_out = alg.GaussianADMMMvIBInc(cov_x1,cov_z,cov_zx1.T,cov_y,cov_zy,cov_x1y,1/beta,maxiter,conv_thres,**param_dict)
				tmp_dict["metrics_v1"].append(np.array([beta,ce1_out["mixcz"],ce1_out["miycz"],ce1_out["niter"],ce1_out["conv"]]))
			for bidx, beta in enumerate(beta_range):
				ce2_out = alg.GaussianADMMMvIBInc(cov_x2,cov_z,cov_zx2.T,cov_y,cov_zy,cov_x2y,1/beta,maxiter,conv_thres,**param_dict)
				tmp_dict["metrics_v2"].append(np.array([beta,ce2_out["mixcz"],ce2_out["miycz"],ce2_out["niter"],ce2_out["conv"]]))
		else:
			# this reduces to single view, distributed fashion
			# covx,covy,covxy,beta,maxiter,convthres
			# or we can solve this with singleview ADMM-IB
			#admm1_out = alg.GaussianADMMIB(cov_x1,cov_y,cov_x1y,beta1,maxiter,conv_thres,**param_dict)
			for bidx, beta in enumerate(beta_range):
				ba1_out = alg.GaussianBA(cov_x1,cov_y,cov_x1y,beta,maxiter,conv_thres,**param_dict)
				ba_A = ba1_out["Ax"]
				ba_eps= ba1_out["cov_eps"]
				cov_v1_z = ba_A @ cov_x1 @ ba_A.T + ba_eps
				cov_v1_zy= ba_A @ cov_x1cy@ ba_A.T + ba_eps
				v1_mizx = ut.calcGMI(cov_v1_z,ba_eps)
				v1_mizy = ut.calcGMI(cov_v1_z,cov_v1_zy)
				tmp_dict["metrics_v1"].append(np.array([beta,v1_mizx,v1_mizy,ba1_out["niter"],ba1_out["conv"]]))
			for bidx, beta in enumerate(beta_range):
				ba2_out = alg.GaussianBA(cov_x2,cov_y,cov_x2y,beta,maxiter,conv_thres,**param_dict)
				ba_A = ba2_out["Ax"]
				ba_eps= ba2_out["cov_eps"]
				cov_v2_z = ba_A @ cov_x2 @ ba_A.T + ba_eps
				cov_v2_zy = ba_A @ cov_x2cy@ ba_A + ba_eps
				v2_mizx = ut.calcGMI(cov_v2_z,ba_eps)
				v2_mizy = ut.calcGMI(cov_v2_z,cov_v2_zy)
				tmp_dict["metrics_v2"].append(np.array([beta,v2_mizx,v2_mizy,ba2_out["niter"],ba2_out["conv"]]))
		result_pkl.append(tmp_dict)

with open(args.save_name+".pkl","wb") as fid:
	pickle.dump(result_pkl,fid)
'''
results_all = np.zeros((gamma_range.shape[0]*args.nrun,9))
# gamma,nrun, conv, niter, mizcx1, mizcx2, mizcy, ent_zc, com_info
for gidx, gamma in enumerate(gamma_range):
	conv_cnt = 0
	for nn in range(args.nrun):
		# use the same gamma
		cc_out = alg.GaussianMvIBCondCc(cov_x1,cov_x2,cov_y,cov_x1x2,cov_x1y,cov_x2y,nc,gamma,gamma,maxiter,conv_thres,**param_dict)
		conv = cc_out['conv']
		niter = cc_out['niter']
		Ax1 = cc_out['Ax1']
		Ax2 = cc_out['Ax2']
		cov_eps_x1 = cc_out['cov_eps1']
		cov_eps_x2 = cc_out['cov_eps2']
		cov_zc = cc_out['cov_z']
		cov_zccy = cc_out['cov_zcy']
		#
		ent_zc = 0.5* np.log(np.linalg.det(cov_zc))
		mizcx1 = ent_zc - 0.5 * np.log(np.linalg.det(cov_eps_x1))
		mizcx2 = ent_zc - 0.5 * np.log(np.linalg.det(cov_eps_x2))
		mizcy = ent_zc - 0.5 * np.log(np.linalg.det(cov_zccy))

		# counting
		conv_cnt += int(conv)
		if conv:
			# calculate the common information metrics I(X_1;X_2|z_c)
			# need h(x_1|Zc) - h(x_1|x2,Zc)
			# sigma_x1|zc = sigma_x1 - sigmax1zc @ sigma_zc^{-1} @ sigmax1zc.T
			# sigma_x1|x2zc = sigma_x1 - sigmax1[x2 zc] @ sigma_{x2...zc} @ sigma[x2zc]x1
			tmp_sig_x1zc = cov_x1@Ax1.T
			out_x1czc = cov_x1 - tmp_sig_x1zc @ np.linalg.inv(cov_zc) @ tmp_sig_x1zc.T

			tmp_sig_x1_bx2zc = np.block([cov_x1x2,tmp_sig_x1zc])
			tmp_ker_x2zc = np.block([[cov_x2,cov_x2@Ax2.T],[Ax2@cov_x2,cov_zc]])
			out_inv_x2zc = np.linalg.inv(tmp_ker_x2zc)
			out_x1cx2zc = cov_x1 - tmp_sig_x1_bx2zc@out_inv_x2zc @ tmp_sig_x1_bx2zc.T
			common_info = 0.5 * np.log(np.linalg.det(out_x1czc)) - 0.5 * np.log(np.linalg.det(out_x1cx2zc))
			results_all[gidx*args.nrun+nn] = np.array([gamma,nn,int(conv),niter,mizcx1,mizcx2,mizcy,ent_zc,common_info])
		else:
			#common_info = -1 # meaning that the value is undefined...
			results_all[gidx*args.nrun+nn] = np.array([gamma,nn,int(conv),niter,0,0,0,0,-1])

# saving as numpy result
save_name = "results_cc_penc{:}_ss{:.6e}_gnum{:}".format(args.penalty,args.ss,args.gamma_num)
safe_save_name = copy.copy(save_name)
repeat_cnt = 0
while os.path.isfile(safe_save_name+".npy"):
	repeat_cnt += 1
	safe_save_name = "{:}_{:}".format(save_name,repeat_cnt)
with open(safe_save_name+".npy","wb") as fid:
	np.save(fid,results_all)