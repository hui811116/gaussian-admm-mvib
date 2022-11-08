import numpy as np
import sys
import os
import algorithms as alg
import matplotlib.pyplot as plt
from scipy.io import savemat

rng = np.random.default_rng()

nx = 2
ny = 2

mu_x = np.array([1.5, -1.5])

mu_y = np.array([2.5, -2.5])
cov_x = np.array([
		[1.0, 0,],
		[0, 1.0],
	])

cov_y = np.array([
		[1.0, 0 ],
		[0, 1.0],
	])

cov_xy = np.array([
		[0.66, -0.35],
		[-0.35, 0.55],
	])
cov_all = np.zeros((nx+ny,nx+ny))
cov_all[:nx,:nx] = cov_x
cov_all[nx:nx+ny,nx:nx+ny] = cov_y
cov_all[nx:nx+ny,:nx] = cov_xy.T
cov_all[:nx,nx:nx+ny] = cov_xy

cov_xcy = cov_x - cov_xy @ np.linalg.inv(cov_y) @ cov_xy.T
cov_ycx = cov_y - cov_xy.T @ np.linalg.inv(cov_x) @ cov_xy

#conv_all = cov_all + np.eye(nx+ny)
#print(cov_all)
mat_L = np.linalg.cholesky(cov_all)
#print(mat_L)
mu = np.concatenate((mu_x,mu_y))
nsamp = 100
noise = np.random.normal(loc=0,scale=1.0,size=(nx+ny,nsamp))

x_samp = mu[:,None] + mat_L @ noise

# beta for GIB
#gamma = 0.12
beta_range = np.geomspace(1,32,num=64)
#beta = 2.4
# cond cov
# nz = nx = 2
#Ax = np.eye(2)
# used to find Z= AX + eps
#cov_eps = np.eye(2)
penalty_c = 32.0
ss_fixed = 4e-3

niter = 50000
conv_thres = 1e-5
debug = False
if debug:
	niter = 30
	beta_range = [2.55]
#nrun = 20
#beta = 1.32
#out_dict = alg.GaussianADMMIB(cov_x,cov_y,cov_xy,beta,niter,conv_thres)


res_all = np.zeros((len(beta_range),4))

for bidx,beta in enumerate(beta_range):
#for beta in test_beta:
	
	out_dict = alg.GaussianBA(cov_x,cov_y,cov_xy,beta,niter,conv_thres)
	#out_dict = alg.GaussianADMMIB(cov_x,cov_y,cov_xy,beta,niter,conv_thres)
	#out_dict = alg.GaussianADMMIB2(cov_x,cov_y,cov_xy,beta,niter,conv_thres)
	itcnt = out_dict['niter']
	conv = out_dict['conv']
	
	ba_A = out_dict['Ax']
	ba_eps=out_dict['cov_eps']
	ker_z = ba_A @ cov_x @ ba_A.T + ba_eps
	ker_zx= ba_eps
	ker_zy= ba_A @ cov_xcy@ba_A.T + ba_eps
	mizx = 0.5 *np.log(np.linalg.det(ker_z)/np.linalg.det(ker_zx)) #nats
	mizy = 0.5 *np.log(np.linalg.det(ker_z)/np.linalg.det(ker_zy)) #nats
	print("method=BA,beta={:.4f},mizx={:.4f}, mizy={:.4f}, niter={:}, conv={:}".format(beta,mizx,mizy,itcnt,conv))
	res_all[bidx,:] = np.array([mizx,mizy,itcnt,conv])
'''	
	tmp_conv = 0
	for nn in range(nrun):
		out_dict = alg.GaussianADMMIB(cov_x,cov_y,cov_xy,beta,niter,conv_thres)
		itcnt = out_dict['niter']
		conv = out_dict['conv']
		tmp_conv += int(out_dict['conv'])
		print("beta={:.4f}, nidx={:}, niter={:}, conv={:}".format(beta,nn,itcnt,conv))
	print("Summary of beta={:.4f}, convergence_rate={:.4f}".format(beta,tmp_conv/nrun))
'''
res_admm1 = np.zeros((len(beta_range),4))
for bidx,beta in enumerate(beta_range):
	out_dict = alg.GaussianADMMIB(cov_x,cov_y,cov_xy,beta,niter,conv_thres,**{"penalty":penalty_c,"ss":ss_fixed})
	itcnt = out_dict['niter']
	conv = out_dict['conv']
	if conv:
		t1_A = out_dict['Ax']
		t1_eps = out_dict['cov_eps']
		ker_z = t1_A @ cov_x @ t1_A.T + t1_eps
		ker_zx = t1_eps
		ker_zy = t1_A @ cov_xcy @ t1_A.T + t1_eps
		mizx = 0.5 * np.log(np.linalg.det(ker_z)/np.linalg.det(ker_zx))
		mizy = 0.5 * np.log(np.linalg.det(ker_z)/np.linalg.det(ker_zy))
	else:
		mizx = 0
		mizy = 0
	res_admm1[bidx,:] = np.array([mizx,mizy,itcnt,conv])

res_admm2 = np.zeros((len(beta_range),4))
for bidx,beta in enumerate(beta_range):
	out_dict = alg.GaussianADMMIB2(cov_x,cov_y,cov_xy,beta,niter,conv_thres,**{"penalty":penalty_c,"ss":ss_fixed})
	itcnt = out_dict['niter']
	conv = out_dict['conv']
	if conv:
		t2_A = out_dict['Ax']
		t2_eps = out_dict['cov_eps_x']
		ker_z = t2_A@ cov_x @ t2_A.T + t2_eps
		ker_zx = t2_eps
		ker_zy = t2_A @ cov_xcy @ t2_A.T + t2_eps
		mizx = 0.5 * np.log(np.linalg.det(ker_z)/np.linalg.det(ker_zx))
		mizy = 0.5 * np.log(np.linalg.det(ker_z)/np.linalg.det(ker_zy))
	else:
		mizx = 0
		mizy=  0
	res_admm2[bidx,:] = np.array([mizx,mizy,itcnt,conv])

# saving results as .mat file
ba_dict = {"results_array":res_all,'method':"ba"}
admm1_dict= {"results_array":res_admm1,"method":"admm1"}
admm2_dict= {"results_array":res_admm2,"method":"admm2"}
savemat("ba_results.mat",ba_dict)
savemat("admm1_results.mat",admm1_dict)
savemat("admm2_results.mat",admm2_dict)

fig,ax = plt.subplots()
sel_idx = res_all[:,3] ==True
sel_t1  = res_admm1[:,3] == True
sel_t2 = res_admm2[:,3] == True
ax.scatter(res_all[sel_idx,0],res_all[sel_idx,1],label="BA",marker="*")
ax.scatter(res_admm1[sel_t1,0],res_admm1[sel_t1,1],label="ADMM_type1",marker="+")
ax.scatter(res_admm2[sel_t2,0],res_admm2[sel_t2,1],label="ADMM_type2",marker="^")
ax.set_xlabel(r"$I(X;Z)$")
ax.set_ylabel(r"$I(Y;Z)$")
ax.grid("on")
ax.legend()
plt.show()