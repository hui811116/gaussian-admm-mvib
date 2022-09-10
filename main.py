import numpy as np
import sys
import os

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
beta = 2.4
# cond cov
# nz = nx = 2
Ax = np.eye(2)
# used to find Z= AX + eps
cov_eps = np.eye(2)

niter = 1000
conv = False
itcnt = 0
conv_thres = 1e-7
while itcnt< niter:
	itcnt +=1
	new_cov_z = Ax @ cov_x  @ Ax.T + cov_eps
	new_cov_zcy= Ax@ cov_xcy @ Ax.T + cov_eps
	inv_cov_zcy = np.linalg.inv(new_cov_zcy)
	cov_eps_nex = np.linalg.inv(beta * inv_cov_zcy - (beta-1)*np.linalg.inv(new_cov_z))
	new_Ax = beta * cov_eps_nex@ inv_cov_zcy@Ax@(np.eye(2)-cov_ycx@np.linalg.inv(cov_x))
	#print("*"*10 + " ITER {:} ".format(nn)+"*"*10)

	#print("new cov eps:")
	#print(cov_eps_nex)
	#print("new Ax:")
	#print(new_Ax)
	# compute the frobineous norm
	fnorm_eps = np.sum( (cov_eps_nex - cov_eps)**2 )
	fnorm_ax  = np.sum( (new_Ax - Ax)**2)
	if fnorm_eps < conv_thres and fnorm_ax < conv_thres:
		conv = True
		break
	else:
		# update
		Ax = new_Ax
		cov_eps = cov_eps_nex
print("End at iteration:{:}".format(itcnt))
print("convergent? {:}".format(conv))