import numpy as np
import sys
import os
import copy
import utils as ut
from numpy.random import default_rng

def GaussianBA(covx,covy,covxy,beta,maxiter,convthres,**kwargs):
	# precomputing
	# always, Nz=Nx
	nx = covx.shape[0]
	ny = covy.shape[0]
	cov_xcy = covx - covxy @ np.linalg.inv(covy) @ covxy.T
	cov_ycx = covy - covxy.T @ np.linalg.inv(covx) @ covxy
	inv_covx = np.linalg.inv(covx)
	id_x = np.eye(nx)
	Ax = np.eye(nx) # linear mapping of X as mean of Z
	cov_eps = np.eye(nx) # second order statistic for Z
	itcnt = 0
	conv_flag = False
	ker_ycxxinv = id_x - cov_xcy@inv_covx
	while itcnt < maxiter:
		itcnt+=1
		new_cov_z = Ax @ covx @ Ax.T + cov_eps
		new_cov_zcy = Ax @ cov_xcy @ Ax.T + cov_eps
		inv_cov_zcy = np.linalg.inv(new_cov_zcy)
		cov_eps_nex = np.linalg.inv(beta * inv_cov_zcy - (beta-1) * np.linalg.inv(new_cov_z))
		new_Ax = beta * cov_eps_nex @ inv_cov_zcy @ Ax@ ker_ycxxinv
		# convergence criterion
		fnorm_ax = np.sqrt(np.sum((new_Ax - Ax)**2))
		fnorm_eps = np.sqrt(np.sum((cov_eps_nex - cov_eps)**2))
		if fnorm_ax < convthres and fnorm_eps < convthres:
			conv_flag = True
			break
		else:
			Ax = new_Ax
			cov_eps = cov_eps_nex
	return {"Ax":Ax,"cov_eps":cov_eps,"conv":conv_flag,"niter":itcnt}

# this is essentially the same as the BA, but the representation dimension is the smallest among the two
def GaussianBACC(covx,covy,covxy,beta,maxiter,convthres,**kwargs):
	# precomputing
	rng = np.random.default_rng()
	nz = kwargs['nz']
	# always, Nz=Nx
	nx = covx.shape[0]
	ny = covy.shape[0]
	cov_xcy = covx - covxy @ np.linalg.inv(covy) @ covxy.T
	cov_ycx = covy - covxy.T @ np.linalg.inv(covx) @ covxy
	inv_covx = np.linalg.inv(covx)
	id_x = np.eye(nx)
	#Ax = np.eye(nx) # linear mapping of X as mean of Z
	ax_append = rng.random((nz,nx-nz))
	# normalize the elements
	ax_append /= np.linalg.norm(ax_append,axis=0)
	Ax = np.block([np.eye(nz),ax_append])
	cov_eps = np.eye(nz) # second order statistic for Z
	itcnt = 0
	conv_flag = False
	ker_ycxxinv = id_x - cov_xcy@inv_covx
	while itcnt < maxiter:
		itcnt+=1
		new_cov_z = Ax @ covx @ Ax.T + cov_eps
		new_cov_zcy = Ax @ cov_xcy @ Ax.T + cov_eps
		inv_cov_zcy = np.linalg.inv(new_cov_zcy)
		cov_eps_nex = np.linalg.inv(beta * inv_cov_zcy - (beta-1) * np.linalg.inv(new_cov_z))
		new_Ax = beta * cov_eps_nex @ inv_cov_zcy @ Ax@ ker_ycxxinv
		# convergence criterion
		fnorm_ax = np.sqrt(np.sum((new_Ax - Ax)**2))
		fnorm_eps = np.sqrt(np.sum((cov_eps_nex - cov_eps)**2))
		if fnorm_ax < convthres and fnorm_eps < convthres:
			conv_flag = True
			break
		else:
			Ax = new_Ax
			cov_eps = cov_eps_nex
	return {"Ax":Ax,"cov_eps":cov_eps,"conv":conv_flag,"niter":itcnt}

def GaussianADMMIB(covx,covy,covxy,beta,maxiter,convthres,**kwargs):
	# FIXME: fixed parameters
	#rng = np.random.default_rng()
	penalty_c = kwargs['penalty']
	ss_x = kwargs['ss']  # now we don't need it to be projected back to simplex, fixed ss available
	ss_y = kwargs['ss']
	ss_eps = kwargs['ss']
	# type I, shared noise vector 
	# precomputing
	# always, Nz=Nx
	nx = covx.shape[0]
	ny = covx.shape[1]
	cov_xcy = covx - covxy @ np.linalg.inv(covy) @ covxy.T
	cov_ycx = covy - covxy.T @ np.linalg.inv(covx) @ covxy
	inv_covx = np.linalg.inv(covx)
	id_x = np.eye(nx)
	cov_eps = np.eye(nx) # second order statistic for Z
	# primal and dual vars
	Ax = np.eye(nx) # linear mapping of X as mean of Z
	Ay = copy.deepcopy(Ax)
	# dual variables
	dual_A = np.zeros(Ax.shape) # element-wise dot product
	itcnt = 0
	conv_flag = False
	# pre-computing inversion
	inv_ax = np.linalg.inv(Ax @ covx @ Ax.T + cov_eps)
	inv_ay = np.linalg.inv(Ay @ cov_xcy @ Ay.T + cov_eps)
	while itcnt<maxiter:
		itcnt+=1
		# using gradient descent worked.
		grad_eps = 0.5 * ((1-beta) * inv_ax.T - np.linalg.inv(cov_eps.T) + beta * inv_ay.T ) 
		new_cov_eps = cov_eps - grad_eps * ss_eps
		# noise update
		inv_ax = np.linalg.inv(Ax @ covx@Ax.T + new_cov_eps)
		inv_ay = np.linalg.inv(Ay @ cov_xcy @ Ay.T + new_cov_eps)
		# step I, update Ax
		# the step loss: (1-\beta)/2 * log det(Ax @ cov_x @ Ax.T + cov_eps)
		# a convex function of Ax as \beta>1 
		# penalties: dot(Ax-Ay,dual_var) + c/2 * |Ax- Ay|^2
		err_x = Ax - Ay
		grad_x = (1-beta) * inv_ax @ Ax @ covx + dual_A + penalty_c* err_x
		new_Ax = Ax - grad_x * ss_x
		
		# dual update
		err_x = new_Ax - Ay
		dual_A += penalty_c * err_x
		# weakly convex update
		grad_y = beta * inv_ay @ Ay @ cov_xcy - dual_A - penalty_c * err_x
		new_Ay = Ay - grad_y * ss_y

		err_x = new_Ax - new_Ay
		# conv check
		conv_A = np.sqrt(np.sum(err_x ** 2))
		conv_eps = np.sqrt(np.sum( (new_cov_eps - cov_eps)**2 ))
		if conv_A<convthres and conv_eps<convthres:
			conv_flag = True
			break
		else:
			Ax = new_Ax
			Ay = new_Ay
			cov_eps = new_cov_eps
	return {"Ax":Ax,"Ay":Ay,"cov_eps":cov_eps,'niter':itcnt,"conv":conv_flag}
def GaussianADMMIB2(covx,covy,covxy,beta,maxiter,convthres,**kwargs):
	penalty_c = kwargs['penalty']
	ss_x = kwargs['ss']  # now we don't need it to be projected back to simplex, fixed ss available
	ss_y = kwargs['ss']
	ss_a = kwargs['ss']
	nx = covx.shape[0]
	ny = covx.shape[1]
	cov_xcy = covx - covxy @ np.linalg.inv(covy) @ covxy.T
	cov_ycx = covy - covxy.T @ np.linalg.inv(covx) @ covxy
	inv_covx = np.linalg.inv(covx)
	id_x = np.eye(nx)
	cov_eps_x = np.eye(nx) # second order statistic for Z
	cov_eps_y = copy.deepcopy(cov_eps_x)
	Ax = np.eye(nx) # linear mapping of X as mean of Z
	itcnt = 0
	conv_flag = False

	dual_eps = np.zeros(covx.shape)

	inv_zx = np.linalg.inv(Ax @ covx @ Ax.T + cov_eps_y)
	inv_zcy = np.linalg.inv(Ax @ cov_xcy @ Ax.T + cov_eps_y)

	while itcnt < maxiter:
		itcnt += 1
		err_x = cov_eps_x - cov_eps_y
		grad_x = -0.5 * np.linalg.inv(cov_eps_x).T + dual_eps+penalty_c * err_x

		new_cov_eps_x = cov_eps_x - ss_x * grad_x

		err_x = new_cov_eps_x - cov_eps_y

		grad_y = 0.5 * (1-beta) * inv_zx.T + 0.5 * beta * inv_zcy.T - dual_eps - penalty_c * err_x
		new_cov_eps_y = cov_eps_y  - ss_y * grad_y

		err_x = new_cov_eps_x - new_cov_eps_y
		dual_eps += penalty_c * err_x

		inv_zx = np.linalg.inv(Ax @ covx @ Ax.T + new_cov_eps_y)
		inv_zcy = np.linalg.inv(Ax @ cov_xcy @ Ax.T + new_cov_eps_y)
		
		grad_Ax = (1-beta) * inv_zx @ Ax @ covx + beta * inv_zcy @ Ax @ cov_xcy
		new_Ax = Ax - ss_a * grad_Ax

		conv_eps = np.sqrt(np.sum(err_x ** 2))
		conv_Ax  = np.sqrt(np.sum((new_Ax-Ax)**2))
		if conv_eps< convthres and conv_Ax < convthres:
			conv_flag = True
			break
		else:
			Ax = new_Ax
			cov_eps_x = new_cov_eps_x
			cov_eps_y = new_cov_eps_y
	return {"Ax":Ax, "cov_eps_x":cov_eps_x,"cov_eps_y":cov_eps_y,"niter":itcnt,"conv":conv_flag}


'''
def GaussianADMMMvIBStep2(cov_x2,cov_z1,cov_x2z1,cov_y,cov_yz1,cov_x2y,gamma,maxiter,convthres,**kwargs):
	penalty_c = 16.0
	ss_x = 1e-2
	ss_z = 1e-2
	ss_a = 1e-2
	
	nx = cov_x2.shape[0]
	nz1 = cov_z1.shape[0]
	nz = nx+nz1
	ny = cov_y.shape[0]
	cov_eps_zcx = np.eye(nz)
	cov_eps_z   = np.eye(nz)
	dual_eps = np.zeros(cov_x2.shape)

	rng = np.random.default_rng()
	Ax = np.zeros((nz,nx+nz1))
	Ax[:,:nz] = np.eye(nz)
	if nx+nz1-nz>0:
		rnd_ele = randint(0,nz-1,size=(nx+nz1-nz))
		for ii,item in enumerate(rnd_ele):
			Ax[item,nz+ii] = 1
	rnd_Ax = np.random.permutation(nx)
	Ax = Ax[:,rnd_Ax]
	# precomputing prior matrices
	# patch matrices
	joint_yz1 = np.eye(ny+nz1)
	joint_yz1[:ny,:ny] = cov_y
	joint_yz1[:ny,ny:] = cov_yz1
	joint_yz1[ny:,:ny] = cov_yz1.T
	joint_yz1[ny:,ny:] = cov_z1
	patch_x2_yz1 = np.zeros((nx,ny+nz1))
	patch_x2_yz1[:,:ny] = cov_x2y
	patch_x2_yz1[:,ny:] = cov_x2z1
	# inversions
	inv_z1 = np.linalg.inv(cov_z1)
	inv_joint_yz1= np.linalg.inv(joint_yz1)

	# prior matrix 
	cov_x2cyz1 = cov_x2 - patch_x2_yz1 @ inv_joint_yz1 @ patch_x2_yz1.T
	cov_x2cz1 = cov_x2 - cov_x2z1 @ inv_z1 @ cov_x2z1.T
	inv_x2cz1 = np.linalg.inv(cov_x2cz1)

	patch_yz1_invz1 = cov_yz1 @ inv_z1
	prior_z1x2_cond_z1 = cov_x2z1.T @ inv_x2cz1 @ cov_x2z1
	prior_yx2_cond_z1  = cov_x2y.T @ inv_x2cz1 @ cov_x2z1

	inv_z2cz1 = np.linalg.inv(Ax[:,:nx] @ cov_x2cz1 @ Ax[:,:nx].T + cov_eps_z)
	inv_z2cyz1  = np.linalg.inv(Ax @ cov_x2cyz1@ Ax.T + cov_eps_z) # TODO

	# precomputed inversion of the variables
	inv_eps_zcx = np.linalg.inv(cov_eps_zcx)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		err_x = cov_eps_zcx - cov_eps_z
		grad_x = -0.5 * gamma * np.linalg.inv(cov_eps_zcx).T + dual_eps + penalty_c* err_x
		# update
		new_cov_eps_zcx = cov_eps_zcx - grad_x * ss_x
		#inv_eps_zcx = np.linalg.inv(new_cov_eps_zcx)
		err_x = new_cov_eps_zcx - cov_eps_z
		grad_z = 0.5 * (gamma -1) * inv_z2cz1.T + 0.5 * inv_z2cyz1.T - dual_eps- penalty_c* err_x
		new_cov_eps_z   = cov_eps_z - grad_z * ss_z

		err_x = new_cov_eps_zcx - new_cov_eps_z
		dual_eps += penalty_c * err_x
		# new combination
		# now we update A2, A21
		grad_Ax = (gamma-1) * inv_z2cz1.T @ Ax@ cov_x2cz1 +  inv_z2cyz1.T@ Ax @ cov_x2cyz1
		new_Ax = Ax - ss_a * grad_Ax

		# update Az...?
		# Az can be updated from old Ax, Az, Z1, X2, Z2
		#mat_k2 = (cov_x2y.T - cov_x1y.T@ inv_z1@cov_x2z1.T)@Ax@inv_x2cz1
		#mat_ycz2z1 =
		# check convergence
		# because Az depends on Ax, eps completely, if the two coverge, Az converges

		inv_z2cz1 = np.linalg.inv(Ax @ cov_x2cz1 @ Ax.T+new_cov_eps_z) # this is z2|z1
		inv_z2cyz1= np.linalg.inv(Ax @ cov_x2cyz1@ Ax.T+ new_cov_eps_z)

		conv_eps = np.sqrt(np.sum(err_x**2))
		conv_Ax  = np.sqrt(np.sum((new_Ax-Ax)**2))
		#conv_Az  = np.sqrt(np.sum((new_Az-Az)**2)) # check!
		if conv_eps < convthres and conv_Ax < convthres:
			conv_flag = True
			break
		else:
			Ax = new_Ax
			cov_eps_zcx = new_cov_eps_zcx
			cov_eps_z   = new_cov_eps_z
	return {"conv":conv_flag,"niter":itcnt,"Ax":Ax,"Az":Az,"cov_eps_x":cov_eps_zcx,"cov_eps_y":cov_eps_z}
'''
def GaussianADMMMvIBInc(cov_x2,cov_z1,cov_x2z1,cov_y,cov_yz1,cov_x2y,gamma,maxiter,convthres,**kwargs):
	penalty_c = kwargs['penalty']
	ss_x = kwargs["ss"]
	ss_z = kwargs["ss"]
	ss_a = kwargs["ss"]
	
	nx = cov_x2.shape[0]
	nz1 = cov_z1.shape[0]
	nz = nx+nz1
	ny = cov_y.shape[0]
	cov_eps_zcx = np.eye(nz)
	cov_eps_z   = np.eye(nz)
	dual_eps = np.zeros((nz,nz))

	rng = np.random.default_rng()
	Ax = np.zeros((nz,nx+nz1))
	Ax[:,:nz] = np.eye(nz)
	if nx+nz1-nz>0:
		Ax[:,nz:] = np.randint(0,1,size=(nz,nx+nz1-nz))
		Ax = Ax/np.linalg.norm(Ax,axis=0)
	#rnd_Ax = np.random.permutation(nx+nz1)
	#Ax = Ax[:,rnd_Ax]
	# precomputing prior matrices
	# patch matrices
	joint_yz1 = np.eye(ny+nz1)
	joint_yz1[:ny,:ny] = cov_y
	joint_yz1[:ny,ny:] = cov_yz1
	joint_yz1[ny:,:ny] = cov_yz1.T
	joint_yz1[ny:,ny:] = cov_z1
	patch_x2_yz1 = np.zeros((nx,ny+nz1))
	patch_x2_yz1[:,:ny] = cov_x2y
	patch_x2_yz1[:,ny:] = cov_x2z1
	# inversions
	inv_z1 = np.linalg.inv(cov_z1)
	inv_joint_yz1= np.linalg.inv(joint_yz1)

	# prior matrix 
	cov_x2cyz1 = cov_x2 - patch_x2_yz1 @ inv_joint_yz1 @ patch_x2_yz1.T
	cov_x2cz1 = cov_x2 - cov_x2z1 @ inv_z1 @ cov_x2z1.T
	inv_x2cz1 = np.linalg.inv(cov_x2cz1)

	patch_yz1_invz1 = cov_yz1 @ inv_z1
	prior_z1x2_cond_z1 = cov_x2z1.T @ inv_x2cz1 @ cov_x2z1
	prior_yx2_cond_z1  = cov_x2y.T @ inv_x2cz1 @ cov_x2z1

	cpy_cov_eps_z = Ax[:,:nx] @ cov_x2cz1 @ Ax[:,:nx].T + cov_eps_z
	inv_z2cz1 = np.linalg.inv(cpy_cov_eps_z)
	cpy_cov_eps_z = Ax[:,:nx] @ cov_x2cyz1@ Ax[:,:nx].T + cov_eps_z
	inv_z2cyz1  = np.linalg.inv(cpy_cov_eps_z)

	conv_flag = False
	itcnt = 0

	# patch
	patch_x2cz1 = np.zeros((nx+nz1,nx+nz1))
	patch_x2cz1[:nx,:nx] = cov_x2cz1
	patch_x2cyz1 = np.zeros((nx+nz1,nx+nz1))
	patch_x2cyz1[:nx,:nx] = cov_x2cyz1
	while itcnt < maxiter:
		itcnt += 1
		err_x = cov_eps_zcx - cov_eps_z
		grad_x = -0.5 * gamma * np.linalg.inv(cov_eps_zcx).T + dual_eps + penalty_c* err_x
		# update
		new_cov_eps_zcx = cov_eps_zcx - grad_x * ss_x
		err_x = new_cov_eps_zcx - cov_eps_z
		grad_z = 0.5 * (gamma -1) * inv_z2cz1.T + 0.5 * inv_z2cyz1.T - dual_eps- penalty_c* err_x
		new_cov_eps_z   = cov_eps_z - grad_z * ss_z

		err_x = new_cov_eps_zcx - new_cov_eps_z
		dual_eps += penalty_c * err_x
		# new combination
		cpy_cov_eps_z = Ax[:,:nx] @ cov_x2cz1 @ Ax[:,:nx].T + new_cov_eps_z
		inv_z2cz1 = np.linalg.inv(cpy_cov_eps_z) # this is z2|z1
		cpy_cov_eps_z = Ax[:,:nx] @ cov_x2cyz1@ Ax[:,:nx].T + new_cov_eps_z
		inv_z2cyz1= np.linalg.inv(cpy_cov_eps_z)
		# now we update A2, A21
		grad_Ax = (gamma-1) * inv_z2cz1.T @ Ax@ patch_x2cz1 + inv_z2cyz1.T@ Ax @ patch_x2cyz1
		new_Ax = Ax - ss_a * grad_Ax

		conv_eps = np.sqrt(np.sum(err_x**2))
		conv_Ax  = np.sqrt(np.sum((new_Ax-Ax)**2))
		if conv_eps < convthres and conv_Ax < convthres:
			conv_flag = True
			break
		else:
			Ax = new_Ax
			cov_eps_zcx = new_cov_eps_zcx
			cov_eps_z   = new_cov_eps_z
	if conv_flag:
		ker_z = Ax[:,:nx] @ cov_x2cz1 @ Ax[:,:nx].T + new_cov_eps_z
		ker_zy = Ax[:,:nx]@ cov_x2cyz1@ Ax[:,:nx].T + new_cov_eps_z
		# calculate mizxcz1
		mizxcz1 = ut.calcGMI(ker_z,cov_eps_zcx)
		mizycz1 = ut.calcGMI(ker_z,ker_zy)
		# calculate mizycz1
	else:
		mizxcz1 = 0
		mizycz1 = 0
	return {"conv":conv_flag,"niter":itcnt,"Ax":Ax,"cov_eps_x":cov_eps_zcx,"cov_eps_y":cov_eps_z,"mixcz":mizxcz1,"miycz":mizycz1}

'''
def GaussianMvIBCc(cov_x1,cov_x2,cov_y,cov_x12,cov_x1y,cov_x2y,nc,gamma1,gamma2,maxiter,convthres,**kwargs):
	penalty_c = kwargs["penalty"]
	ss_all = kwargs["ss"]
	inv_x1 = np.linalg.inv(cov_x1)
	inv_x2 = np.linalg.inv(cov_x2)
	inv_y  = np.linalg.inv(cov_y)

	cov_x_all = np.block([[cov_x1,cov_x12],[cov_x12.T,cov_x2]])
	cov_xallx1 = np.block([[cov_x1],[cov_x12.T]])
	cov_xallx2 = np.block([[cov_x12],[cov_x2]])
	cov_xally  = np.block([[cov_x1y],[cov_x2y]])
	cov_xallcx1 = cov_x_all - cov_xallx1 @ inv_x1 @ cov_xallx1.T
	cov_xallcx2 = cov_x_all - cov_xallx2 @ inv_x2 @ cov_xallx2.T
	cov_xallcy  = cov_x_all - cov_xally  @ inv_y  @ cov_xally.T

	cov_xallcy  = cov_x_all - cov_xally @ inv_y @ cov_xally.T

	#Ax = np.zeros((nc,cov_x1.shape[0]+cov_x2.shape[0]))
	#Ax[:,:nc] = np.eye(nc)
	Ax = np.block([np.eye(nc),np.eye(nc)])
	#rng = np.random.default_rng()
	#Ax[:,nc:] = rng.random((nc,cov_x1.shape[0]+cov_x2.shape[0]-nc))
	#Ax = Ax / np.linalg.norm(Ax,axis=0)
	cov_eps_c = np.eye(nc)
	cov_eps_x1 = np.eye(nc)
	cov_eps_x2 = np.eye(nc)
	dual_x1 = np.zeros(cov_eps_x1.shape)
	dual_x2 = np.zeros(cov_eps_x2.shape)

	# precomputing inversion
	inv_eps_x1 = np.linalg.inv(Ax @ cov_xallcx1 @ Ax.T + cov_eps_x1)
	inv_eps_x2 = np.linalg.inv(Ax @ cov_xallcx2 @ Ax.T + cov_eps_x2)
	inv_eps_c  = np.linalg.inv(Ax @ cov_x_all @ Ax.T + cov_eps_c)
	inv_eps_cy = np.linalg.inv(Ax @ cov_xallcy @ Ax.T + cov_eps_c)
	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt +=1
		# gradient 1
		err_x1 = cov_eps_x1 - cov_eps_c
		grad_x1 = -0.5 * gamma1 * inv_eps_x1.T + dual_x1 + penalty_c * err_x1
		new_eps_x1 = cov_eps_x1 - ss_all * grad_x1
		# gradient 2
		err_x2 = cov_eps_x2 - cov_eps_c
		grad_x2 = -0.5 * gamma2 * inv_eps_x2.T + dual_x2 + penalty_c * err_x2
		new_eps_x2 = cov_eps_x2 - ss_all * grad_x2
		# gradient common
		err_x1 = new_eps_x1 - cov_eps_c
		err_x2 = new_eps_x2 - cov_eps_c
		grad_c = 0.5 * (gamma1 + gamma2 -1) * inv_eps_c.T + 0.5 * inv_eps_cy.T \
				 - dual_x1 - penalty_c * err_x1 - dual_x2 - penalty_c* err_x2
		new_eps_c = cov_eps_c - ss_all * grad_c

		# dual updates
		err_x1 = new_eps_x1 - new_eps_c
		err_x2 = new_eps_x2 - new_eps_c
		dual_x1 += penalty_c * err_x1
		dual_x2 += penalty_c * err_x2

		# inversion update
		inv_eps_x1 = np.linalg.inv(Ax @ cov_xallcx1 @ Ax.T + new_eps_x1)
		inv_eps_x2 = np.linalg.inv(Ax @ cov_xallcx2 @ Ax.T + new_eps_x2)
		inv_eps_c  = np.linalg.inv(Ax @ cov_x_all   @ Ax.T + new_eps_c)
		inv_eps_cy = np.linalg.inv(Ax @ cov_xallcy  @ Ax.T + new_eps_c)

		# update the Ax
		grad_Ax = -gamma1 * inv_eps_x1.T @ Ax @ cov_xallcx1 -gamma2 * inv_eps_x2.T @ Ax @ cov_xallcx2\
					+(gamma1+gamma2-1) * inv_eps_c.T @ Ax @ cov_x_all + inv_eps_cy.T @ Ax @ cov_xallcy
		new_Ax = Ax - ss_all * grad_Ax

		conv_x1 = np.sqrt(np.sum(err_x1 ** 2))
		conv_x2 = np.sqrt(np.sum(err_x2 ** 2))
		conv_A  = np.sqrt(np.sum((new_Ax-Ax)**2))
		#print(conv_x1,conv_x2,conv_A)
		if conv_x1 < convthres and conv_x2 < convthres and conv_A<convthres:
			conv_flag = True
			break
		else:
			Ax= new_Ax
			cov_eps_x1 = new_eps_x1
			cov_eps_x2 = new_eps_x2
			cov_eps_c  = new_eps_c
			Ax = new_Ax
	cov_zc = Ax @ cov_x_all @ Ax.T + cov_eps_c
	cov_zcy = Ax@ cov_xallcy @ Ax.T + cov_eps_c
	if conv_flag:
		micx = ut.calcGMI(cov_zc,cov_eps_c)
		micy = ut.calcGMI(cov_zc,cov_zcy)
	else:
		micx = 0
		micy = 0
	return {"conv":conv_flag,"niter":itcnt,"cov_eps_z":cov_eps_c,"Ax":Ax,"cov_z":cov_zc,"cov_zcy":cov_zcy,"micx":micx,"micy":micy}
'''

def GaussianMvIBCondCc(cov_x1,cov_x2,cov_y,cov_x12,cov_x1y,cov_x2y,nc,gamma1,gamma2,maxiter,convthres,**kwargs):
	# distributed learning version
	(nx1,nx2) = cov_x12.shape
	rng = np.random.default_rng()
	penalty_c = kwargs["penalty"]
	ss_s = kwargs["ss"]
	#Ax1 = np.zeros((nc,cov_x1.shape[0]))
	#Ax1[:nc,:nc] = np.eye(nc)
	tmp_rnd1 = rng.random((nc,nx1-nc))
	tmp_rnd1 /= np.linalg.norm(tmp_rnd1,axis=0)
	Ax1 = np.block([np.eye(nc),tmp_rnd1])
	#Ax2 = np.zeros((nc,cov_x2.shape[0]))
	#Ax2[:nc,:nc] = np.eye(nc)
	tmp_rnd2 = rng.random((nc,nx2-nc))
	tmp_rnd2 /= np.linalg.norm(tmp_rnd2,axis=0)
	Ax2 = np.block([np.eye(nc),tmp_rnd2])
	cov_eps_x1 = np.eye(nc)
	cov_eps_x2 = np.eye(nc)

	cov_x1cy = cov_x1 - cov_x1y @ np.linalg.inv(cov_y) @ cov_x1y.T
	cov_x2cy = cov_x2 - cov_x2y @ np.linalg.inv(cov_y) @ cov_x2y.T

	# create common info
	cov_zc = 0.5 * (Ax1@cov_x1 @ Ax1.T + cov_eps_x1 + Ax2@cov_x2@Ax2.T + cov_eps_x2)
	cov_zcy = 0.5 * (Ax1@cov_x1cy@Ax1.T + cov_eps_x1 + Ax2@cov_x2cy@Ax2.T+cov_eps_x2)

	dual_x1= Ax1 @ cov_x1 @ Ax1.T+cov_eps_x1 - cov_zc
	dual_x2= Ax2 @ cov_x2 @ Ax2.T+cov_eps_x2 - cov_zc
	dual_x1y = Ax1@cov_x1cy @ Ax1.T+cov_eps_x1 -cov_zcy
	dual_x2y = Ax2@cov_x2cy @ Ax2.T+cov_eps_x2 - cov_zcy


	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		# gradients
		err_x1 = Ax1@cov_x1  @ Ax1.T + cov_eps_x1 - cov_zc
		err_y1 = Ax1@cov_x1cy@ Ax1.T + cov_eps_x1 - cov_zcy
		grad_x1 = -0.5 * gamma1 * np.linalg.inv(cov_eps_x1).T + dual_x1 + penalty_c * err_x1 + dual_x1y + penalty_c * err_y1
		new_eps_x1 = cov_eps_x1 -  ss_s * grad_x1
		err_x2 = Ax2@cov_x2 @ Ax2.T + cov_eps_x2 - cov_zc
		err_y2 = Ax2@cov_x2cy@Ax2.T + cov_eps_x2 - cov_zcy
		grad_x2 = -0.5 * gamma2 * np.linalg.inv(cov_eps_x2).T + dual_x2 + penalty_c * err_x2 + dual_x2y + penalty_c * err_y2
		new_eps_x2 = cov_eps_x2 - ss_s * grad_x2

		# zc update
		grad_c = 0.5 * (gamma1+gamma2-1) * np.linalg.inv(cov_zc).T -dual_x1 - penalty_c * err_x1 - dual_x2 - penalty_c* err_x2
		new_zc = cov_zc- ss_s * grad_c
		grad_cy = 0.5 * np.linalg.inv(cov_zcy).T -dual_x1y - penalty_c*err_y1 - dual_x2y - penalty_c* err_y2
		new_zcy = cov_zcy- ss_s * grad_cy
		# dual update
		tmp_cov_z1 = Ax1 @ cov_x1 @ Ax1.T + new_eps_x1
		tmp_cov_z1y= Ax1@ cov_x1cy @ Ax1.T + new_eps_x1
		tmp_cov_z2 = Ax2 @ cov_x2 @ Ax2.T + new_eps_x2
		tmp_cov_z2y= Ax2@ cov_x2cy @ Ax2.T + new_eps_x2

		err_x1 = tmp_cov_z1 - new_zc
		err_x2 = tmp_cov_z2 - new_zc
		dual_x1 += penalty_c * err_x1
		dual_x2 += penalty_c * err_x2
		err_x1y = tmp_cov_z1y -new_zcy
		dual_x1y += penalty_c* err_x1y 
		err_x2y = tmp_cov_z2y -new_zcy
		dual_x2y += penalty_c * err_x2y 

		
		grad_Ax1 = (dual_x1.T @ Ax1 @ cov_x1.T + dual_x1 @ Ax1 @ cov_x1) + penalty_c * err_x1 @ (Ax1 @ cov_x1 + Ax1 @ cov_x1.T) \
					+ (dual_x1y.T @ Ax1 @ cov_x1cy.T + dual_x1y @ Ax1 @ cov_x1cy) + penalty_c * err_x1y @ (Ax1 @ cov_x1cy + Ax1 @ cov_x1cy.T)
		new_Ax1 = Ax1 - ss_s * grad_Ax1
		
		grad_Ax2 = (dual_x2.T @ Ax2 @ cov_x2.T + dual_x2 @ Ax2 @ cov_x2) + penalty_c * err_x2 @ (Ax2 @ cov_x2 + Ax2 @ cov_x2.T) \
					+ (dual_x2y.T @ Ax2 @ cov_x2cy.T + dual_x2y @ Ax2 @ cov_x2cy) + penalty_c * err_x2y @ (Ax2 @ cov_x2cy + Ax2 @ cov_x2cy.T)
		new_Ax2 = Ax2 - ss_s * grad_Ax2

		# check error
		# convergence check
		conv_z1 = np.sqrt(np.sum(err_x1 ** 2))
		conv_z2 = np.sqrt(np.sum(err_x2 ** 2))
		conv_z1y= np.sqrt(np.sum(err_x1y** 2))
		conv_z2y= np.sqrt(np.sum(err_x2y** 2))
		conv_Ax1= np.sqrt(np.sum((new_Ax1-Ax1)**2))
		conv_Ax2= np.sqrt(np.sum((new_Ax2-Ax2)**2))
		conv_vec =np.array([conv_z1,conv_z2,conv_z1y,conv_z2y,conv_Ax1,conv_Ax2],dtype="float32") 
		#print(conv_vec)
		if np.all(conv_vec<convthres):
			conv_flag = True
			break
		else:
			cov_eps_x1 = new_eps_x1
			cov_eps_x2 = new_eps_x2
			cov_zc = new_zc
			cov_zcy = new_zcy
			Ax1 = new_Ax1
			Ax2 = new_Ax2
	return {"conv":conv_flag,"niter":itcnt,"cov_z":cov_zc,"cov_zcy":cov_zcy,"Ax1":Ax1,"Ax2":Ax2,"cov_eps1":cov_eps_x1,"cov_eps2":cov_eps_x2}


def GaussianMvIBIncBA(cov_x2,cov_z1,cov_x2z1,cov_y,cov_yz1,cov_x2y,gamma,maxiter,convthres,**kwargs):
	nx = cov_x2.shape[0]
	nz1 = cov_z1.shape[0]
	nz = nx+nz1
	ny = cov_y.shape[0]
	cov_eps_zcx = np.eye(nz)

	rng = np.random.default_rng()
	Ax = np.zeros((nz,nx+nz1))
	Ax[:,:nz] = np.eye(nz)
	if nx+nz1-nz>0:
		Ax[:,nz:] = np.randint(0,1,size=(nz,nx+nz1-nz))
		Ax = Ax/np.linalg.norm(Ax,axis=0)
	#rnd_Ax = np.random.permutation(nx+nz1)
	#Ax = Ax[:,rnd_Ax]
	# precomputing prior matrices
	# inversions
	inv_z1 = np.linalg.inv(cov_z1)
	cov_ycz1 = cov_y - cov_yz1 @ inv_z1 @ cov_yz1.T
	inv_ycz1 = np.linalg.inv(cov_ycz1)

	# prior matrix
	patch_yx2_z1x2 = np.concatenate((cov_x2y.T,cov_x2z1.T),axis=0)
	joint_yz1 = np.block([[cov_y,cov_yz1],[cov_yz1.T,cov_z1]])
	cov_x2cyz1 = cov_x2 - patch_yx2_z1x2.T @ np.linalg.inv(joint_yz1) @ patch_yx2_z1x2
	cov_x2cz1 = cov_x2 - cov_x2z1 @ inv_z1 @ cov_x2z1.T
	inv_x2cz1 = np.linalg.inv(cov_x2cz1)

	# block matrix
	cov_x2z1_y = np.concatenate((cov_x2y,cov_yz1.T),axis=0)
	cov_x2z1_z1= np.concatenate((cov_x2z1,cov_z1),axis=0)

	conv_flag = False
	itcnt = 0
	while itcnt < maxiter:
		itcnt += 1
		# compute the needed matrices
		mat_z2_z1 = Ax[:,:nx] @ cov_x2cz1 @ Ax[:,:nx].T + cov_eps_zcx
		mat_z2_yz1 = Ax[:,:nx] @ cov_x2cyz1 @ Ax[:,:nx].T + cov_eps_zcx
		inv_z2_z1 = np.linalg.inv(mat_z2_z1)
		inv_z2_yz1 = np.linalg.inv(mat_z2_yz1)
		# covariance update
		mat_K2 = (1-1/gamma)*inv_z2_z1 + 1/gamma * inv_z2_yz1
		new_eps_zcx = np.linalg.inv(mat_K2)

		# mean operator update
		helper_mat = Ax[:,:nx]@ (cov_x2y - cov_x2z1 @ inv_z1 @ cov_yz1.T)
		helper_mat_yx2= cov_x2y.T - cov_yz1 @ inv_z1 @ cov_x2z1.T
		part_yx2 = (cov_x2y.T - cov_yz1 @ inv_z1 @ cov_x2z1.T) @ inv_x2cz1
		part_yz1 = helper_mat.T @ inv_z2_z1 @ Ax @ np.concatenate((cov_x2z1,cov_z1),axis=0) @ inv_z1 \
					- helper_mat_yx2 @ cov_x2cz1@ cov_x2z1@ inv_z1
		new_Ax = new_eps_zcx @ inv_z2_yz1 @ helper_mat @ inv_ycz1 @ np.concatenate((part_yx2,part_yz1),axis=1)

		# convergence
		conv_eps = np.sum((new_eps_zcx - cov_eps_zcx)**2)
		conv_Ax  = np.sum((new_Ax - Ax)**2)
		if conv_eps < convthres and conv_Ax < convthres:
			conv_flag = True
			break
		else:
			cov_eps_zcx = new_eps_zcx
			Ax = new_Ax
	if conv_flag:
		ker_z = Ax[:,:nx] @ cov_x2cz1 @ Ax[:,:nx].T + cov_eps_zcx
		ker_zy =Ax[:,:nx] @ cov_x2cyz1 @ Ax[:,:nx].T + cov_eps_zcx
		mizxcz1 = ut.calcGMI(ker_z,cov_eps_zcx)
		mizycz1 = ut.calcGMI(ker_zy,cov_eps_zcx)
	else:
		mizxcz1 = 0
		mizycz1 = 0
	return {"conv":conv_flag,"niter":itcnt,"mixcz":mizxcz1,"miycz":mizycz1,"Axz":Ax,"cov_eps":cov_eps_zcx}