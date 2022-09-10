import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# loading the results
with open("result_all/result_sv_v1_ba.npy",'rb') as fid:
	res_v1 = np.load(fid)
with open("result_all/result_sv_v2_ba.npy","rb") as fid:
	res_v2 = np.load(fid)
with open("result_all/result_merge_ba.npy",'rb') as fid:
	res_merge = np.load(fid)
#with open("result_all/gmvib_inc_c32_ss4e-3_b32_g32_x12.npy","rb") as fid:
#	res_inc_x12 = np.load(fid)
#with open("result_all/gmvib_inc_c32_ss4e-3_b32_g32_x21.npy","rb") as fid:
#	res_inc_x21 = np.load(fid)

with open("result_all/gmvib_inc_c32_ss4e-3_b64_g64_gmin2e-3_bmax1024_x21.npy",'rb') as fid:
	res_inc_grid_x21 = np.load(fid)
with open("result_all/gmvib_incba_b128_g128_bmax1024_gmin1e-3_x12.npy",'rb') as fid:
	res_incba_x12 = np.load(fid)
#with open("result_all/gmvib_incba_b128_g128_bmax1024_gmin1e-3_x21.npy",'rb') as fid:
#	res_incba_x21 = np.load(fid)

def ibOptSelect(pairs_mi,precision):
	(mizx,mizy) = pairs_mi
	hold_dict = {}
	pre_format = "{{:.{:}f}}".format(precision)
	#print(pre_format)
	for iidx in range(len(mizx)):
		izx = mizx[iidx]
		izy = mizy[iidx]
		izx_str = pre_format.format(izx)
		if not hold_dict.get(izx_str,False):
			hold_dict[izx_str] = izy
		if izy>hold_dict[izx_str]:
			hold_dict[izx_str] = izy
	process_out = []
	for k,v in hold_dict.items():
		process_out.append([float(k),v])
	process_out.sort(key=lambda x:x[0])
	return np.array(process_out).astype("float32")

#preprocess inc results
# inc format:
# beta, gamma, iz1x1, iz1y, iz2x1cz1, iz2cz1y, niter1, niter2, conv2
#sel_idx = res_inc_x12[:,8] == 1
#combine_inc_x12 = res_inc_x12[sel_idx,:]

#sel_idx = res_inc_x21[:,8] == 1
#combine_inc_x21 = res_inc_x21[sel_idx,:]


sel_idx = res_incba_x12[:,8] == 1
combine_incba_x12 = res_incba_x12[sel_idx,:]
#sel_idx = res_incba_x21[:,8] == 1
#combine_incba_x21 = res_incba_x21[sel_idx,:]

sel_idx = res_inc_grid_x21[:,8] == 1
combine_inc_grid_x21 = res_inc_grid_x21[sel_idx,:]

# basic format of single view results
# (gamma,mizx,mizy,niter,conv)

fs_lab= 18
fs_tick = 16
fs_leg = 16
# plot the mi
fig,ax = plt.subplots()
ax.plot(res_merge[:,1],res_merge[:,2],label=r"Merge",color="k",linestyle="dotted")
#ax.plot(res_v1[:,1],res_v1[:,2],label=r"Single $V_1$",color="tab:green",linestyle="dashdot")
#ax.plot(res_v2[:,1],res_v2[:,2],label=r"Single $V_2$",color="tab:olive",linestyle="dashed")
# process the resuts
#process_inc_x12 = ibOptSelect((combine_inc_x12[:,2]+combine_inc_x12[:,4],combine_inc_x12[:,3]+combine_inc_x12[:,5]),1)
#ax.scatter(process_inc_x12[:,0],process_inc_x12[:,1],24,
#	label=r"Inc. $X_{1,2}$",color="tab:blue",marker="+")
#ax.scatter(combine_inc_x12[:,2]+combine_inc_x12[:,4],combine_inc_x12[:,3]+combine_inc_x12[:,5],24,
#	label=r"Inc. $X_{1,2}$",color="tab:blue",marker="+")

#process_inc_x21 = ibOptSelect((combine_inc_x21[:,2]+combine_inc_x21[:,4],combine_inc_x21[:,3]+combine_inc_x21[:,5]),1)
#ax.scatter(process_inc_x21[:,0],process_inc_x21[:,1],24,
#	label=r"Inc. $X_{2,1}$",color="tab:red",marker="*")
#ax.scatter(combine_inc_x21[:,2]+combine_inc_x21[:,4],combine_inc_x21[:,3]+combine_inc_x21[:,5],24,
#	label=r"Inc. $X_{2,1}$",color="tab:red",marker="*")

process_incba_x12 = ibOptSelect((combine_incba_x12[:,2]+combine_incba_x12[:,4],combine_incba_x12[:,3]+combine_incba_x12[:,5]),1)
ax.scatter(process_incba_x12[:,0],process_incba_x12[:,1],24,label=r"Inc-BA. $X_{1,2}$",color="tab:green",marker="^")
#process_incba_x21 = ibOptSelect((combine_incba_x21[:,2]+combine_incba_x21[:,4],combine_incba_x21[:,3]+combine_incba_x21[:,5]),1)
#ax.scatter(process_incba_x21[:,0],process_incba_x21[:,1],24,label=r"Inc-BA. $X_{2,1}$",color="tab:olive",marker="s")

process_grid_x21 = ibOptSelect((combine_inc_grid_x21[:,2]+combine_inc_grid_x21[:,4],combine_inc_grid_x21[:,3]+combine_inc_grid_x21[:,5]),1)
ax.scatter(process_grid_x21[:,0],process_grid_x21[:,1],24,label=r"Inc-Grid. $X_{2,1}$",color="tab:cyan",marker="o")

ax.grid("on")
ax.set_xlabel(r"$I(\{Z\};X)$ nats",fontsize=fs_lab)
ax.set_ylabel(r"$I(\{Z\};Y)$ nats",fontsize=fs_lab)
ax.tick_params(axis="both",labelsize=fs_tick)
ax.legend(loc="best", fontsize=fs_leg)
ax.set_xlim([0,6.25])
#ax.set_ylim([0.8,1.0])
plt.tight_layout()
#plt.savefig("figure_full_ibcurve.eps",format="eps")
plt.show()

# plot the progressing

# plot the niter