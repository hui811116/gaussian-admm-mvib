import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# loading the results
with open("result_sv_v1_ba.npy",'rb') as fid:
	res_v1 = np.load(fid)
with open("result_sv_v2_ba.npy","rb") as fid:
	res_v2 = np.load(fid)
with open("result_merge_ba.npy",'rb') as fid:
	res_merge = np.load(fid)

# basic format of single view results
# (gamma,mizx,mizy,niter,conv)

fs_lab= 18
fs_tick = 16
fs_leg = 16
# plot the mi
fig,ax = plt.subplots()
ax.plot(res_merge[:,1],res_merge[:,2],label=r"Merge",color="k",linestyle="dotted")
ax.plot(res_v1[:,1],res_v1[:,2],label=r"Single $V_1$",color="tab:green",linestyle="dashdot")
ax.plot(res_v2[:,1],res_v2[:,2],label=r"Single $V_2$",color="tab:olive",linestyle="dashed")
ax.grid("on")
ax.set_xlabel(r"$I(\{Z\};X)$ nats",fontsize=fs_lab)
ax.set_ylabel(r"$I(\{Z\};Y)$ nats",fontsize=fs_lab)
ax.tick_params(axis="both",labelsize=fs_tick)
ax.legend(loc="best", fontsize=fs_leg)
plt.tight_layout()

plt.show()

# plot the progressing

# plot the niter