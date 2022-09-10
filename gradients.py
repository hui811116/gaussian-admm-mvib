import numpy as np
import sys
import os

def naiveStepSize(prob,update,ss_init,ss_scale):
	ss_out = ss_init
	while np.any(prob + ss_out*update>=1.0  or prob + ss_out * update <= 0.0):
		ss_out *= ss_scale
		if ss_out < 1e-11:
			ss_out = 0
			break
	return ss_out

# acceleration methods
# heavyball or nesterov
