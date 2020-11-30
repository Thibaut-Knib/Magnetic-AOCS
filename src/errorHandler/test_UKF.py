
import sys
sys.path.append('..')
import filter as flt
import numpy as np
from scao.quaternion import Quaternion

dim = 6 # 3 pour les quaternions, 3 pour la rotation
q0 = Quaternion(1,0,0,0)
W0 = np.array([0,0,0])
P0 = np.eye(dim)
Qcov = np.eye(dim)
dt = 1

ufk = flt.UKF(dim,q0,W0,P0,Qcov,dt)
