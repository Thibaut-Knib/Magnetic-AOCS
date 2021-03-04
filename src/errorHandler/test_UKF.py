
import sys
sys.path.append('..')
import filter as flt
import numpy as np
import os as os
import shutil as shutil
from scao.quaternion import Quaternion

os.chdir("../../tst/sim")
sys.path.append(os.getcwd())
print(os.getcwd())
if (not os.path.isfile('conf.py')):
    shutil.copy2('conf.default.py', 'conf.py')
from conf import *

dim = 6 # 3 pour les quaternions, 3 pour la rotation
q0 = Quaternion(1,0,0,0)
W0 = np.array([[1.0],[0.0],[0.0]])
P0 = np.eye(dim)
Qcov = np.eye(dim)
Rcov = np.eye(6) # ATTENTION, dimension de la mesure ici, pas de l'état
dt = 1
I = np.diag((m * (ly ** 2 + lz ** 2) / 3, m * (lx ** 2 + lz ** 2) / 3,
             m * (lx ** 2 + ly ** 2) / 3))  # Tenseur inertie du satellite
ukf = flt.UKF(q0,W0,I,gyroModel[0],P0,Qcov,Rcov,dt)

state = ukf.curState
#print(state)


################################
#  Pour un passage de boucle
################################

# Entree
WM = np.array([[1.], [0], [0]])
BM = np.array([[0], [0.9], [0]])
B = np.array([[0], [1], [0.]])

#Qcorr, Wcorr = ukf.errorCorrection(WM, BM, B)
xk_ = ukf.errorCorrection(WM, BM, B)
#print(xk_)

print(q0.vec())
