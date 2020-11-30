
import sys
sys.path.append('..')
import filter as flt
import numpy as np
from scao.quaternion import Quaternion

dim = 6 # 3 pour les quaternions, 3 pour la rotation
q0 = Quaternion(1,0,0,0)
W0 = np.array([[1.0],[0.0],[0.0]])
P0 = np.eye(dim)
Qcov = np.eye(dim)
Rcov = np.eye(6) # ATTENTION, dimension de la mesure ici, pas de l'état
dt = 1

ukf = flt.UKF(dim,q0,W0,P0,Qcov,Rcov,dt)


################################
#  Pour un passage de boucle
################################

# Entree
WM = np.array([[1.], [0], [0]])
BM = np.array([[0], [0.9], [0]])
B = np.array([[0], [1], [0.]])

Qcorr, Wcorr = ukf.errorCorrection(WM, BM, B)
