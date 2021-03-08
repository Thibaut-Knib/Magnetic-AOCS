import numpy as np
from random import random
from math import pi
###############################
# Paramètres de la simulation #
###############################
# Temps
dt = 1 #pas de temps de la simulation
fAffichage = 25 #fréquence d'affichage
Affichage_3D = False

# Géométrie
lx,ly,lz = 10,10,10 #longueur du satellit selon les axes x,y,z
m = 1 #masse du satellite

# Mouvement
W0 = 0*np.array([[2*(random()-0.5)] for i in range(3)]) #rotation initiale dans le référentiel R_r


# SCAO parameters
SCAOratio = 0
RW_P = 3
RW_dP = 2
RW_D = 3
MT_P = 5e3
MT_dP = 2
MT_D = 2e7

# Hardware
n_windings = 400
A_coils = 8.64e-3
M_max = 0.13 #Moment maximum des MT
J = 1 # moment d'inertie des RW

# Orbite
omega = 0
i = pi/6
e = 0.01
r_p = 7e6
mu = 3.986004418e14
tau = 0

# Environment
B_model = 'dipole'

epsilon_gyro = 0 # 1e-4
epsilon_mag = 0 # 1e-7
epsilon_sun = 0

#Error models
biais_gyro = np.array([[0],[0],[0]])
standard_deviation_gyro = np.array([[epsilon_gyro],[epsilon_gyro],[epsilon_gyro]])
scaling_factor_gyro = np.array([[1],[1],[1]])
drift_gyro = np.array([[0],[0],[0]])

biais_mag = np.array([[0],[0],[0]])
standard_deviation_mag = np.array([[epsilon_mag],[epsilon_mag],[epsilon_mag]])
scaling_factor_mag = np.array([[1],[1],[1]])
drift_mag = np.array([[0],[0],[0]])

biais_sun = np.array([[0],[0],[0],[0],[0],[0]])
standard_deviation_sun = np.array([[epsilon_sun],[epsilon_sun],[epsilon_sun],[epsilon_sun],[epsilon_sun],[epsilon_sun]])
scaling_factor_sun = np.array([[1],[1],[1],[1],[1],[1]])
drift_sun = np.array([[0],[0],[0],[0],[0],[0]])

gyroModel = (biais_gyro, standard_deviation_gyro, scaling_factor_gyro, drift_gyro)
magneticModel = (biais_mag, standard_deviation_mag, scaling_factor_mag, drift_mag)
sunSensorModel = (biais_sun, standard_deviation_sun, scaling_factor_sun, drift_sun)
