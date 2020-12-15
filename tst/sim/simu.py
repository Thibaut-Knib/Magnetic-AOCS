# coding=utf-8
import sys
import os
#os.chdir(r"C:\Users\thiba\Documents\Polytechnique\P3A\Magnetic-AOCS\tst\sim")
sys.path.append(os.path.join(*['..'] * 2))
print(os.getcwd())
sys.path.append("../../src/")
sys.path.append("src/")

import shutil
import vpython as vp
from math import *
import numpy as np
from simulator import Simulator
from scao.scao import SCAO
from scao.stabAlgs import PIDRW, PIDMT
from environnement.environment import Environment
from environnement.orbit import Orbit
from hardware.hardwares import Hardware
import matplotlib.pyplot as plt
from scao.quaternion import Quaternion
from errorHandler.errorModel import MeasuredWorld
import seaborn as sns

###############################
# Paramètres de la simulation #
###############################
if (not os.path.isfile('conf.py')):
    shutil.copy2('conf.default.py', 'conf.py')
from conf import *


for kkk in range(100):
    print("Etape ", kkk)
    epsilon_gyro = 10**-(random()*4+2)  #entre 10^-3 et 10^-7
    standard_deviation_gyro = np.array([[epsilon_gyro],[epsilon_gyro],[epsilon_gyro]])
    gyroModel = (biais_gyro, standard_deviation_gyro, scaling_factor_gyro, drift_gyro)

    ###################################
    # Initialisation de la simulation #
    ###################################
    # Initialisation des variables de simulation
    t = 0
    dw = np.array([[0.], [0.], [0.]])  # vecteur de l'accélération angulaire des RW
    M = np.array([[0.], [0.], [0.]])  # vecteur du moment magnétique des bobines
    I = np.diag((m * (ly ** 2 + lz ** 2) / 3, m * (lx ** 2 + lz ** 2) / 3,
                 m * (lx ** 2 + ly ** 2) / 3))  # Tenseur inertie du satellite
    L0 = np.dot(I, W0)

    # Environnement
    orbite = Orbit(omega, i, e, r_p, mu, tau)
    environnement = Environment(B_model)

    # paramètres hardware
    n_windings = 500
    r_wire = 125e-6
    r_coil = 75e-4
    U_max = 5
    mu_rel = 31
    J = 1  # moment d'inertie des RW
    # r_coil, r_wire, n_coils, mu_rel, U_max
    mgt_parameters = r_coil, r_wire, n_windings, mu_rel, U_max
    #####
    hardW = Hardware(mgt_parameters, 'custom coil')

    # Initialisation du champ magnétique:
    orbite.setTime(t)
    environnement.setPosition(orbite.getPosition())
    B = environnement.getEnvironment()  # dans le référentiel du satellite

    # Simulateur
    Q0 = Quaternion(1, 0, 0, 0)
    sim = Simulator(dt, L0, Q0)
    Qt = Quaternion(1, 0, 0, 0) # Quaternion objectif

    #Monde mesuré
    mWorld = MeasuredWorld(gyroModel,magneticModel,dt, Q0)

    # Algortihmes de stabilisation
    stab = SCAO(PIDRW(RW_P, RW_dP, RW_D), PIDMT(MT_P, MT_dP, MT_D), SCAOratio, I, J)  # stabilisateur



    sns.set(context = 'talk', style = 'white')
    plt.rcParams["figure.figsize"] = (8,4)

    #####################
    # Boucle principale #
    #####################
    alpha_max = 0
    while t<dt*2000:
        # on récupère la valeur actuelle du champ magnétique et on actualise l'affichage du champ B
        orbite.setTime(t)  # orbite.setTime(t)
        environnement.setPosition(orbite.getPosition())
        B = environnement.getEnvironment()  # dans le référentiel géocentrique

        # on récupère le prochain vecteur rotation (on fait ube étape dans la sim)
        W = sim.getNextIteration(M, dw, J, B, I)

        #Update monde mesuré
        mWorld.getNextIteration(W,B,sim.Q)

        # Sauvegarder les valeurs de simulation actuelles: (valeurs mesurées)
        stab.setAttitude(mWorld.QM)
        stab.setRotation(mWorld.WM)
        stab.setMagneticField(sim.Q.V2R(mWorld.BM))

        # Prise de la commande de stabilisation
        dw, M = stab.getCommand(Qt)  # dans Rv
        U, M = hardW.getRealCommand(dw, M)

        t += dt

        alpha_max = max(sim.Q.angle(),alpha_max)

    plt.plot([-log10(epsilon_gyro)],[alpha_max], marker = 'o', color = 'black')

plt.xlabel('$-\log_{10}(\sigma_{gyromètre}$)')
plt.ylabel("Écart angulaire maximal à la position d'origine")
plt.title("Stabilité du satellite en fonction de l'écart-type de l'erreur sur les gyromètres")
plt.show()
