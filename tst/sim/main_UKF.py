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
from errorHandler import filter as flt
from errorHandler.state import State
import seaborn as sns

###############################
# Paramètres de la simulation #
###############################
if (not os.path.isfile('conf.py')):
    shutil.copy2('conf.default.py', 'conf.py')
from conf import *

###################################
# Initialisation de la simulation #
###################################
# Initialisation des variables de simulation
t = 0
nbit = 0
dw = np.array([[0.], [0.], [0.]])  # vecteur de l'accélération angulaire des RW
M = np.array([[0.], [0.], [0.]])  # vecteur du moment magnétique des bobines
I = np.diag((m * (ly ** 2 + lz ** 2) / 3, m * (lx ** 2 + lz ** 2) / 3,
             m * (lx ** 2 + ly ** 2) / 3))  # Tenseur inertie du satellite
L0 = np.dot(I, W0)
Wr = []
qs = []
qm = []
qc = []
sigma1 = []

# Environnement
orbite = Orbit(omega, i, e, r_p, mu, tau)
environnement = Environment(B_model)

# Hardware
#####
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
mWorld = MeasuredWorld(gyroModel,magneticModel,sunSensorModel,dt, Q0)

# Algortihmes de stabilisation
stab = SCAO(PIDRW(RW_P, RW_dP, RW_D), PIDMT(MT_P, MT_dP, MT_D), SCAOratio, I, J)  # stabilisateur

############################
# Initialisation du filtre #
############################

dimState = 9 # 3 pour les quaternions, 3 pour la rotation, 3 pour les biais
dimObs = 12
CovRot = 1e-6  #Paramètre à régler pour le bon fonctionnement du filtre

P0 = np.eye(dimState)*1e-2
#variance d'évolution
Qcov = np.zeros((dimState,dimState))
Qcov[0:3,0:3] = 1e-4*np.eye(3)
Qcov[3:6,3:6] = CovRot*np.eye(3)
Qcov[6:9,6:9] = 1e-8*np.eye(3)
Rcov = np.zeros((dimObs,dimObs)) # ATTENTION, dimension de la mesure ici, pas de l'état
Rcov[0,0] = gyroModel[1][0,0]**2
Rcov[1,1] = gyroModel[1][1,0]**2
Rcov[2,2] = gyroModel[1][2,0]**2
Rcov[3,3] = magneticModel[1][0,0]**2
Rcov[4,4] = magneticModel[1][1,0]**2
Rcov[5,5] = magneticModel[1][2,0]**2

ukf = flt.UKF(Q0,W0,I,gyroModel[0],P0,Qcov,Rcov,dt)

############################
# Initialisation graphique #
############################
if (Affichage_3D):
    ux = vp.vector(1, 0, 0)
    uy = vp.vector(0, 1, 0)
    uz = vp.vector(0, 0, 1)
    # trièdre (z,x,y)
    axe_x_r = vp.arrow(pos=vp.vector(0, 0, 0), axis=10 * ux, shaftwidth=0.5, color=vp.vector(1, 0, 0))
    axe_y_r = vp.arrow(pos=vp.vector(0, 0, 0), axis=10 * uy, shaftwidth=0.5, color=vp.vector(0, 1, 0))
    axe_z_r = vp.arrow(pos=vp.vector(0, 0, 0), axis=10 * uz, shaftwidth=0.5, color=vp.vector(0, 0, 1))
    # création du satellite avec son repère propre
    axe_x_s = vp.arrow(pos=vp.vector(10, 10, 10), axis=10 * ux, shaftwidth=0.1, color=vp.vector(1, 0, 0))
    axe_y_s = vp.arrow(pos=vp.vector(10, 10, 10), axis=10 * uy, shaftwidth=0.1, color=vp.vector(0, 1, 0))
    axe_z_s = vp.arrow(pos=vp.vector(10, 10, 10), axis=10 * uz, shaftwidth=0.1, color=vp.vector(0, 0, 1))
    sugarbox = vp.box(pos=vp.vector(10, 10, 10), size=vp.vector(lx, ly, lz), axis=vp.vector(0, 0, 0), up=uy)
    satellite = vp.compound([axe_x_s, axe_y_s, axe_z_s, sugarbox])
    # vecteur champ B
    b_vector = vp.arrow(pos=vp.vector(-5, -5, -5), axis=1e5 * vp.vector(B[0][0], B[1][0], B[2][0]), shaftwidth=0.1,
                        color=vp.vector(1, 1, 1))


####################
# Fonctions utiles #
####################
def plotAttitude():
    color = ['b','r','g','black']
    for i in range(4):
        plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.vec()[i, 0] for q in qs], color = color[i])
        plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.vec()[i, 0] for q in qc], color = color[i], linestyle = 'dotted')
    plt.xlabel("t (orbits)")
    plt.ylabel("Quaternion component")
    plt.show()


#####################
# Boucle principale #
#####################
output = {'t': [], 'M': [], 'U': []}
outputW = {'t': [], 'W': [], 'WM': [], 'WC': [], 'sig': []}
outputB = {'t': [], 'B': [], 'BM': []}
while t<dt*5000:
    # on récupère la valeur actuelle du champ magnétique et on actualise l'affichage du champ B
    orbite.setTime(t)  # orbite.setTime(t)
    environnement.setAttitudePosition(sim.Q,orbite.getPosition())
    B,uSun = environnement.getEnvironment()  # dans le référentiel géocentrique

    # on récupère le prochain vecteur rotation (on fait ube étape dans la sim)
    W = sim.getNextIteration(M, dw, J, B, I)

    #Update monde mesuré
    mWorld.getNextIteration(W,B,uSun,sim.Q)
    # Correction des erreurs avec un filtre
    ukf.errorCorrection(t, mWorld.WM, mWorld.BM, mWorld.uSunM, B, M)
    # Sauvegarder les valeurs de simulation actuelles: (valeurs mesurées)
    stab.setAttitude(ukf.curState.Q)
    stab.setRotation(ukf.curState.W)
    stab.setMagneticField(sim.Q.V2R(mWorld.BM)) #non recalé

    # Enregistrement de variables pour affichage
    Wr.append(np.linalg.norm(W))
    qs.append(sim.Q)
    qm.append(mWorld.QM)
    qc.append(ukf.curState.Q)
    sigma1.append(sqrt(ukf.P[0,0]))

    # Prise de la commande de stabilisation
    dw, M = stab.getCommand(Qt)  # dans Rv
    U, M = hardW.getRealCommand(dw, M)

    # affichage de données toute les 10 itérations
    if nbit % 10 == 0:
        print("t :",t)
        #print(
        #"W :", str(W[:, 0]), "|| norm :", str(np.linalg.norm(W)), "|| dw :", str(dw[:, 0]), "|| B :", str(B[:, 0]),
        #"|| Q :", str(sim.Q.axis()[:, 0]), "|| M :", str(np.linalg.norm(M)))

    # Actualisation de l'affichage graphique
    if (Affichage_3D):
        b_vector.axis = 1e5 * vp.vector(B[0][0], B[1][0], B[2][0])
        satellite.rotate(angle=np.linalg.norm(W) * dt, axis=vp.vector(W[0][0], W[1][0], W[2][0]),
                         origin=vp.vector(10, 10, 10))

    # Rate : réalise 25 fois la boucle par seconde
    #vp.rate(fAffichage)  # vp.rate(1/dt)
    nbit += 1
    t += dt
    output['t'].append(t)
    output['M'].append(M)
    output['U'].append(U)

    outputW['t'].append(t)
    outputW['W'].append(W)
    outputW['WM'].append(mWorld.WM)
    outputW['WC'].append(ukf.curState.W)
    outputW['sig'].append(ukf.P[3,3])

    outputB['t'].append(t)
    outputB['B'].append(np.linalg.norm(B))
    outputB['BM'].append(np.linalg.norm(mWorld.BM))

#plotAttitude()

#print([ukf.record['K'][i][3:6,0:6] for i in range(50,55)])
#print([ukf.record['stateOut'][i].gyroBias for i in range(310,325)])

#print(ukf.record['K'][3000])

sns.set(context = 'talk', style = 'white')
plt.rcParams["figure.figsize"] = (8,4)

#plt.plot(range(1000),ukf.record['NormKal'])
#plt.show()

plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [(q1*q2.inv()).angle() for q1,q2 in zip(qs,qc)], color = 'black')
plt.show()

plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.angle()*q.axis()[1,0] for q in qs], color = 'black', label = 'Vraie valeur')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.angle()*q.axis()[1,0] for q in qm], color = 'r', label = 'Valeur bruitée')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.angle()*q.axis()[1,0] for q in qc], color = 'g', label = 'Valeur recalé')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.angle()*q.axis()[1,0] + 3*s for  q,s in zip(qc,sigma1)], color = 'b', label = 'Encadrement à 3$\sigma$')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(qs))], [q.angle()*q.axis()[1,0] - 3*s for  q,s in zip(qc,sigma1)], color = 'b')
plt.xlabel("t (orbites)")
plt.ylabel("Élément du quaternion")
plt.title("Evolution d'un élément du quaternion recalé")
plt.legend(loc = 'best')
plt.show()

plt.plot([dt * j / orbite.getPeriod() for j in range(len(outputW['W']))],[x[0,0] for x in outputW['W']],color = 'black', label = 'Vraie valeur')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(outputW['W']))],[x[0,0]for x in outputW['WM']],color = 'r', label = 'Valeur bruitée')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(outputW['W']))],[x[0,0] for x in outputW['WC']],color = 'g', label = 'Valeur recalé')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(outputW['W']))],[x1[0,0] + 3*sqrt(x2) for x1,x2 in zip(outputW['WC'],outputW['sig'])],color = 'b', label = 'Encadrement à 3$\sigma$')
plt.plot([dt * j / orbite.getPeriod() for j in range(len(outputW['W']))],[x1[0,0] - 3*sqrt(x2) for x1,x2 in zip(outputW['WC'],outputW['sig'])],color = 'b')
plt.xlabel("t (orbites)")
plt.ylabel("Élément de la vitesse de rotation")
plt.title("Evolution d'un élément de la vitesse de rotation recalée")
plt.legend(loc = 'best')
plt.show()
