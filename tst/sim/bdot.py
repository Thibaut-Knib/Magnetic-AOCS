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

plt.ion()

###############################
# Paramètres de la simulation #
###############################
if (not os.path.isfile('conf_bdot.py')):
    shutil.copy2('tst/sim/conf.default.py', 'conf_bdot.py')
from conf_bdot import *

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

K_Bdot = 1e5

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
sim = Simulator(dt, L0)

# Algortihmes de stabilisation
stab = SCAO(PIDRW(RW_P, RW_dP, RW_D), PIDMT(MT_P, MT_dP, MT_D), SCAOratio, I, J)  # stabilisateur

############################
# Initialisation graphique #
############################
plot_3D = False

if plot_3D :
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
    b_vector = vp.arrow(pos=vp.vector(-5, -5, -5), axis=10 * vp.vector(B[0][0], B[1][0], B[2][0]), shaftwidth=0.1,
                        color=vp.vector(1, 1, 1))

    # création de l'axe du vecteur rotation
    axe_W = vp.arrow(pos=vp.vector(10, 10, 10), axis=10 * ux, shaftwidth=0.1, color=vp.vector(1, 1, 1))


plot_2D = True
liveplot2D = False
if plot_2D:

    fig, axarr = plt.subplots(2, 3)

    [ ax1, ax2, ax3, ax4, ax5, ax6] = axarr.flatten()

    linew, = ax1.plot([], [])
    linewx, = ax1.plot([], [])
    linewy, = ax1.plot([], [])
    linewz, = ax1.plot([], [])
    ax1.set_title("|| W ||")
    line2, = ax2.plot([], [])
    ax2.set_title("|| M ||")

    linebx, = ax3.plot([], [])
    lineby, = ax3.plot([], [])
    linebz, = ax3.plot([], [])
    lineb, = ax3.plot([], [])
    ax3.set_title("B")

    lineAngle, = ax4.plot([], [])
    ax4.set_title("Angle W - B")

    linedB, = ax5.plot([], [])
    ax5.set_title("dB")

    plt.show()

live_print = True
n_print = 100

####################
# Fonctions utiles #
####################
def plotAttitude():
    for i in range(4):
        plt.plot([dt * i / orbite.getPeriod() for i in range(len(qs))], [q.vec()[i, 0] for q in qs])
    plt.xlabel("t (orbits)")
    plt.ylabel("Quaternion component")
    plt.show()


#####################
# Boucle principale #
#####################
output = {'t': [], 'M': [], 'U': [], "W" : [], "W_norm" : [], "M_norm" : [],
          "Bv" : [], "B_norm" : [], "angle" : [], "dB" : []}
B_old = 0



Mr = []

while nbit < 5e3 :
    # on récupère la valeur actuelle du champ magnétique et on actualise l'affichage du champ B
    orbite.setTime(t)  # orbite.setTime(t)
    environnement.setPosition(orbite.getPosition())
    B = environnement.getEnvironment()  # dans le référentiel géocentrique
    # B *= 10  # Speedup the calculation

    # on récupère le prochain vecteur rotation (on fait ube étape dans la sim)
    W = sim.getNextIteration(M, dw, J, B, I)

    # Sauvegarder les valeurs de simulation actuelles:
    stab.setAttitude(sim.Q)
    stab.setRotation(W)
    stab.setMagneticField(B)

    # Enregistrement de variables pour affichage
    Wr.append(np.linalg.norm(W))
    qs.append(sim.Q)

    # Prise de la commande de stabilisation
    # dw, M = stab.getCommand(Quaternion(Qt[0][0], Qt[1][0], Qt[2][0], Qt[3][0]))  # dans Rv

    # B-dot law
    B_v = sim.Q.R2V(B) # B_v in Rv
    if nbit > 0 :
        dB = B_v - B_old
        M = - K_Bdot * (B_v - B_old)/dt  # M is in Rv here
        # print(B_v, B_old)
    else :
        dB = np.zeros_like(B_v)
        M = np.zeros_like(B_v)


    B_old = B_v.copy()  # Save B in The Vehicul referentiel for the next B_dot iteration

    Mr.append(np.linalg.norm(M))

    U, M = hardW.getRealCommand(dw, M)
    U = np.zeros_like(M)

    # affichage de données toute les 10 itérations
    if nbit % n_print == 0:
        print(nbit,
        " W :", str(W[:, 0]), "|| norm :", str(np.linalg.norm(W)), "|| dw :", str(dw[:, 0]), "|| B :", str(B[:, 0]),
        "|| Q :", str(sim.Q.axis()[:, 0]), "|| M :", str(np.linalg.norm(M)))

    # Actualisation de l'affichage graphique
    if plot_3D:
        b_vector.axis = 1e6 * vp.vector(B[0][0], B[1][0], B[2][0])
        satellite.rotate(angle=np.linalg.norm(W) * dt, axis=vp.vector(W[0][0], W[1][0], W[2][0]),
                        origin=vp.vector(10, 10, 10))

        axe_W.axis = 5*vp.vector(vp.vector(*(W/ np.linalg.norm(W))))

    # Rate : réalise 25 fois la boucle par seconde
    # vp.rate(fAffichage)  # vp.rate(1/dt)
    nbit += 1
    t += dt
    output['t'].append(t)
    output['M'].append(M)
    output['U'].append(U)
    output['Bv'].append(B_v)
    output['dB'].append(np.linalg.norm(dB)/np.linalg.norm(B_v))

    output['B_norm'].append(np.linalg.norm(B_v))
    output['W'].append(W)
    output['W_norm'].append(np.linalg.norm(W))
    output['M_norm'].append(np.linalg.norm(M))
    output['angle'].append( np.arccos(np.clip(np.dot(W.T, B_v)/ ( np.linalg.norm(B_v) * np.linalg.norm(W) ), -1.0, 1.0)) )



    if plot_2D and liveplot2D and np.mod(nbit+1, 10) == 0 :
        linew.set_data(output["t"], output["W_norm"])
        linewx.set_data(output["t"], np.array(output["W"])[:, 0])
        linewy.set_data(output["t"], np.array(output["W"])[:, 1])
        linewz.set_data(output["t"], np.array(output["W"])[:, 2])
        line2.set_data(output["t"], output["M_norm"])

        linebx.set_data(output["t"], np.array(output["Bv"])[:, 0])
        lineby.set_data(output["t"], np.array(output["Bv"])[:, 1])
        linebz.set_data(output["t"], np.array(output["Bv"])[:, 2])

        lineb.set_data(output["t"], output["B_norm"])

        lineAngle.set_data(output["t"], output["angle"])

        linedB.set_data(output["t"], output["dB"])



        for ax in axarr.flatten():
            ax.relim()
            ax.autoscale_view(True, True, True)

        ax4.set_ylim(0, 2*np.pi)

        plt.suptitle("Nt = {:1.1e}, ".format(nbit) +
                     "t = {:2.2e} $[s]$".format(nbit * dt), fontsize=12)

        plt.draw()
        plt.pause(0.01)  # Note this correction



if plot_2D and not liveplot2D :
    """The liveplot is off, but we want to see the 2D at the end"""
    linew.set_data(output["t"], output["W_norm"])
    linewx.set_data(output["t"], np.array(output["W"])[:, 0])
    linewy.set_data(output["t"], np.array(output["W"])[:, 1])
    linewz.set_data(output["t"], np.array(output["W"])[:, 2])
    line2.set_data(output["t"], output["M_norm"])

    linebx.set_data(output["t"], np.array(output["Bv"])[:, 0])
    lineby.set_data(output["t"], np.array(output["Bv"])[:, 1])
    linebz.set_data(output["t"], np.array(output["Bv"])[:, 2])

    lineb.set_data(output["t"], output["B_norm"])

    lineAngle.set_data(output["t"], output["angle"])

    linedB.set_data(output["t"], output["dB"])



    for ax in axarr.flatten():
        ax.relim()
        ax.autoscale_view(True, True, True)

    ax4.set_ylim(0, 2*np.pi)

    plt.suptitle("Nt = {:1.1e}, ".format(nbit) +
                 "t = {:2.2e} $[s]$".format(nbit * dt), fontsize=12)

    plt.draw()
    plt.pause(0.01)  # Note this correction
plt.show()

while True:
    pass