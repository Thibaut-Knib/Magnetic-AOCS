# coding=utf-8
import sys
import os

# os.chdir(r"C:\Users\thiba\Documents\Polytechnique\P3A\Magnetic-AOCS\tst\sim")
sys.path.append(os.path.join(*['..'] * 2))
print(os.getcwd())
sys.path.append("../src/")
sys.path.append("src/")

import shutil
from tst.sim.simulator import Simulator
from scao.scao import SCAO, Bdot
from environnement.environment import Environment
from environnement.orbit import Orbit
from hardware.hardwares import Hardware
import matplotlib.pyplot as plt
from visualization import LiveVisu3D, LiveVisu2D

try:
    """using custom style for nice plots"""
    plt.style.use("presentation")
except:
    pass

plt.ion()

from matplotlib.ticker import FuncFormatter


def MyFormatter(x, lim):
    if x == 0:
        return 0
    return '{0:.1f}e{1:.0f}'.format(np.sign(x) * 10 ** (-np.floor(np.log10(abs(x))) + np.log10(abs(x))),
                                    np.floor(np.log10(abs(x))))
    # The first argument of the format gives the first significant digits of the number with the sign preserved and
    # brought to a range between [1-10), The next argument gives the  numbers integer exponent of 10
    # Both the first and second arguments are formatted to display only 1 decimal places due to the lack of space.


def sci_format(x, lim):
    return '{:.1e}'.format(x)


majorFormatter = FuncFormatter(MyFormatter)

###############################
# Paramètres de la simulation #
###############################
if (not os.path.isfile('conf_bdot.py')):
    shutil.copy2('tst/sim/conf.default.py', 'conf_bdot.py')
from conf_bdot import *


def running_bdot_convergence(k_Bdot=1e3,
                             plot_3D=False,
                             plot_2D=True,
                             liveplot2D=True,
                             nt_liveplot2D=200,
                             live_print=True,
                             n_print=300,
                             W0=[[0], [0], [0]],
                             tmax=60e3):
    """run the Bdot stabilisation and return the temporal values"""

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

    # Environnement

    orbite = Orbit(omega=0, i=pi / 6, e=0.01, r_p=7e6, mu=3.986004418e14, tau=0)
    environnement = Environment(B_model)

    # Initialisation du champ magnétique:
    orbite.setTime(t)
    environnement.setPosition(orbite.getPosition())
    B = environnement.getEnvironment()  # dans le référentiel du satellite

    # Simulateur
    sim = Simulator(dt, L0)
    # Algortihmes de stabilisation
    bdot_algorithm = Bdot(k_Bdot=k_Bdot, dt=dt)

    ############################
    # Initialisation graphique #
    ############################
    if plot_3D:
        live3Dvisualisation = LiveVisu3D(lx, ly, lz, B)
    if plot_2D:
        live2Dvisualisation = LiveVisu2D()

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
    output = {'t': [], 'M': [], 'U': [], "W": [], "W_norm": [], "M_norm": [],
              "Bv": [], "B_norm": [], "angle": [], "dB": []}
    B_old = 0

    Mr = []

    while t < tmax:
        # on récupère la valeur actuelle du champ magnétique et on actualise l'affichage du champ B
        orbite.setTime(t)  # orbite.setTime(t)
        environnement.setPosition(orbite.getPosition())
        B = environnement.getEnvironment()  # dans le référentiel géocentrique
        # B *= 10  # Speedup the calculation

        # on récupère le prochain vecteur rotation (on fait ube étape dans la sim)
        W = sim.getNextIteration(M, dw, J, B, I)

        # Enregistrement de variables pour affichage
        Wr.append(np.linalg.norm(W))
        qs.append(sim.Q)

        # B-dot law
        B_v = sim.Q.R2V(B)  # B_v in Rv
        M = bdot_algorithm.mag_moment(B_measured=B_v)
        dw = np.zeros_like(M)

        Mr.append(np.linalg.norm(M))
        U, M = hardW.getRealCommand(dw, M)

        # affichage de données toute les 10 itérations
        if nbit % n_print == 0 and live_print:
            print(nbit,
                  " W :", str(W[:, 0]), "|| norm :", str(np.linalg.norm(W)), "|| dw :", str(dw[:, 0]), "|| B :",
                  str(B[:, 0]),
                  "|| Q :", str(sim.Q.axis()[:, 0]), "|| M :", str(np.linalg.norm(M)))

        # Actualisation de l'affichage graphique
        if plot_3D:
            live3Dvisualisation.update_visu(B, W, dt)

            # Rate : réalise 25 fois la boucle par seconde
            # vp.rate(fAffichage)  # vp.rate(1/dt)

        nbit += 1
        t += dt
        output['t'].append(t)
        output['M'].append(M)
        output['U'].append(U)
        output['Bv'].append(B_v)
        output['dB'].append(np.linalg.norm(bdot_algorithm.dB_v) / np.linalg.norm(B_v))

        output['B_norm'].append(np.linalg.norm(B_v))
        output['W'].append(W)
        output['W_norm'].append(np.linalg.norm(W))
        output['M_norm'].append(np.linalg.norm(M))
        output['angle'].append(
            np.arccos(np.clip(np.dot(W.T, B_v) / (np.linalg.norm(B_v) * np.linalg.norm(W)), -1.0, 1.0)))

        if plot_2D and liveplot2D and np.mod(nbit + 1, nt_liveplot2D) == 0:
            live2Dvisualisation.update_visu(output)

    if plot_2D and not liveplot2D:
        """The liveplot is off, but we want to see the 2D at the end"""
        live2Dvisualisation.update_visu(output)

    plt.show()

    return output


velocity_min = 0.01

kBdot_list = np.linspace(1e3, 2e5, 50)

W0_list = [[[0.5], [0], [0]],
           [[0], [0.5], [0]],
           [[0.289], [0.289], [0.289]],
           [[0.354], [0.354], [0]],
           [[0], [0.354], [0.354]],
           [[0.354], [0], [0.354]],
           [[0], [0], [0.5]]]

shape_list = [[0.1, 0.1, 0.1],
              [0.2, 0.1, 0.1],
              [0.1, 0.2, 0.1],
              [0.1, 0.1, 0.2]]

# shape_list = [ [0.1, 0.1, 0.1] ]
# kBdot_list = [1e4, 1e5]
# W0_list = [ [[0], [0.5], [0]],
#             [[0], [0], [0.5]]]
for lx, ly, lz in shape_list:
    print(lx)
    plt.figure(figsize=(7, 4))
    plt.show()
    plt.pause(1)

    convergence_time = np.empty_like(kBdot_list)

    for W0 in W0_list:
        """iterrate the convergence with different velocity vectors"""

        for i, k_Bdot in enumerate(kBdot_list):
            """Iterrate over the range of K_Bdot values that we want"""

            print(i)
            output = running_bdot_convergence(k_Bdot=k_Bdot, plot_2D=False, plot_3D=False, live_print=False, W0=W0,
                                              tmax=2e4)
            Wn = np.array(output["W_norm"])
            time = np.array(output["t"])

            index = np.argmax(Wn < velocity_min)
            if index == 0:
                index = len(Wn) - 1

            convergence_time[i] = time[index]

            # if time[index] < 5000:
            #     live2Dvisualisation = LiveVisu2D()
            #     live2Dvisualisation.update_visu(output)
            #     live2Dvisualisation.fig.savefig(f"Convergence_simulation_W0={W0}_lx={lx},ly={ly},lz={lz}_kbdot={k_Bdot}.png")
            #     plt.close(live2Dvisualisation.fig)

        plt.plot(kBdot_list, convergence_time, label=f"W$_0$ = {W0[0][0]} $e_x$ + {W0[1][0]} $e_y$ + {W0[2][0]} $e_z$")

    plt.xlabel("KBdot value [SI]")
    plt.ylabel("Convergence time [s]")
    plt.legend()

    ax = plt.gca()
    # ax.ticklabel_format(axis="both", style="sci", scilimits=(0,0), useOffset=False, useMathText=True)
    ax.get_xaxis().set_major_formatter(majorFormatter)
    # ax.get_xaxis().set_major_formatter(plt.LogFormatter(10,  labelOnlyBase=False))

    plt.savefig(f"Bdot_convergence_lx={lx},ly={ly},lz={lz}.png")
