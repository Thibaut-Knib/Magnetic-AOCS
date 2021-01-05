# coding=utf-8
import os
import sys

# os.chdir(r"C:\Users\thiba\Documents\Polytechnique\P3A\Magnetic-AOCS\tst\sim")
sys.path.append(os.path.join(*['..'] * 2))
print(os.getcwd())
sys.path.append("../../src/")
sys.path.append("src/")

import shutil
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

from tqdm import tqdm
import seaborn as sns

import pickle as pck

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

def runStatisticalInvestigation(numbreOfOrbites=1, Nstat=50, doSCA=False, useUKF=True):
    alpha_values = []
    beta_values = []
    epsGyro_values = []

    for kkk in tqdm(range(Nstat), total=Nstat):

        epsilon_gyro = 10 ** -(random() * 4 + 3)  # entre 10^-2 et 10^-6
        epsGyro_values.append(epsilon_gyro)

        standard_deviation_gyro = np.array([[epsilon_gyro], [epsilon_gyro], [epsilon_gyro]])
        gyroModel = (biais_gyro, standard_deviation_gyro, scaling_factor_gyro, drift_gyro)

        W0 = 0 * np.array([[2 * (random() - 0.5)] for i in range(3)])  # rotation initiale dans le référentiel R_r

        t = 0
        dw = np.array([[0.], [0.], [0.]])  # vecteur de l'accélération angulaire des RW
        M = np.array([[0.], [0.], [0.]])  # vecteur du moment magnétique des bobines
        I = np.diag((m * (ly ** 2 + lz ** 2) / 3, m * (lx ** 2 + lz ** 2) / 3,
                     m * (lx ** 2 + ly ** 2) / 3))  # Tenseur inertie du satellite
        L0 = np.dot(I, W0)

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
        LocalMagField_Rr = environnement.getEnvironment()  # dans le référentiel du satellite

        # Simulateur
        Q0 = Quaternion(1, 0, 0, 0)
        sim = Simulator(dt, L0, Q0)
        Qt = Quaternion(1, 0, 0, 0)  # Quaternion objectif

        # Monde mesuré
        mWorld = MeasuredWorld(gyroModel, magneticModel, dt, Q0)

        # Algortihmes de stabilisation
        stab = SCAO(PIDRW(RW_P, RW_dP, RW_D), PIDMT(MT_P, MT_dP, MT_D), SCAOratio, I, J)  # stabilisateur

        ############################
        # Initialisation du filtre #
        ############################

        dimState = 9  # 3 pour les quaternions, 3 pour la rotation, 3 pour les biais
        dimObs = 6

        Rcov = np.eye(dimObs)  # ATTENTION, dimension de la mesure ici, pas de l'état

        Rcov[0:3, 0:3] *= epsilon_gyro ** 2
        Rcov[3:6, 3:6] *= epsilon_mag ** 2

        CovRot = 1e-2  # Paramètre à régler pour le bon fonctionnement du filtre

        P0 = np.eye(dimState) * 1e-3
        Qcov = np.zeros((dimState, dimState))
        Qcov[0:3, 0:3] = 1e-12 * np.eye(3)  # uncertainty on the evolution of the attitude
        Qcov[3:6, 3:6] = CovRot * np.eye(3)  # uncertainty on the evolution of the velocity
        Qcov[6:9, 6:9] = 1e-12 * np.eye(3)  # uncertainty on evolution of the biais

        ukf = flt.UKF(Q0, W0, gyroModel[0], P0, Qcov, Rcov, dt)

        #####################
        # Boucle principale #
        #####################
        output = {'t': [], 'M': [], 'U': []}
        outputB = {'t': [], 'B': [], 'BM': []}
        outputUKF = {'Qs': [], 'Qm': [], "Qc": [], 'Ws': [], "Wm": [], "Wc": [], "P": []}

        timesteps = np.arange(t, numbreOfOrbites * orbite.getPeriod(), dt)

        # for t in tqdm(timesteps):
        for t in timesteps:
            # on récupère la valeur actuelle du champ magnétique et on actualise l'affichage du champ B
            orbite.setTime(t)  # orbite.setTime(t)
            environnement.setPosition(orbite.getPosition())
            LocalMagField_Rr = environnement.getEnvironment()  # dans le référentiel géocentrique

            # on récupère le prochain vecteur rotation (on fait ube étape dans la sim)
            W = sim.getNextIteration(M, dw, J, LocalMagField_Rr, I)

            # Update monde mesuré
            mWorld.getNextIteration(W, LocalMagField_Rr, sim.Q)

            if useUKF:
                # Correction des erreurs avec un filtre
                ukf.errorCorrection(mWorld.WM, mWorld.BM, LocalMagField_Rr)

                # Sauvegarder les valeurs de simulation actuelles: (valeurs mesurées)
                stab.setAttitude(ukf.curState.Q)
                stab.setRotation(ukf.curState.W)
            else:
                # Sauvegarder les valeurs de simulation actuelles: (valeurs mesurées)
                stab.setAttitude(mWorld.QM)
                stab.setRotation(mWorld.WM)

            stab.setMagneticField(sim.Q.V2R(mWorld.BM))  # non recalé

            # Prise de la commande de stabilisation
            if doSCA:
                dw, M = stab.getCommand(Qt)  # dans Rv
                U, M = hardW.getRealCommand(dw, M)

            # output['t'].append(t)
            # output['M'].append(M)
            # output['U'].append(U)

            outputUKF['Ws'].append(W)
            # outputUKF['Wm'].append(mWorld.WM)
            # outputUKF['Wc'].append(ukf.curState.W)
            outputUKF['Qs'].append(sim.Q)
            # outputUKF['Qm'].append(mWorld.QM)
            # outputUKF['Qc'].append(ukf.curState.Q)
            # outputUKF['P'].append(ukf.P)

            # outputB['t'].append(t)
            # outputB['B'].append(sim.Q.R2V(LocalMagField_Rr))
            # outputB['BM'].append(mWorld.BM)

        index = len(timesteps) // 10  # take the last 10%
        angles = [abs(quat.angle()) for quat in outputUKF['Qs'][-index:]]
        velocities = [np.linalg.norm(w) for w in outputUKF['Ws'][-index:]]

        alpha_max = max(angles)
        beta_max = max(velocities)
        alpha_values.append(alpha_max)
        beta_values.append(beta_max)

    return alpha_values, beta_values, epsGyro_values


def saveRun(alpha_values, beta_values, epsGyro_values, filename="unitile.data"):
    with open(filename, "w") as f:
        dictdata = {"angle": alpha_values, "velocity": beta_values, "epsGyro": epsGyro_values}
        pck.dump(dictdata, f)


def plotRun(alpha_values, beta_values, epsGyro_values, filename=None, fig=None, axarr=None, color="black"):
    ###################
    # plot
    ##################
    sns.set(context='paper', style='white')
    # sns.set_theme(style="ticks")
    plt.rcParams["figure.figsize"] = (8, 4)

    if axarr is None:
        fig, [ax1, ax2] = plt.subplots(2, 1)
    else:
        ax1, ax2 = axarr

    ax1.scatter([-log10(e) for e in epsGyro_values], alpha_values, marker='o', color=color)
    ax2.scatter([-log10(e) for e in epsGyro_values], beta_values, marker='o', color=color)

    ax2.set_xlabel('$-\log_{10}(\sigma_{gyromètre}$)')
    ax1.set_ylabel("Erreur angulaire ")
    ax2.set_ylabel("Erreur vitesse")

    ax1.set_title("Stabilité du satellite en fonction de l'écart-type de l'erreur sur les gyromètres")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    plt.show()


alpha_valuesNoUKF, beta_valuesNoUKF, epsGyro_valuesNoUKF = runStatisticalInvestigation(5, Nstat=50, doSCA=True,
                                                                                       useUKF=False)

suffix = "_qw10-2"
alpha_values, beta_values, epsGyro_values = runStatisticalInvestigation(5, Nstat=50, doSCA=True, useUKF=True)
plotRun(alpha_values, beta_values, epsGyro_values, "rests_UKF_yesUKF" + suffix + ".png")
plotRun(alpha_valuesNoUKF, beta_valuesNoUKF, epsGyro_valuesNoUKF, "rests_UKF_noUKF_" + suffix + ".png")

fig, [ax1, ax2] = plt.subplots(2, 1)
plotRun(alpha_valuesNoUKF, beta_valuesNoUKF, epsGyro_valuesNoUKF, None, fig, [ax1, ax2], color="r")
plotRun(alpha_values, beta_values, epsGyro_values, "rests_UKF_bothUKF" + suffix + ".png", fig, [ax1, ax2], color="b")
ax1.legend(["without UKF", "With UKF"])
plt.savefig("rests_UKF_bothUKF" + suffix + ".png")

ax1.set_yscale("log")
ax2.set_yscale("log")

plt.savefig("rests_UKF_bothUKF" + suffix + "_log.png")
