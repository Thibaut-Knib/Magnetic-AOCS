import numpy as np
from src.scao.quaternion import Quaternion
from math import acos, asin

class SCAO:

    """This is some scao docstrings"""

    def __init__(self, stabRW, stabMT, rwRatio, I, J):
        """
        * listQ: liste des attitudes dans Rr
        * listL: liste des moments cinétiques dans Rr
        * listB: liste des champs magnétiques calculées dans Rv
        * Mrv : matrice de transfert de r -> v
        * Mvr : inverse de Mvr
        * P_d_mt: facteurs P du PID pour les magnetorquers pour le detumbling (matrice 3x3 diagonale)
        * P_d_rw: facteurs P du PID pour les roues à inertie pour le détumbling (matrice 3x3 diagonale)
        * P_s: facteur P du PID pour le controle d'attitude
        * D_s: facteur D du PID pour le controle d'attitude
        """
        self.Q = None  #: Quaternier d'Attitude exprimé dans Rr
        self.W = None  #: Rotation, exprimé dans Rr
        self.B = None  #: Champ magnetic, exprimé dans Rr
        self.I = I  #: Moment d'inertie, exprimé dans Rv
        self.rwRatio = rwRatio  #: ratio de reaction wheel (Quesaqo ?)
        self.stabRW = stabRW  #: retourne un vecteur dans Rv
        self.stabMT = stabMT  #: retourne un vecteur dans Rv

    def setAttitude(self, Q):
        """ Ajoute le quaternion actuel à la liste des attitudes."""
        self.Q = Q

    def setRotation(self, W):
        """ Ajoute le vecteur rotation actuel à la liste des rotations."""
        self.W = W

    def setMagneticField(self, B):
        """ Calcule B dans Rr et l'ajoute."""
        self.B = B

    def getCommand(self, Qt):
        """

        :param Qt: Un quaternion (?)
        :type Qt: Quaternion
        :return:  (dw, torqueM)
            |  dw : la variation de la vitesse de rotation de la roue
            |  torqueM : le moment magnetique
        """
        torqueM = (1-self.rwRatio)*self.stabMT(self.Q,self.W,Qt,self.B,self.I)

        dw = -self.rwRatio*self.stabRW(self.Q,self.W,Qt,self.B,self.I)

        return dw, torqueM  #:dans Rv



class Bdot:
    """A class for the Bdot Algorithm"""

    def __init__(self, k_Bdot = None, dt = None):
        """Init the class with the main parameters
        :param k_Bdot: the proportional parameter of the Bdot Algorithm
        :param dt: timestep of the simulation
        """

        self.k_bdot = k_Bdot
        self.dt = dt
        self.B_measured_kmoins1 = None  # Meqsured magnetic field at the previous time step.
        self.dB_v = None  # The temporal derivative of the magnetic field measured in the Vehicle reference frame

    def mag_moment(self, B_measured):
        """compute the magnetic moment with the Bdot algorithm
        :param B_measured:  of the magnetic fied measured in the Vehicule Reference frame
         :type B_measured: np.array of dimention 3
        :return: M
          M: Magnetic moment to generate in the Vehicule reference frame
        """
        if self.B_measured_kmoins1 is None:
            self.B_measured_kmoins1 = B_measured

        self.dB_v = B_measured - self.B_measured_kmoins1
        self.dB_v /= self.dt

        M = - self.k_bdot * self.dB_v

        self.B_measured_kmoins1 = B_measured

        return M