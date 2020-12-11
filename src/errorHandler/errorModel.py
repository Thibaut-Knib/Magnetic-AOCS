from random import gauss
from scao.quaternion import Quaternion
import numpy as np

class MeasuredWorld:

    """
    gyroModel, magneticModel : (biais, standard_deviation, scaling_factor, drift)
    t : time form the beginning of the simulation
    dt : time step
    QM : Measured quaternion
    BM : Measured magnetic field
    WM : Measured rotation speed
    """

    def __init__(self,gyroModel,magneticModel,dt,Q0=Quaternion(1, 0, 0, 0)):
        self.gyroModel = gyroModel
        self.magneticModel = magneticModel
        self.dt = dt
        self.t = 0
        self.QM = Q0
        self.BM = None
        self.WM = None

    def dQ(self):
        """
        renvoie la dérivée du quaternion
        """
        qw, qx, qy, qz = self.QM[0], self.QM[1], self.QM[2], self.QM[3]
        expQ = np.array([[-qx, -qy, -qz],
                         [qw, qz, -qy],
                         [-qz, qw, qx],
                         [qy, -qx, qw]])
        return 1 / 2 * np.dot(expQ, self.WM)


    def getNextIteration(self,W,B,Qtrue):
        self.setBM(B,Qtrue)
        self.setWM(W,Qtrue)
        Qnump = self.QM.vec() + self.dQ() * self.dt  # calcul de la nouvelle orientation
        Qnump /= np.linalg.norm(Qnump)
        self.QM = Quaternion(*Qnump[:, 0])
        self.t += self.dt

    def setBM(self,B,Q):
        self.BM = measuredValue(Q.R2V(B), self.magneticModel, self.t)

    def setWM(self,W,Q):
        self.WM = Q.V2R(measuredValue(Q.R2V(W), self.gyroModel, self.t))

def measuredValue(trueValue, model, t):

    biais, standard_deviation, scaling_factor, drift = model
    measured = (trueValue + biais + np.array([gauss(0,standard_deviation[i,0]) for i in range(3)]).reshape(3,1) + drift*t) * scaling_factor

    return measured
