import numpy as np
from math import sqrt
from scao.quaternion import Quaternion
from errorHandler.state import State
from copy import copy

class UKF:

    def __init__(self,q0,W0,gyroBias,P0,Qcov,Rcov,dt):
        self.dim = 9  #Dimension of state
        self.curState = State(q0,W0,gyroBias)  #Current state
        self.P = P0  #Covariance matrix on the state
        self.Qcov = Qcov  #Process noise
        self.Rcov = Rcov #covariance du modèle d'erreur de la mesure
        self.dt = dt  #Time step
        self.record = {'stateIn':[],
                        'xk_': [],
                        'Pk_': [],
                        'nu': [],
                        'K': [],
                        'stateOut': [],
                        'PCorr': []}


    def sigmaPoints(self):  #List of sigma points used to calculate the new mean and standard deviation
        sqrtmatrix = np.linalg.cholesky(self.P + self.Qcov)
        #print(sqrtmatrix)

        res = []
        for i in range(self.dim):
            ajout = np.atleast_2d(sqrtmatrix[:,i]).T  #sqrt(2*self.dim) *
            res.append(self.curState.addition(ajout))
            res.append(self.curState.addition(-ajout))
        return res

    def evolv(self, Xi):  #Xi the list of state vectors
        Yi = copy(Xi)
        for yi in Yi:
            yi.evolv(self.dt)

        return Yi

    def stateMean(self, Yi):  #Return the mean of the list Yi (yMean gave the initialisation for the quaternion mean)
        #Nécessaire pour initialiser la moyenne des quaternions
        yMean = copy(self.curState)
        yMean.evolv(self.dt)

        return State.stateMean(yMean.Q,Yi)

    def WiCalculus(self, Yi, xk_):
        WiPrime = []
        for yi in Yi:
            q = yi.Q*xk_.Q.inv()
            vec = q.axis()*q.angle()
            wi = np.zeros((self.dim,1))  #Reduced state
            wi[0:3] = vec
            wi[3:6] = yi.W - xk_.W
            wi[6:9] = yi.gyroBias - xk_.gyroBias
            WiPrime.append(wi)
        return WiPrime

    def aPrioriProcessCov(self, WiPrime):
        Pk_ = np.zeros((self.dim,self.dim))
        for wi in WiPrime:
            Pk_ += np.dot(wi,wi.T)
        Pk_ /= (2*self.dim)
        return Pk_

    def predictObs(self, Yi, B):
        Zi = []
        ZRot = predictRotation(Yi)
        ZMagnet = predictMagnetField(Yi,B)
        for rot,mag in zip(ZRot,ZMagnet):
            zi = np.zeros((6,1))
            zi[0:3] = rot
            zi[3:6] = mag
            Zi.append(zi)
        return Zi

    def innovation(self, zk_, WM, BM):
        Zmesur = np.zeros((6,1))
        Zmesur[0:3] = WM
        Zmesur[3:6] = BM
        return Zmesur - zk_

    def ObsCov(self, Zi, zk_):
        cov = np.zeros((6,6))
        for z in Zi:
            cov += np.dot(z-zk_,(z-zk_).T)
        cov /= (2*self.dim)
        return cov

    def crossCorrelationMatrix(self, WiPrime, Zi, zk_):
        crosscov = np.zeros((self.dim,6))
        for w,z in zip(WiPrime,Zi):
            crosscov += np.dot(w,(z-zk_).T)
        crosscov /= (2*self.dim)
        return crosscov

    def errorCorrection(self, WM, BM, B):
        '''
        Renvoie au pas de temps de l'appel la correction de la mesure
        '''
        # prediction of state
        Xi = self.sigmaPoints() # Caclul des Wi, calcul des Xi et sauvegarde dans self.sigPoints
        #print([xi.Q for xi in Xi])
        Yi = self.evolv(Xi) # process model, le bruit étant intégré dans les sigmaPoints
        #print([yi.Q for yi in Yi])
        xk_ = self.stateMean(Yi)
        #print(xk_.Q.axis()*xk_.Q.angle())
        WiPrime = self.WiCalculus(Yi, xk_)
        Pk_ = self.aPrioriProcessCov(WiPrime)
        # prediction of measure
        Zi = self.predictObs(Yi, B)
        zk_ = obsMean(Zi)
        nu = self.innovation(zk_, WM, BM)
        Pzz = self.ObsCov(Zi, zk_)
        Pnunu = self.Rcov + Pzz
        #print(Pnunu)
        Pxz = self.crossCorrelationMatrix(WiPrime, Zi, zk_)
        K = kalmanGain(Pxz, Pnunu)
        #print(K)
        #K[0:3,3:6] = np.zeros((3,3))
        K[6:9,0:6] = np.zeros((3,6))
        #K = np.zeros((9,6))
        xCorr = xk_.addition(np.dot(K,nu))
        #print(xCorr.Q)
        PCorr = Pk_ - np.dot(K, np.dot(Pnunu, K.T))

        #record
        self.record['stateIn'].append(self.curState)
        self.record['xk_'].append(xk_)
        self.record['Pk_'].append(Pk_)
        self.record['nu'].append(nu)
        self.record['K'].append(K)
        self.record['stateOut'].append(xCorr)
        self.record['PCorr'].append(PCorr)

        # Update
        self.curState = xCorr
        self.P = PCorr

def predictRotation(Yi):
    ZRot = [state.W + state.gyroBias for state in Yi]
    return ZRot

def predictMagnetField(Yi,B):
    ZMagnet = []
    for yi in Yi:
        q = yi.Q
        ZMagnet.append(q.R2V(B))
    return ZMagnet

def obsMean(Zi):
    mean = sum(Zi)/len(Zi)
    return mean

def kalmanGain(Pxz, Pnunu):
    return np.dot(Pxz, np.linalg.inv(Pnunu))
