import numpy as np
from math import sqrt
from scao.quaternion import Quaternion

class UKF:

    def __init__(self,dim,q0,W0,P0,Qcov,Rcov,dt):
        self.dim = dim  #Dimension of state
        self.x = [q0,W0]  #Current state
        self.P = P0  #Covariance matrix on the state
        self.Qcov = Qcov  #Process noise
        self.dt = dt  #time step
        self.Rcov = Rcov #covariance du modèle d'erreur de la mesure de


    def sigmaPoints(self):
        sqrtmatrix = np.linalg.cholesky(self.P + self.Qcov)

        res = []
        for i in range(self.dim):
            res.append( addition(self.x, sqrt(2*self.dim) * np.atleast_2d(sqrtmatrix[:,i]).T) )
        for i in range(self.dim):
            res.append( addition(self.x, -sqrt(2*self.dim) * np.atleast_2d(sqrtmatrix[:,i]).T) )
        return res

    def evolv(self, Xi):  #list of state vectors
        Yi = []
        for xi in Xi:
            ajout = np.zeros((self.dim,1))
            ajout[0:3] = xi[1]*self.dt
            Yi.append(addition(xi,ajout))

        #Nécessaire pour initialiser la moyenne des quaternions
        ajout = np.zeros((self.dim,1))
        ajout[0:3] = self.x[1]*self.dt
        yMean = addition(self.x,ajout)
        return yMean, Yi

    def stateMean(self, yMean, Yi):

        mean = []
        quatInit = yMean[0]
        LQuat = [x[0] for x in Yi]
        mean.append(Quaternion.mean(quatInit,LQuat,5e-2,100))
        rotMean = np.zeros((3,1))
        for i in range(2*self.dim):
            rotMean += Yi[i][1]
        rotMean /= (2*self.dim)
        mean.append(rotMean)

        return mean

    def WiCalculus(self, Yi, xk_):
        WiPrime = []
        for yi in Yi:
            q = yi[0]*xk_[0].inv()
            vec = q.axis()*q.angle()
            wi = np.zeros((self.dim,1))
            wi[0:3] = vec
            wi[3:6] = yi[1] - xk_[1]
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
            zi = np.zeros((self.dim,1))
            zi[0:3] = rot
            zi[3:6] = mag
            Zi.append(zi)
        return Zi

    def innovation(self, zk_, WM, BM):
        Zmesur = np.zeros((self.dim,1))
        Zmesur[0:3] = WM
        Zmesur[3:6] = BM
        return Zmesur - zk_

    def ObsCov(self, Zi, zk_):
        cov = np.zeros((self.dim,self.dim))
        for z in Zi:
            cov += np.dot(z-zk_,(z-zk_).T)
        cov /= (2*self.dim)
        return cov

    def crossCorrelationMatrix(self, WiPrime, Zi, zk_):
        crosscov = np.zeros((self.dim,self.dim))
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
        yMean, Yi = self.evolv(Xi) # process model, le bruit étant intégré dans les sigmaPoints
        xk_ = self.stateMean(yMean, Yi)
        WiPrime = self.WiCalculus(Yi, xk_)
        Pk_ = self.aPrioriProcessCov(WiPrime)
        # prediction of measure
        Zi = self.predictObs(Yi, B)
        zk_ = obsMean(Zi)
        nu = self.innovation(zk_, WM, BM)
        Pzz = self.ObsCov(Zi, zk_)
        Pnunu = self.Rcov + Pzz
        Pxz = self.crossCorrelationMatrix(WiPrime, Zi, zk_)
        K = kalmanGain(Pxz, Pnunu)
        xCorr = addition(xk_,np.dot(K,nu))
        PCorr = Pk_ - np.dot(K, np.dot(Pnunu, K.T))
        # Update
        self.x = xCorr
        self.P = PCorr

def addition(state,L):  #state (quaternion + rotation) and L is an array(two 3-dim vectors = 6-dim vector)
    alpha = np.linalg.norm(L[0:3])
    if alpha < 1e-4:  #alpha environ égal à 0
        direction = np.array([[1.0],[0.0],[0.0]])  #quaternion nul
    else:
        direction = L[0:3]/alpha
    QuatDelta = Quaternion(np.cos(alpha/2),direction[0,0]*np.sin(alpha/2),direction[1,0]*np.sin(alpha/2),direction[2,0]*np.sin(alpha/2))
    return [state[0] * QuatDelta, state[1] + L[3:6]]

def predictRotation(Yi):
    ZRot = [x[1] for x in Yi]
    return ZRot

def predictMagnetField(Yi,B):
    ZMagnet = []
    for yi in Yi:
        q = yi[0]
        ZMagnet.append(q.R2V(B))
    return ZMagnet

def obsMean(Zi):
    mean = sum(Zi)/len(Zi)
    return mean

def kalmanGain(Pxz, Pnunu):
    return np.dot(Pxz, np.linalg.inv(Pnunu))
