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

    def stateMean(self,Yi):

        mean = []
        LQuat = [x[0] for x in Yi]
        mean.append(Quaternion.mean(LQuat,1e-2))
        rotMean = np.zeros((3,1))
        for i in range(2*self.dim):
            rotMean += Yi[i][1]
        rotMean /= (2*self.dim)
        mean.append(rotMean)

        return mean

    def evolv(self,Xi):  #list of state vectors
        Yi = []
        for i in range(len(Xi)):
            ajout = np.zeros((6,1))
            ajout[0:3] = Xi[i][1]*self.dt
            Yi.append(addition(Xi[i],ajout))
        return Yi

    def errorCorrection(self, WM, BM, B):
        '''
        Renvoie au pas de temps de l'appel la correction de la mesure
        '''
        # prediction of state
        Xi = self.sigmaPoints() # Caclul des Wi, calcul des Xi et sauvegarde dans self.sigPoints
        Yi = self.evolv(Xi) # process model, le bruit étant intégré dans les sigmaPoints
        xk_ = self.stateMean(Yi)
        WiPrime = WiCalculus(Yi, xk_)
        Pk_ = aPrioriProcessCov(WiPrime)
        # prediction of measure
        Zi = predictObs(Yi,B)
        zk_ = obsMean(Zi)
        nu = innovation(zk_, WM, BM)
        Pzz = ObsCov(Zi, zk_)
        Pnunu = self.Rcov + Pzz
        Pxz = crossCorrelationMatrix(WiPrime, Zi, zk_)
        K = kalmanGain(Pxz, Pnunu)
        xCorr = addition(xk_,Kk*nu)


        return

def addition(x,L):  #x is a state(quaternion + rotation) and L is an array(two 3-dim vectors = 6-dim vector)
    alpha = np.linalg.norm(L[0:3])
    if alpha < 1e-4:
        direction = np.array([[1.0],[0.0],[0.0]])
    else:
        direction = L[0:3]/alpha


    return [x[0]*Quaternion(np.cos(alpha/2),direction[0,0]*np.sin(alpha/2),direction[1,0]*np.sin(alpha/2),direction[2,0]*np.sin(alpha/2)),x[1] + L[3:6]]

def WiCalculus(Yi, xk_):
    WiPrime = []
    for i in range(len(Yi)):
        q = Yi[i][0]*xk_[0].inv()
        vec = q.axis()*q.angle()
        elmt = np.zeros((6,1))
        elmt[0:3] = vec
        elmt[3:6] = Yi[i][1] - xk_[1]
        WiPrime.append(elmt)
    return WiPrime

def aPrioriProcessCov(WiPrime):
    Pk_ = np.zeros((6,6))
    for i in range(len(WiPrime)):
        Pk_ += np.dot(WiPrime[i],WiPrime[i].T)
    Pk_ /= (2*6)
    return Pk_

def predictRotation(Yi):
    return [x[1] for x in Yi]

def predictMagnetField(Yi,B):
    Zi = []
    for i in range(len(Yi)):
        q = Yi[i][0]
        Zi.append(q.R2V(B))
    return Zi

def predictObs(Yi,B):
    Zi = []
    Zi1 = predictRotation(Yi)
    Zi2 = predictMagnetField(Yi,B)
    for i in range(len(Yi)):
        Z = np.zeros((6,1))
        Z[0:3] = Zi1[i]
        Z[3:6] = Zi2[i]
        Zi.append(Z)
    return Zi

def obsMean(Zi):
    return sum(Zi)/len(Zi)

def innovation(xk_,WM,BM):
    Zmesur = np.zeros((7,1))
    Zmesur[0:3] = WM
    Zmesur[3:6] = BM
    return Zmesur - xk_

def ObsCov(Zi,zk_):
    cov = np.zeros((len(Zi),len(Zi)))
    for z in Zi:
        cov += np.dot(z-zk_,(z-zk_).T)
    cov /= len(Zi)
    return cov


def kalmanGain(Pxz, Pnunu):
    return np.dot(Pxz, np.linalg.inv(Pnunu))

def crossCorrelationMatrix(WiPrime, Zi, zk_):
    crosscov = np.zeros((len(Zi),len(Zi)))
    for w,z in zip(WiPrime,Zi):
        crosscov += np.dot(w,(z-zk_).T)
    crosscov /= len(Zi)
    return crosscov
