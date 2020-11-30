import numpy as np
from math import sqrt
from scao.quaternion import Quaternion

class UKF:

    def __init__(self,dim,q0,W0,P0,Qcov,dt):
        self.dim = dim  #Dimension of state
        self.x = [q0,W0]  #Current state
        self.P = P0  #Covariance matrix on the state
        self.Qcov = Qcov  #Process noise
        self.sigPoints = np.zeros(2*dim)  #List of sigma points (Xi)
        self.dt = dt  #time step

    def addition(x,L):  #x is a state(quaternion + rotation) and L is an array(two 3-dim vectors = 6-dim vector)
        alpha = np.linalg.norm(L[:,0:3])
        direction = L[:,0:3]/alpha

        return [x[0]*Quaternion(np.cos(alpha/2),direction[0,0]*np.sin(alpha/2),direction[1,0]*np.sin(alpha/2),direction[2,0]*np.sin(alpha/2)),x[1] + L[:,3:6]]

    def sigmaPoints(self):
        sqrtmatrix = np.linalg.sqrtm(self.P + self.Qcov)

        for i in range(self.dim):
            self.sigPoints[i] = addition(self.x, sqrt(2*self.dim) * sqrtmatrix[:,i])
        for i in range(self.dim):
            self.sigPoints[i+self.dim] = addition(self.x, -sqrt(2*self.dim) * sqrtmatrix[:,i])

    def stateMean(self,Yi):

        mean = []
        mean.append(Quaternion.mean(Yi[:][0],1e-4))
        rotMean = np.zeros((3,1))
        for i in range(2*self.dim):
            rotMean += Yi[i][1]
        rotMean /= (2*self.dim)
        mean.append(rotMean)

        return mean

    def evolv(self,x):  #x a state vector
        ajout = np.zeros((6,1))
        ajout[:,0:3] = x[1]*self.dt
        return addition(x,ajout)

    vecEvolv = np.vectorize(evolv)

    def errorCorrection(self, w, B):
        '''
        Renvoie au pas de temps de l'appel la correction de la mesure
        '''
        self.sigmaPoints() # Caclul des Wi, calcul des Xi et sauvegarde dans self.sigPoints
        Yi = self.vecEvolv(self.sigPoints) # process model, le bruit étant intégré dans les sigmaPoints
        xk_ = self.stateMean(Yi)
        WiPrime = WiCalculus(Yi, xk_)
        Pk_ = aPrioriProcessCov(WiPrime)
        return
