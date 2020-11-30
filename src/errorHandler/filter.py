import numpy as np
from math import sqrt
from scao.quaternion import Quaternion

class UKF:

    def __init__(self,dim,q0,W0,P0,Qcov,dt):
        self.dim = dim  #Dimension of state
        self.x = [q0,W0]  #Current state
        self.P = P0  #Covariance matrix on the state
        self.Qcov = Qcov  #Process noise
        self.dt = dt  #time step


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
        print(Yi[:][0])
        mean.append(Quaternion.mean(Yi[:][0],1e-4))
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

    def errorCorrection(self, WM, BM):
        '''
        Renvoie au pas de temps de l'appel la correction de la mesure
        '''
        Xi = self.sigmaPoints() # Caclul des Wi, calcul des Xi et sauvegarde dans self.sigPoints
        #print("\n")
        #print(Xi)
        #print("\n")
        Yi = self.evolv(Xi) # process model, le bruit étant intégré dans les sigmaPoints
        xk_ = self.stateMean(Yi)
        WiPrime = WiCalculus(Yi, xk_)
        Pk_ = aPrioriProcessCov(WiPrime)
        return

def addition(x,L):  #x is a state(quaternion + rotation) and L is an array(two 3-dim vectors = 6-dim vector)
    alpha = np.linalg.norm(L[0:3])
    if alpha < 1e-4:
        direction = np.array([[1.0],[0.0],[0.0]])
    else:
        direction = L[0:3]/alpha


    return [x[0]*Quaternion(np.cos(alpha/2),direction[0,0]*np.sin(alpha/2),direction[1,0]*np.sin(alpha/2),direction[2,0]*np.sin(alpha/2)),x[1] + L[3:6]]
