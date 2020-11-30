import numpy as np
from math import sqrt
from scao.quaternion import Quaternion

class UKF:

    def __init__(self,dim,q0,W0,P0,Qcov,dt):
        self.dim = dim  #Dimension of state
        self.x = [q0,W0]  #Current state
        self.P = P0  #Covariance matrix on the state
        self.Qcov = Qcov  #Process noise
        self.sigPoints = None  #List of sigma points
        self.dt = dt  #time step

    def addition(x,L):  #x is a state(quaternion + rotation) and L is an array(two 3-dim vectors = 6-dim vector)
        alpha = np.linalg.norm(L[:,0:3])
        direction = L[:,0:3]/alpha

        return [x[0]*Quaternion(np.cos(alpha/2),direction[0,0]*np.sin(alpha/2),direction[0,1]*np.sin(alpha/2),direction[0,2]*np.sin(alpha/2)),x[1] + L[:,3:6]]

    def sigmaPoints(self):
        sqrtmatrix = np.linalg.sqrtm(self.P + self.Qcov)

        self.sigPoints = []
        for i in range(self.dim):
            self.sigPoints.append(addition(self.x, sqrt(2*self.dim) * sqrtmatrix[:,i]))
        for i in range(self.dim):
            self.sigPoints.append(addition(self.x, -sqrt(2*self.dim) * sqrtmatrix[:,i]))

    def prediction(self):
        self.sigmaPoints()

        return

    def quaternMean(self,tol):
        qt = Quaternion(1,0,0,0)
        return qt


    def evolv(self,x):  #x a state vector
        ajout = np.zeros((6,1))
        ajout[:,0:3] = x[1]*self.dt
        return addition(x,ajout)
