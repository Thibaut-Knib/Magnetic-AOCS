import sys
sys.path.append('..')
from math import *
import numpy as np
from random import *
import wmm2015 as wmm
from environnement.magneticdipole import idm

class Model:

    def __init__(self, model):
        """
        r in meters, i and u in radians.
        """
        self.r = None
        self.i = None
        self.u = None
        self.model = model
        self.functions = {'wmm' :  self.wmmMagneticField, 'dipole' : self.dipoleMagneticField}

    def setPosition(self, position):
        """
        Takes the tuple (r,i,u) as an argument.
        """
        self.r = position[0]
        self.i = position[1]
        self.u = position[2]

    def getMagneticField(self):
        try:
            return self.functions[self.model]()
        except KeyError:
            print("Unknown model. Please use a value in this list: " + str(list(self.function.keys())))

    def wmmMagneticField(self):
        """
        In Tesla, in the local orbital frame.
        """
        mag = wmm.wmm(self.i, self.u, self.r/1000 - 6371.0088, 2019)
        return 1e-9*np.array([[-mag.down.values[0][0]], [mag.north.values[0][0]], [mag.east.values[0][0]]])

    def dipoleMagneticField(self):
        """
        In Tesla, in the local orbital frame.
        """
        return np.dot(self.A_xs(), np.dot(self.A_sy(), idm(self.i, self.u, self.r)))

    def A_sx(self): #V_s = A_sx . V_x
        cu = cos(self.u)
        su = sin(self.u)
        return np.array([[cu, su, 0],[-su, cu, 0],[0, 0, 1]])

    def A_ys(self): #V_y = A_ys . V_s
        si = sin(self.i)
        ci = cos(self.i)
        return np.array([[1,0,0],[0,ci,si],[0,-si,ci]])

    def A_xs(self):
        return np.transpose(self.A_sx())

    def A_sy(self):
        return np.transpose(self.A_ys())
