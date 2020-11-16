import sys
sys.path.append('..')
# import vpython as vp
from math import *
import numpy as np
# from random import *
import matplotlib.pyplot as plt
from orbit import Orbit
from environment import Environment

#ux = vp.vector(1,0,0)
#uy = vp.vector(0,1,0)
#uz = vp.vector(0,0,1)
#
## trièdre (z,x,y)
#axe_x_r = vp.arrow(pos=vp.vector(0,0,0), axis=10*ux, shaftwidth=0.5, color=vp.vector(1,0,0))
#axe_y_r = vp.arrow(pos=vp.vector(0,0,0), axis=10*uy, shaftwidth=0.5, color=vp.vector(0,1,0))
#axe_z_r = vp.arrow(pos=vp.vector(0,0,0), axis=10*uz, shaftwidth=0.5, color=vp.vector(0,0,1))

t=0
dt=10

#premier vecteur de base
e_1 = np.array([[1],[0],[0]])

# génération des paramètres de la mission
orbite = Orbit(0, pi/6, 0.01, 7e6)
environnement = Environment('wmm')

#vecteur champ magnétique
#B = vp.arrow(pos = vp.vector(0,0,0), axis=7.7425*ux, shaftwidth=0.2, color=vp.vector(1,1,1))

values = {'t':[], 'r':[], 'i':[], 'u':[], 'B':[]}
while t<orbite.getPeriod():
    t+=dt
    orbite.setTime(t)
    environnement.setPosition(orbite.getPosition())
    B_ = environnement.getEnvironment()
    # B_ = np.dot(orbite.A_xs(), np.dot(orbite.A_sy(), B_))
    # B.axis = 100*vp.vector(B_[0][0],B_[1][0],B_[2][0])
    values['t'].append(t)
    values['r'].append(r)
    values['i'].append(i)
    values['u'].append(u)
    values['B'].append(B_)
    # vp.rate(1/dt)



#plt.clf()
#for i in range(3):
#    plt.plot(values['t'], np.array(values['B'])[:,i])
#plt.show()

#tmins = [1e3,1e3,1e3]
#tcur = np.array([0.0,0.0,0.0])
#Bi = values['B'][0][:,0]
#for B in values['B'][1:]:
#    tcur+=dt
#    i = B[:,0]
#    D = np.abs(i-Bi)
#    for x in range(3):
#        if D[x]>1e-7:
#            tmins[x] = min(tcur[x], tmins[x])
#            tcur[x] = 0
#            Bi[x] = i[x]
