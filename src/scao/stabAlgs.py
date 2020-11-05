import numpy as np
from src.scao.quaternion import Quaternion
from math import acos, cos, pi, sin

def PIDRW(P, dP, D): ## BUG: CHECK HERE FOR REF CONSISTENCY
    def res(Q,W,Qt,B,I):
        #Qt = targeted attitude
        Qr = Qt*Q.inv() #quaternion relatif qui effectue la rotation depuis Q vers Qt
        dynamicalP = P/(1+np.linalg.norm(W))**dP #dynamical P-factor
        u = Q.R2V(dynamicalP*Qr.axialPart() - D*W) #REMPLACER Qr.angle()*Qr.axis() par Qr.axialPart()!!
        #moment à appliquer
        torque = np.dot(I,u)
        return torque
    return res

def PIDMT(P, dP, D): # PDMT
    def res(Q,W,Qt,B,I):
        Qr = Q*Qt.inv()
        dynamicalP = P # P/(1+np.linalg.norm(W))**dP
        u = Q.R2V((dynamicalP*Qr.axialPart()+D*W))
        Bb = Q.R2V(B)
        M = np.cross(u,Bb,axisa=0,axisb=0,axisc=0)
        return M
    return res

def PIDMTI(P, dP, D, CI):
    Qs = []
    def res(Q,W,Qt,B,I):
        Qs.append(Q.vec())
        subQs = Qs[-300:]
        Qint = sum(subQs)/len(subQs)
        Qint = Quaternion(*Qint[:,0])
        Qintr = Qint*Qt.inv()
        Qr = Q*Qt.inv()
        dynamicalP = P # P/(1+np.linalg.norm(W))**dP
        u = Q.R2V(dynamicalP*Qr.axialPart() + D*W + CI*Qintr.axialPart())
        Bb = Q.R2V(B)
        M = np.cross(u,Bb,axisa=0,axisb=0,axisc=0)
        return M
    return res
