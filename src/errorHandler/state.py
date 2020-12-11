from scao.quaternion import Quaternion
import numpy as np

class State:

    def __init__(self,q0,W0,gyroBias):
        self.Q = q0  #Quaternion
        self.W = W0  #Rotation speed
        self.gyroBias = gyroBias

    def addition(self,reducedState):  #reducedState is a 9-dim column vector (3x3-dim column vectors)
        alpha = np.linalg.norm(reducedState[0:3])
        if alpha < 1e-4:  #alpha environ égal à 0
            direction = np.array([0.0,0.0,0.0])  #quaternion nul avec une direction quelconque
        else:
            direction = reducedState[0:3,0]/alpha
        C,S = np.cos(alpha/2),np.sin(alpha/2)
        QuatDelta = Quaternion(C,direction[0]*S,direction[1]*S,direction[2]*S)

        return State(self.Q * QuatDelta, self.W + reducedState[3:6], self.gyroBias + reducedState[6:9])

    def dQ(self):
        """
        renvoie la dérivée du quaternion
        """
        qw, qx, qy, qz = self.Q[0], self.Q[1], self.Q[2], self.Q[3]
        expQ = np.array([[-qx, -qy, -qz],
                         [qw, qz, -qy],
                         [-qz, qw, qx],
                         [qy, -qx, qw]])
        return 1 / 2 * np.dot(expQ, self.W)

    def evolv(self,dt):

        Qnump = self.Q.vec() + self.dQ() * dt  # calcul de la nouvelle orientation
        self.Q = Quaternion(*Qnump[:, 0])

    def stateMean(quatInit,LState):

        LQuat = [state.Q for state in LState]
        #quatMean = Quaternion(sum([quat[0] for quat in LQuat]),sum([quat[1] for quat in LQuat]),sum([quat[2] for quat in LQuat]),sum([quat[3] for quat in LQuat]))
        quatMean = Quaternion.mean(quatInit,LQuat,1e-2,1000)  #Quaternion moyen

        rotMean = np.zeros((3,1))  #Rotation moyenne
        biasMean = np.zeros((3,1))  #Biais moyen
        for state in LState:
            rotMean += state.W
            biasMean += state.gyroBias
        length = len(LState)
        rotMean /= length
        biasMean /= length

        return State(quatMean, rotMean, biasMean)
