import numpy as np

class SunSensor:

    def __init__(self):
        self.Q = None

    def sunDir(t):
        angle = t * (2*np.pi)/365.25/24/3600 #Rotation of earth arround the sun each second
        earthRot = np.array([[np.cos(angle), np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
        sunDir = np.dot(earthRot,np.array([[1],[0],[0]]))
        return sunDir

    def update(Q):
        self.Q = Q

    def getNormalizedTension(self,t):
        U = np.zeros((6,1)
        sunDir_satRef = Q.V2R(sunDir(t))

        #Selon x
        u = np.dot(sunDir_satRef,np.array([[1],[0],[0]]))/np.norm(sunDir_satRef)
        if (u > 0):
            U[0,0] = u
        else:
            U[1,0] = -u

        #Selon y
        u = np.dot(sunDir_satRef,np.array([[0],[1],[0]]))/np.norm(sunDir_satRef)
        if (u > 0):
            U[2,0] = u
        else:
            U[3,0] = -u

        #Selon z
        u = np.dot(sunDir_satRef,np.array([[0],[0],[1]]))/np.norm(sunDir_satRef)
        if (u > 0):
            U[4,0] = u
        else:
            U[5,0] = -u

        return U
