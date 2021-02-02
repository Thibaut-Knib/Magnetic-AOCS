from math import acos, sqrt

import numpy as np


class Quaternion:
    """ The quaternion class

    Allows a simple use of quaternion
    """
    def __init__(self,a,b,c,d):
        """

        :param a: q_0
        :type a: float
        :param b: q_1
        :type b: float
        :param c: q_2
        :type c: float
        :param d: q_3
        :type d: float

        Normalize the quaternion at initialization.
        """
        norm = sqrt(a**2+b**2+c**2+d**2)
        self.a = a/norm  #: first element of the Quaternion q_0
        self.b = b/norm  #: second element of the Quaternion q_1
        self.c = c/norm  #: third element of the Quaternion q_2
        self.d = d/norm  #: forth element of the Quaternion q_3
        self.tmsave = None
        self.tminvsave = None

    def inv(self):
        """ Calcule le conjugué du quaternion.
            Ceux-ci étant unitaires, inverse = conjugué.
        """
        return Quaternion(self.a,-self.b,-self.c,-self.d)

    def vec(self):
        return np.array([[self.a],[self.b],[self.c],[self.d]])

    def __mul__(self,value):
        return Quaternion(
            self.a*value.a - self.b*value.b - self.c*value.c - self.d*value.d,
            self.b*value.a + self.a*value.b - self.d*value.c + self.c*value.d,
            self.c*value.a + self.d*value.b + self.a*value.c - self.b*value.d,
            self.d*value.a - self.c*value.b + self.b*value.c + self.a*value.d,
        )

    def tm(self): #transfer matrix from Rr to Rv i.e. X_Rr = M * X_Rv
        if self.tmsave is None:
            q0,q1,q2,q3 = self.a,self.b,self.c,self.d
            self.tmsave = np.array(
                [[2*(q0**2+q1**2)-1, 2*(q1*q2-q0*q3)  , 2*(q1*q3+q0*q2)  ],
                 [2*(q1*q2+q0*q3)  , 2*(q0**2+q2**2)-1, 2*(q2*q3-q0*q1)  ],
                 [2*(q1*q3-q0*q2)  , 2*(q2*q3+q0*q1)  , 2*(q0**2+q3**2)-1]]
            )
        return self.tmsave

    def tminv(self): #transfer matrix from Rv to Rr i.e. X_Rv = M * X_Rr
        if self.tminvsave is None:
            self.tminvsave = np.linalg.inv(self.tm())
        return self.tminvsave

    def __getitem__(self,index):
        if index == 0:
            return self.a
        elif index == 1:
            return self.b
        elif index == 2:
            return self.c
        elif index == 3:
            return self.d
        else:
            raise IndexError("Accessing a non-existing value of a 4 elements vector")

    def axis(self):
        res = np.array([[self.b],[self.c],[self.d]])
        if np.linalg.norm(res) == 0:
            return np.array([[1],[0],[0]])
        return res/np.linalg.norm(res)

    def angle(self):
        alpha = acos(max(-1,min(self.a,1)))*2
        if (alpha > np.pi):
            alpha -= (2*np.pi)
        return alpha

    def axialPart(self):
        res = np.array([[self.b],[self.c],[self.d]])
        return res

    def V2R(self,vec):
        return np.dot(self.tm(),vec)

    def R2V(self,vec):
        return np.dot(self.tminv(),vec)

    def __repr__(self):
        return "(" + str(self.a) + ", " + str(self.b) + ", " + str(self.c) + ", " + str(self.d) + ")"

    def __str__(self):
        return "(" + str(self.a) + ", " + str(self.b) + ", " + str(self.c) + ", " + str(self.d) + ")"

    def mean(quatInit, LQuat, tol, nMax):
        qt = quatInit

        testTol = False
        compteur = 0  #Compteur tours de boucle
        while not(testTol) and compteur < nMax:
            e = np.zeros((3,1))
            qtinf = qt.inv()
            for qi in LQuat:
                eiQuat = qi*qtinf
                alpha_i = eiQuat.angle()
                axis_i = eiQuat.axis()
                e += alpha_i*axis_i
            e /= (len(LQuat))

            alpha = np.linalg.norm(e)
            #Au lieu de considérer alpha/2, on prend alpha pour rendre la convergence plus rapide (en supposant que l'initialisation est pas trop loin du résultat)
            eQuat = Quaternion(np.cos(alpha),e[0,0]*np.sin(alpha),e[1,0]*np.sin(alpha),e[2,0]*np.sin(alpha))
            qt = eQuat*qt
            testTol = (alpha/2 < tol)

            compteur += 1

        return qt


# This file implements correct quaternion averaging.
#
# This method is computationally expensive compared to naive mean averaging.
# If only low accuracy is required (or the quaternions have similar orientations),
# then quaternion averaging can possibly be done through simply averaging the
# components.
#
# Based on:
#
# Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
# "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
# no. 4 (2007): 1193-1197.
# Link: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
#
# Code based on:
#
# Tolga Birdal. "averaging_quaternions" Matlab code.
# http://jp.mathworks.com/matlabcentral/fileexchange/40098-tolgabirdal-averaging-quaternions
#
# Comparison between different methods of averaging:
#
# Claus Gramkow. "On Averaging Rotations"
# Journal of Mathematical Imaging and Vision 15: 7–16, 2001, Kluwer Academic Publishers.
# https://pdfs.semanticscholar.org/ebef/1acdcc428e3ccada1a047f11f02895be148d.pdf
#
# Side note: In computer graphics, averaging or blending of two quaternions is often done through
# spherical linear interploation (slerp). Even though it's often used it might not be the best
# way to do things, as described in this post:
#
# Jonathan Blow.
# "Understanding Slerp, Then Not Using It", February 2004
# http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
#


# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4, 4))

    for i in range(0, M):
        q = Q[i, :]
        # multiply q with its transposed version q' and add A
        A = np.outer(q, q) + A

    # scale
    A = (1.0 / M) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0])


# Average multiple quaternions with specific weights
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4, 4))
    weightSum = 0

    for i in range(0, M):
        q = Q[i, :]
        A = w[i] * np.outer(q, q) + A
        weightSum += w[i]

    # scale
    A = (1.0 / weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0])
