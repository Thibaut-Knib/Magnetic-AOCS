import numpy as np
from errorHandler.state import State
from copy import copy
from environnement.sunSensor import SunSensor

class UKF:

    def __init__(self,q0,W0,I,gyroBias,P0,Qcov,Rcov,dt,recBool = False):
        """UKF filter
        :param q0: Initial estimation of the attitude quaternion
        :param W0: Initial estimation of the rotational velocity componant
        :param I: Inertial Matrix of the satellite
        :param gyroBias: Initial estimation of gyroscop bias error
        :param P0: Initial estimation of the Likelyhood covariance Matrix
        :param Qcov: Model noise covariace matrix (dimention 9, first attitude noise, then Velocity noise, then Biais Noise)
        :param Rcov: Measurment Noise covariance matrix (dimention 6, first gyroscop, then Magnetometer)
        :param dt: Time step of the different estimations
        """

        self.dim = 9  # Dimension of state (
        self.curState = State(q0,W0,I,gyroBias)  #Current state
        self.P = P0  #Covariance matrix on the state
        self.Qcov = Qcov  #Process noise
        self.Rcov = Rcov #covariance du modèle d'erreur de la mesure
        self.dt = dt  #Time step
        self.recBool = recBool
        if (self.recBool):
            self.record = {'stateIn':[],
                            'xk_': [],
                            'Pk_': [],
                            'nu': [],
                            'K': [],
                            'stateOut': [],
                            'PCorr': [],
                            'NormKal': []}

    def sigmaPoints(self):
        """ Generate a List of sigma points used to estimate the  mean and standard deviation of the pediction.
        Ref : Section 3.1 and 3.2 of Kraft 2003
        Cholesky algorythm is used to obtain the "square root matrix" of P + Q
        """
        try:
            sqrtmatrix = np.linalg.cholesky(self.P + self.Qcov)
            # print(sqrtmatrix)
        except np.linalg.LinAlgError as e:
            print(self.P)
            print(self.Qcov)
            raise Exception('Stranger things') from e

        res = [self.curState]  ## add Current estimate as a sigma point

        for i in range(self.dim):
            ajout = np.atleast_2d(sqrtmatrix[:,i]).T  #sqrt(2*self.dim) *
            res.append(self.curState.addition(ajout))
            res.append(self.curState.addition(-ajout))

        return res

    def evolvSigmaPoints(self, u, Xi):  #Xi the list of state vectors
        Yi = copy(Xi)
        for yi in Yi:
            yi.evolv(u,self.dt)

        return Yi

    def stateMean(self, Yi):  #Return the mean of the list Yi (yMean gave the initialisation for the quaternion mean)
        #Nécessaire pour initialiser la moyenne des quaternions
        yMean = copy(self.curState)
        # yMean.evolv(self.dt)

        return self.curState.stateMean(yMean.Q, Yi)

    def WiCalculus(self, Yi, xk_):
        """Calculation of the transformed sigma points error vectors.
        WiPrime = Yi - xk_ """
        WiPrime = []
        for yi in Yi:
            q = yi.Q*xk_.Q.inv()
            vec = q.axis()*q.angle()
            wi = np.zeros((self.dim,1))  #Reduced state
            wi[0:3] = vec
            wi[3:6] = yi.W - xk_.W
            wi[6:9] = yi.gyroBias - xk_.gyroBias
            WiPrime.append(wi)
        return WiPrime

    def aPrioriProcessCov(self, WiPrime):
        Pk_ = np.zeros((self.dim,self.dim))
        for wi in WiPrime:
            Pk_ += np.dot(wi,wi.T)
        Pk_ /= (2*self.dim)
        return Pk_

    def predictObs(self, Yi, LocalMagField_Rr, t):

        Zi = []
        ZRot = predictRotation(Yi)
        ZMagnet = predictMagnetField(Yi, LocalMagField_Rr)
        ZSun = predictSunSensor(Yi, t)
        for rot, mag, sun in zip(ZRot, ZMagnet, ZSun):
            zi = np.zeros((12, 1))
            zi[0:3] = rot
            zi[3:6] = mag
            zi[6:12] = sun
            Zi.append(zi)
        return Zi

    def innovation(self, zk_, WM, BM, uSunM):
        Zmesur = np.zeros((12,1))
        Zmesur[0:3] = WM
        Zmesur[3:6] = BM
        Zmesur[6:12] = uSunM
        return Zmesur - zk_

    def ObsCov(self, Zi, zk_):
        """Covariance between Expected measurments
         Correspond to Eq 68 from Kraft 2003
         :param Zi: Expected measurments from Sigma points
         :param zk_: Mea, expected measurment from Sigma point
         :return: P_zz Ucertainty of the predicted measurment

         """
        cov = np.zeros((12,12))
        for z in Zi:
            cov += np.dot(z-zk_,(z-zk_).T)
        cov /= len(Zi)

        return cov

    def crossCorrelationMatrix(self, WiPrime, Zi, zk_):
        """
        Compute cross corelation matrix from the
        :param WiPrime: Error state between sigma points and their mean
        :param Zi: Expected measurements from sigmapoints
        :param zk_: mean expected measurement
        :return: Pxz Cross correlation matrix between sigma points and measurments
        """
        crosscov = np.zeros((self.dim, 12))
        for w, z in zip(WiPrime, Zi):
            crosscov += np.dot(w, (z - zk_).T)

        crosscov /= len(Zi)

        return crosscov

    def errorCorrection(self, t, WM, BM, uSunM, LocalMagField_Rr, magMoment=np.array([[0],[0],[0]])):
        '''
        Renvoie au pas de temps de l'appel la correction de la mesure
        '''

        # prediction of state
        Xi = self.sigmaPoints() # Caclul des Wi, calcul des Xi et sauvegarde dans self.sigPoints
        #print([xi.Q for xi in Xi])
        Yi = self.evolvSigmaPoints(self.curState.Q.V2R(np.cross(magMoment, self.curState.Q.R2V(BM), axisa=0, axisb=0, axisc=0)),Xi) # couple dans Rv du au MC, Xi)  # process model, le bruit étant intégré dans les sigmaPoints
        #print([yi.Q for yi in Yi])
        xk_ = self.stateMean(Yi)
        #print(xk_.Q.axis()*xk_.Q.angle())
        WiPrime = self.WiCalculus(Yi, xk_)
        Pk_ = self.aPrioriProcessCov(WiPrime)

        # prediction of measure
        Zi = self.predictObs(Yi, LocalMagField_Rr, t)
        zk_ = obsMean(Zi)
        nu = self.innovation(zk_, WM, BM, uSunM)

        Pzz = self.ObsCov(Zi, zk_)
        Pnunu = self.Rcov + Pzz
        #print(Pnunu)
        Pxz = self.crossCorrelationMatrix(WiPrime, Zi, zk_)

        kalmanGain_k = cumputeKalmanGain(Pxz, Pnunu)
        #print(K)
        #K[0:3,3:6] = np.zeros((3,3))
        # K[6:9,0:6] = np.zeros((3,6))
        #K = np.zeros((9,6))

        xCorr = xk_.addition(np.dot(kalmanGain_k, nu))

        PCorr = Pk_ - np.dot(kalmanGain_k, np.dot(Pnunu, kalmanGain_k.T))

        #record
        if (self.recBool):
            self.record['stateIn'].append(self.curState)
            self.record['xk_'].append(xk_)
            self.record['Pk_'].append(Pk_)
            self.record['nu'].append(nu)
            self.record['K'].append(kalmanGain_k)
            self.record['stateOut'].append(xCorr)
            self.record['PCorr'].append(PCorr)
            self.record['NormKal'].append(np.linalg.norm(Pk_[0:3][0:3]))

        # Update
        self.curState = xCorr
        self.P = PCorr

def predictRotation(Yi):
    ZRot = [state.W + state.gyroBias for state in Yi]
    ZRot = [state.W for state in Yi]
    return ZRot


def predictMagnetField(Yi, LocalMagField_Rr):
    ZMagnet = []
    for yi in Yi:
        q = yi.Q
        ZMagnet.append(q.R2V(LocalMagField_Rr))
    return ZMagnet

def predictSunSensor(Yi,t):
    ZSun = []
    sunSensor = SunSensor()
    for yi in Yi:
        q = yi.Q
        sunSensor.update(q)
        ZSun.append(sunSensor.getNormalizedTension(t))
    return ZSun

def obsMean(Zi):
    mean = sum(Zi)/len(Zi)
    return mean


def cumputeKalmanGain(Pxz, Pnunu):
    return np.dot(Pxz, np.linalg.inv(Pnunu))
