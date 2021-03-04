import sys
sys.path.append('..')
from src.environnement.magneticmodel import Model
from src.environnement.sunSensor import SunSensor


class Environment:

    """
    Calcul du champ magnétique à la position du véhicule.

    """

    def __init__(self, magnetic_model, dt):
        """
        r : radius, expressed in km
        i : inclination of the orbit
        u : argument of periapsis + true anomaly
        magnetic_model must be either 'dipole' or 'wmm'
        """
        self.r = None
        self.i = None
        self.u = None
        self.mag_model = Model(magnetic_model)
        self.sun_sensor = SunSensor()
        self.dt = dt
        self.t = 0


    def setAttitudePosition(self, Q, position):
        """
        Takes the tuple (r,i,u) as an argument.
        """
        self.mag_model.setPosition(position)
        self.sun_sensor.update(Q)
        self.t += self.dt


    def getEnvironment(self):
        """
        Get the magnetic field value at the current position of the satellite (expressed in the intertial frame of reference).
        """
        B = self.model.getMagneticField()
        U = self.sun_sensor.getNormalizedTension(self.t)
        return B,U
