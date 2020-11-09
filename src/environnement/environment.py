import sys
sys.path.append('..')
from src.environnement.magneticmodel import Model


class Environment:

    """
    Calcul du champ magnétique à la position du véhicule.

    """

    def __init__(self, magnetic_model):
        """
        r : radius, expressed in km
        i : inclination of the orbit
        u : argument of periapsis + true anomaly
        magnetic_model must be either 'dipole' or 'wmm'
        """
        self.r = None
        self.i = None
        self.u = None
        self.model = Model(magnetic_model)


    def setPosition(self, position):
        """
        Takes the tuple (r,i,u) as an argument.
        """
        self.model.setPosition(position)


    def getEnvironment(self):
        """
        Get the magnetic field value at the current position of the satellite (expressed in the intertial frame of reference).
        """
        B = self.model.getMagneticField()
        return B
