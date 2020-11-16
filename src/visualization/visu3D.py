
import vpython as vp
import numpy as np

class LiveVisu3D:
    """Vizualise in real time the evolution of the Satellite attitude"""

    def __init__(self, lx, ly, lz, B = [[1], [0], [0]]):
        """initialize a bunch of object to displays them : referance frame, satellite, and so on."""

        self.ux = vp.vector(1, 0, 0)
        self.uy = vp.vector(0, 1, 0)
        self.uz = vp.vector(0, 0, 1)
        # trièdre (z,x,y)
        self.axe_x_r = vp.arrow(pos=vp.vector(0, 0, 0), axis=10 * self.ux, shaftwidth=0.5, color=vp.vector(1, 0, 0))
        self.axe_y_r = vp.arrow(pos=vp.vector(0, 0, 0), axis=10 * self.uy, shaftwidth=0.5, color=vp.vector(0, 1, 0))
        self.axe_z_r = vp.arrow(pos=vp.vector(0, 0, 0), axis=10 * self.uz, shaftwidth=0.5, color=vp.vector(0, 0, 1))
        # création du satellite avec son repère propre
        self.sat_origin = vp.vector(10, 10, 10)
        self.axe_x_s = vp.arrow(pos=self.sat_origin, axis=10 * self.ux, shaftwidth=0.1, color=vp.vector(1, 0, 0))
        self.axe_y_s = vp.arrow(pos=self.sat_origin, axis=10 * self.uy, shaftwidth=0.1, color=vp.vector(0, 1, 0))
        self.axe_z_s = vp.arrow(pos=self.sat_origin, axis=10 * self.uz, shaftwidth=0.1, color=vp.vector(0, 0, 1))
        self.sugarbox = vp.box(pos=self.sat_origin, size=vp.vector(lx, ly, lz), axis=vp.vector(0, 0, 0), up=self.uy)
        self.satellite = vp.compound([self.axe_x_s, self.axe_y_s, self.axe_z_s, self.sugarbox])
        # vecteur champ B
        self.b_vector = vp.arrow(pos=vp.vector(-5, -5, -5), axis=10 * vp.vector(B[0][0], B[1][0], B[2][0]), shaftwidth=0.1,
                            color=vp.vector(1, 1, 1))

        # création de l'axe du vecteur rotation
        self.axe_W = vp.arrow(pos=self.sat_origin, axis=10 * self.ux, shaftwidth=0.1, color=vp.vector(1, 1, 1))

    def update_visu(self, B_r, sat_W, dt):
        """ update the visualisation """

        self.b_vector.axis = 1e6 * vp.vector(B_r[0][0], B_r[1][0], B_r[2][0])
        self.satellite.rotate(angle=np.linalg.norm(sat_W) * dt, axis=vp.vector(sat_W[0][0], sat_W[1][0], sat_W[2][0]),
                        origin=self.sat_origin)

        self.axe_W.axis = 5*vp.vector(vp.vector(*(sat_W/ np.linalg.norm(sat_W))))