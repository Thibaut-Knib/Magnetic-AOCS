import numpy as np
import matplotlib.pyplot as plt

class LiveVisu2D:
    """Vizualise in real time the evolution of the Satellite attitude and parameters"""

    def __init__(self):

        """Create the figure with some sumblopts"""
        self.fig, self.axarr = plt.subplots(2, 3, figsize=(12, 7))

        [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6] = self.axarr.flatten()

        self.linew, = self.ax1.plot([], [])
        self.linewx, = self.ax1.plot([], [])
        self.linewy, = self.ax1.plot([], [])
        self.linewz, = self.ax1.plot([], [])
        self.ax1.set_title("Rotational velocity in $R_r$")
        self.ax1.legend(["Norm", "$W_x$", "$W_y$", "$W_z$"])

        self.line2, = self.ax2.plot([], [])
        self.ax2.set_title("Amplitude of Magnetic moment")

        self.linebx, = self.ax3.plot([], [])
        self.lineby, = self.ax3.plot([], [])
        self.linebz, = self.ax3.plot([], [])
        self.lineb, = self.ax3.plot([], [])
        self.ax3.set_title("Measured Magnetic field in $R_v$")
        self.ax3.legend(["Norm", "$B_x$", "$B_y$", "$B_z$"])

        self.lineAngle, = self.ax4.plot([], [])
        self.ax4.set_title("Angle between W and B [rad]")

        self.linedB, = self.ax5.plot([], [])
        self.ax5.set_title("$\partial B / \partial t$")

        plt.show()

    def update_visu(self, output):
        """update the plots with the latests data"""

        self.linew.set_data(output["t"], output["W_norm"])
        self.linewx.set_data(output["t"], np.array(output["W"])[:, 0])
        self.linewy.set_data(output["t"], np.array(output["W"])[:, 1])
        self.linewz.set_data(output["t"], np.array(output["W"])[:, 2])
        self.line2.set_data(output["t"], output["M_norm"])

        self.linebx.set_data(output["t"], np.array(output["Bv"])[:, 0])
        self.lineby.set_data(output["t"], np.array(output["Bv"])[:, 1])
        self.linebz.set_data(output["t"], np.array(output["Bv"])[:, 2])

        self.lineb.set_data(output["t"], output["B_norm"])

        self.lineAngle.set_data(output["t"], output["angle"])

        self.linedB.set_data(output["t"], output["dB"])

        for ax in self.axarr.flatten():
            ax.relim()
            ax.autoscale_view(True, True, True)

        self.ax4.set_ylim(0, np.pi)

        plt.suptitle("Nt = {:1.1e}, ".format(len(output["t"])) +
                     "t = {:2.2e} $[s]$".format(output["t"][-1]), fontsize=12)

        plt.draw()
        plt.pause(0.01)  # Note this correction
