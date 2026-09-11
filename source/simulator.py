from dataclasses import dataclass
import numpy as np
from source.heating_curves import TEMP_CURVES
from source.attraction_functions import F_LJ


@dataclass
class SimulationConfig:
    N: int = 20
    dt: float = 0.01
    T: int = 5000
    temp_curve = "first_hot_then_cold"


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.N = config.N
        self.dt = config.dt
        self.T = config.T
        self.i = 1
        self.temp_curve = config.temp_curve
        # initial geometry of the system, in this case Triangular grid

        X, Y = np.meshgrid(range(self.N), range(int(self.N * 2 / np.sqrt(3))))

        X = np.reshape(X, X.size).astype(float)
        Y = np.reshape(Y, Y.size).astype(float)
        offsets = (Y.astype(int) % 2) / 2
        X += offsets
        Y *= np.sqrt(3) / 2

        # plt.scatter(X,Y)
        # plt.show()
        P0 = np.array([X, Y])
        self.m = P0[1] * 0 + 1

        self.P = np.zeros((self.T, P0.shape[0], P0.shape[1]))

        # initial conditions
        self.P[0] = P0
        self.P[1] = P0

    def contain(self):
        self.P[0:2, :] = np.clip(self.P[0:2, :], -1, 20)

    def distMat(self):
        self.dP = np.array(
            [
                np.reshape(p, (len(p), 1)) - np.reshape(p, (1, len(p)))
                for p in self.P[self.i]
            ]
        )
        self.d = np.sum(self.dP**2, axis=0)

    def force(self):
        self.d[np.diag_indices_from(self.d)] = (
            1  # just to avoid divide by zero error message
        )
        self.f = [dp / self.d * F_LJ(self.d, r0=1) for dp in self.dP]
        for F in self.f:  # No self interaction
            # F +=np.random.normal(0,0.01,size=F.size)

            F[np.diag_indices_from(F)] = 0

    def update(self, i):
        self.i = i
        self.distMat()
        self.force()
        self.contain()
        a = self.f @ self.m
        a += np.random.normal(0, TEMP_CURVES[self.temp_curve](t=i), size=a.shape)
        self.P[i + 1] = self.P[i] * 2 - self.P[i - 1] + self.dt**2 * a
