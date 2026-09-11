from dataclasses import dataclass
import numpy as np
from source.heating_curves import TEMP_CURVES
from source.attraction_functions import F_LJ


@dataclass
class SimulationConfig:
    N: int = 20
    dt: float = 0.01
    T: int = 5000
    temp_curve: str = "first_hot_then_cold"


@dataclass
class SimulatorState:
    P: np.ndarray  # shape (T+1, 2, n_particles)
    m: np.ndarray  # shape (n_particles,)
    i: int = 1


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.N = config.N
        self.dt = config.dt
        self.T = config.T
        self.temp_curve = config.temp_curve
        self.state = self._build_initial_state()

    def _build_initial_state(self) -> SimulatorState:
        X, Y = np.meshgrid(range(self.N), range(int(self.N * 2 / np.sqrt(3))))
        X = X.flatten().astype(float)
        Y = Y.flatten().astype(float)
        offsets = (Y.astype(int) % 2) / 2
        X += offsets
        Y *= np.sqrt(3) / 2

        P0 = np.array([X, Y])
        n_particles = P0.shape[1]

        P = np.zeros((self.T + 1, 2, n_particles))
        P[0] = P0
        P[1] = P0

        return SimulatorState(P=P, m=np.ones(n_particles), i=1)

    @staticmethod
    def _contain_in_box(P_i: np.ndarray) -> np.ndarray:
        return np.clip(P_i, -1, 20)

    @staticmethod
    def _get_dist_mat(P_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dP = np.empty((2, P_i.shape[1], P_i.shape[1]))
        for k in range(2):
            p = P_i[k]
            dP[k] = np.reshape(p, (len(p), 1)) - np.reshape(p, (1, len(p)))
        d = np.sum(dP**2, axis=0)
        return dP, d

    @staticmethod
    def _get_force(dP: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = d.shape[0]
        d_safe = d + np.eye(n)  # diagonal is always 0 (self-distance); +1 there avoids div-by-zero

        f = dP / d_safe * F_LJ(d_safe, r0=1)

        idx = np.arange(n)
        f[:, idx, idx] = 0  # no self-interaction
        return f

    def _get_new_state(self, i: int) -> SimulatorState:
        P_i = self._contain_in_box(self.state.P[i])
        self.state.P[i] = P_i

        dP, d = self._get_dist_mat(P_i)
        f = self._get_force(dP, d)

        a = f @ self.state.m
        a += np.random.normal(0, TEMP_CURVES[self.temp_curve](t=i), size=a.shape)

        self.state.P[i + 1] = 2 * P_i - self.state.P[i - 1] + self.dt**2 * a
        return self.state

    def update(self, i: int):
        self.state.i = i
        self.state = self._get_new_state(i)

        # P_i = self._contain_in_box(self.state.P[i])
        # self.state.P[i] = P_i

        # dP, d = self._get_dist_mat(P_i)
        # f = self._get_force(dP, d)

        # a = f @ self.state.m
        # a += np.random.normal(0, TEMP_CURVES[self.temp_curve](t=i), size=a.shape)

        # self.state.P[i + 1] = 2 * P_i - self.state.P[i - 1] + self.dt**2 * a

    def run(self):
        for i in range(1, self.T):
            self.update(i)
        return self.state.P
