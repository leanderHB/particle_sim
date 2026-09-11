from source.heating_curves import TEMP_CURVES
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from source.simulator import Simulator


@dataclass
class GifPlotterConfig:
    system: Simulator
    output_file: str


class GifPlotter:
    def __init__(self, config: GifPlotterConfig):
        self.system = config.system
        self.output_file = config.output_file
        self.fig, self.ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        self.dots1 = self.ax[0].plot(
            self.system.state.P[1][0], self.system.state.P[1][1], "o", color="blue"
        )
        self.dots2 = self.ax[0].plot(
            self.system.state.P[1][0], self.system.state.P[1][1], "o", color="red"
        )
        self.tempPlot = self.ax[1].plot([], [])

    def init(self):
        # fig.tight_layout()
        self.fig.set_size_inches((5, 8))

        self.ax[0].set_xlim([-2, self.system.N + 1])
        self.ax[0].set_ylim([-2, self.system.N + 1])
        self.ax[0].set_aspect("equal")
        self.ax[0].set_axis_off()

        self.ax[1].set_xlim([0, self.system.T])
        self.ax[1].set_ylim([0, 12])
        self.ax[1].set_xlabel("time")
        self.ax[1].set_ylabel("temperature")
        # self.ax[1].set_yscale('log')

    def update(self, i):

        self.system.update(i)
        self.ax[0].set_title(
            "$T=" + str(round(TEMP_CURVES[self.system.temp_curve](i), 1)) + "$"
        )

        L = self.system.state.P.shape[2]
        self.dots1[0].set_data(
            self.system.state.P[i, 0, : int(L / 2) + 10],
            self.system.state.P[i, 1, : int(L / 2) + 10],
        )
        self.dots2[0].set_data(
            self.system.state.P[i, 0, int(L / 2) + 10 :],
            self.system.state.P[i, 1, int(L / 2) + 10 :],
        )

        time_elapsed = list(range(1, i + 1))
        self.tempPlot[0].set_data(
            time_elapsed, [TEMP_CURVES[self.system.temp_curve](t) for t in time_elapsed]
        )

    def run(self):

        self.system.update(1)

        _, ax = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

        self.dots1 = ax[0].plot(
            self.system.state.P[1][0], self.system.state.P[1][1], "o", color="blue"
        )
        self.dots2 = ax[0].plot(
            self.system.state.P[1][0], self.system.state.P[1][1], "o", color="red"
        )
        self.tempPlot = ax[1].plot([], [])

        time = range(1, int(self.system.T - 1))
        ani = FuncAnimation(self.fig, self.update, time, init_func=self.init)

        writer = PillowWriter(fps=25)
        ani.save(self.output_file, writer=writer)
