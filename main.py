from source.simulator import Simulator, SimulationConfig
from source.gif_plotter import GifPlotter, GifPlotterConfig


def main():
    system = Simulator(SimulationConfig(T=10000, dt=0.01))
    plotter = GifPlotter(
        GifPlotterConfig(system=system, output_file="output.gif", frame_interval=40)
    )
    plotter.run()


if __name__ == "__main__":
    main()
