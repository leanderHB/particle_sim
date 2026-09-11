from source.simulator import Simulator, SimulationConfig
from source.gif_plotter import GifPlotter, GifPlotterConfig


def main():
    system = Simulator(SimulationConfig(T=500))
    plotter = GifPlotter(GifPlotterConfig(system=system, output_file="output.gif"))
    plotter.run()


if __name__ == "__main__":
    main()
