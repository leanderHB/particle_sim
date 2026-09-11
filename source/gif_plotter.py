from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from source.heating_curves import TEMP_CURVES
from source.simulator import Simulator

# Fixed palette so every frame is already paletted -> no per-frame PIL
# adaptive-quantize pass when saving the animated GIF.
_PALETTE = [
    255,
    255,
    255,  # 0 background
    0,
    0,
    0,  # 1 axes / text
    30,
    60,
    220,  # 2 group 1 dots
    220,
    40,
    40,  # 3 group 2 dots
]
BG, FG, BLUE, RED = 0, 1, 2, 3


@dataclass
class GifPlotterConfig:
    system: Simulator
    output_file: str
    width: int = 500
    height: int = 800
    fps: int = 25
    dot_radius: int = 3
    frame_interval: int = 1  # how many simulation steps to skip between frames


class GifPlotter:
    def __init__(self, config: GifPlotterConfig):
        self.system = config.system
        self.output_file = config.output_file
        self.width = config.width
        self.height = config.height
        self.fps = config.fps
        self.dot_radius = config.dot_radius
        self.frame_interval = config.frame_interval

        self.margin = 30
        self.top_h = int(self.height * 3 / 4)

        lo, hi = -2, self.system.N + 1
        self.lo = lo
        self.span = hi - lo
        plot_size = self.top_h - 2 * self.margin
        self.scale = plot_size / self.span
        self.ox = (self.width - self.span * self.scale) / 2

        self.font = ImageFont.load_default()

    def _to_px(self, x: float, y: float) -> tuple[float, float]:
        px = self.ox + (x - self.lo) * self.scale
        py = self.top_h - self.margin - (y - self.lo) * self.scale
        return px, py

    def _render_frame(self, i: int, temps: list[tuple[int, float]]) -> Image.Image:
        im = Image.new("P", (self.width, self.height), BG)
        im.putpalette(_PALETTE)
        draw = ImageDraw.Draw(im)

        P = self.system.state.P[i]
        n_particles = P.shape[1]
        split = n_particles // 2 + 10
        r = self.dot_radius
        for k in range(n_particles):
            px, py = self._to_px(P[0, k], P[1, k])
            draw.ellipse(
                [px - r, py - r, px + r, py + r], fill=BLUE if k < split else RED
            )

        temp = TEMP_CURVES[self.system.temp_curve](i)
        draw.text((10, 5), f"T={temp:.1f}", fill=FG, font=self.font)

        bx0, by0 = self.margin, self.top_h + 10
        bx1, by1 = self.width - self.margin, self.height - self.margin
        draw.rectangle([bx0, by0, bx1, by1], outline=FG)

        def bx(t: float) -> float:
            return bx0 + t / self.system.T * (bx1 - bx0)

        def by(v: float) -> float:
            return by1 - min(v, 12) / 12 * (by1 - by0)

        if len(temps) > 1:
            points = [(bx(t), by(v)) for t, v in temps]
            draw.line(points, fill=FG, width=1)

        draw.text((bx0, by1 + 5), "time", fill=FG, font=self.font)
        draw.text((2, by0), "temp", fill=FG, font=self.font)

        return im

    def run(self):
        frames = []
        temps = []
        for i in range(1, self.system.T - 1):
            self.system.update(i)
            if i % self.frame_interval != 0:
                continue
            temps.append((i, TEMP_CURVES[self.system.temp_curve](i)))
            frames.append(self._render_frame(i, temps))

        frames[0].save(
            self.output_file,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
        )
