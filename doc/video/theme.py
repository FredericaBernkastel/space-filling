"""Shared visual language for the *space-filling* explainer video.

Every scene subclasses :class:`VideoScene`, which fixes the background and
exposes a few helpers. Keeping the palette and helpers here lets the ten
scenes stay visually consistent.
"""

from pathlib import Path

import numpy as np
from manim import *

# --- palette -------------------------------------------------------------
BG = "#0d1117"         # near-black background
INK = "#e6edf3"        # primary text / equations
MUTED = "#8b949e"      # secondary text, captions, chrome
ACCENT = "#f0b429"     # warm highlight (arg-max, key terms)
COOL = "#58a6ff"       # cool highlight (min, links)
FIELD_LO = "#1f6feb"   # distance field: far / low
FIELD_MID = "#2dd4bf"  # distance field: mid
FIELD_HI = "#f2cc60"   # distance field: near maximum
TRAIL = "#7ee7ff"      # optimizer trajectory (contrasts with the gold maxima)

# Typography — deliberately small; the final composition is QHD (2560x1440),
# where large fonts read as clunky. Sizes are in manim's resolution-independent
# points, so these proportions hold at any render resolution.
FS_TITLE = 32     # scene titles / big statements
FS_H2 = 24        # section headers
FS_BODY = 20      # captions read at normal distance
FS_CAPTION = 17   # secondary captions
FS_CHIP = 15      # source chips, tile labels

ASSETS = Path(__file__).resolve().parent / "assets"


def asset(name: str) -> str:
    """Absolute path to a file in ``assets/`` (robust to the working dir)."""
    return str(ASSETS / name)


def plot_deco(center, u, Bd=2.0, grid=True):
    """Mathematica-ContourPlot-style chrome around a [-Bd,Bd]^2 panel at
    ``center`` with ``u`` scene units per field unit: integer grid, inward
    frame ticks with minors, numbers on the bottom/left edges, x/y labels."""
    def p(x, y):
        return center + np.array([x * u, y * u, 0.0])
    majors = range(-int(Bd), int(Bd) + 1)
    minors = [0.25 * k for k in range(-8, 9) if k % 4]
    parts = []
    if grid:
        glines = VGroup(*[Line(p(v, -Bd), p(v, Bd)) for v in majors if abs(v) < Bd],
                        *[Line(p(-Bd, v), p(Bd, v)) for v in majors if abs(v) < Bd])
        parts.append(glines.set_stroke(INK, width=1.0, opacity=0.3))
    ticks = VGroup()
    for vals, ln in ((majors, 0.09), (minors, 0.05)):
        for v in vals:
            ticks.add(Line(p(v, -Bd), p(v, -Bd) + ln * UP),
                      Line(p(v, Bd), p(v, Bd) + ln * DOWN),
                      Line(p(-Bd, v), p(-Bd, v) + ln * RIGHT),
                      Line(p(Bd, v), p(Bd, v) + ln * LEFT))
    parts.append(ticks.set_stroke(MUTED, width=1.4))
    parts.append(VGroup(*[m for v in majors for m in (
        MathTex(str(v), color=MUTED, font_size=15).next_to(p(v, -Bd), DOWN, buff=0.09),
        MathTex(str(v), color=MUTED, font_size=15).next_to(p(-Bd, v), LEFT, buff=0.09))]))
    parts.append(VGroup(
        MathTex("x", color=MUTED, font_size=18).next_to(p(0, -Bd), DOWN, buff=0.38),
        MathTex("y", color=MUTED, font_size=18).next_to(p(-Bd, 0), LEFT, buff=0.44)))
    return VGroup(*parts)


class VideoScene(MovingCameraScene):
    """Base scene: dark background, camera movement, and shared helpers."""

    def setup(self) -> None:
        super().setup()
        self.camera.background_color = BG

    # -- helpers ----------------------------------------------------------
    def source_chip(self, label: str) -> Text:
        """A small bottom-left tag naming an external source.

        Beats that stand in for a live screen-recording (a web page, the
        rendered README) carry one of these so the source is legible.
        """
        return Text(label, font_size=FS_CHIP, color=MUTED).to_corner(DL, buff=0.35)

    def eq_number(self, n: int, mob: Mobject) -> MathTex:
        """A right-margin equation number vertically aligned with ``mob``."""
        tag = MathTex(rf"({n})", color=MUTED).scale(0.75)
        return tag.to_edge(RIGHT, buff=0.8).set_y(mob.get_y())

    def scroll_image(self, path, width, run_time, framed=False, chip_label=None):
        """Fade in a tall image aligned to the top edge, scroll it up, fade out.

        Used for beats that show a pre-rendered document/webpage strip
        (README, Bourke page, ASDF paper).
        """
        img = ImageMobject(path)
        img.width = width
        top_gap = 0.4
        img.move_to([0, (config.frame_height / 2 - top_gap) - img.height / 2, 0])
        grp = Group(img)
        if framed:
            border = SurroundingRectangle(img, color=MUTED, buff=0.0).set_stroke(width=1.5)
            grp.add(border)
        chip = self.source_chip(chip_label) if chip_label else None
        intro = [FadeIn(grp)]
        if chip:
            intro.append(FadeIn(chip))
        self.play(*intro, run_time=0.8)
        scroll = max(img.height - config.frame_height + 2 * top_gap, 0.3)
        self.play(grp.animate.shift(UP * scroll), run_time=run_time, rate_func=linear)
        self.wait(0.3)
        outro = [FadeOut(grp)]
        if chip:
            outro.append(FadeOut(chip))
        self.play(*outro)
