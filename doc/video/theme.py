"""Shared visual language for the *space-filling* explainer video.

Every scene subclasses :class:`VideoScene`, which fixes the background and
exposes a few helpers. Keeping the palette and helpers here lets the ten
scenes stay visually consistent.
"""

from pathlib import Path

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
