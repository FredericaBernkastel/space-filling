"""Scene 9 — limitations and future work.

Three numbered points, each with a small glyph: (1) 2D only today, but nothing
assumes a dimensionality — N-D is straightforward; (2) insertion only — no
deletion, no movement; (3) basic drawing included, designed to sit in front of
any drawing API.

Render:
    uv run manim -ql scene09_limitations.py Scene09Limitations
"""

import numpy as np
from manim import *

from theme import (
    VideoScene, INK, MUTED, ACCENT, COOL, TRAIL,
    FS_H2, FS_BODY, FS_CAPTION, FS_CHIP,
)

GREEN = "#3fb950"
RED = "#ff6b6b"


def wire_cube(side=0.62, off=0.26, color=COOL):
    """A square-to-octree hint: front face, back face, connecting edges."""
    front = Square(side_length=side, color=color, stroke_width=2)
    back = Square(side_length=side, color=color, stroke_width=1.4)
    back.shift(np.array([off, off, 0.0])).set_stroke(opacity=0.65)
    edges = VGroup(*[
        Line(front.get_corner(c), back.get_corner(c), color=color,
             stroke_width=1.4, stroke_opacity=0.65)
        for c in (UL, UR, DL, DR)
    ])
    return VGroup(back, edges, front)


def op_chip(symbol, ok):
    """A small operation chip: ring + symbol, struck through when unsupported."""
    col = GREEN if ok else MUTED
    ring = Circle(radius=0.26, color=col, stroke_width=2)
    sym = MathTex(symbol, color=col).scale(0.62).move_to(ring)
    chip = VGroup(ring, sym)
    if not ok:
        chip.add(Line(ring.point_at_angle(PI * 3 / 4), ring.point_at_angle(-PI / 4),
                      color=RED, stroke_width=2.5))
    return chip


def frame_glyph(color=TRAIL):
    """A tiny picture-frame icon: canvas, sun, mountain."""
    canvas = RoundedRectangle(width=1.0, height=0.78, corner_radius=0.08,
                              color=color, stroke_width=2)
    sun = Dot(canvas.get_center() + np.array([-0.22, 0.16, 0.0]), radius=0.06, color=color)
    hill = VMobject(color=color, stroke_width=2).set_points_as_corners([
        canvas.get_corner(DL) + np.array([0.12, 0.02, 0.0]),
        canvas.get_center() + np.array([0.05, -0.08, 0.0]),
        canvas.get_center() + np.array([0.22, 0.10, 0.0]),
        canvas.get_corner(DR) + np.array([-0.10, 0.02, 0.0]),
    ])
    return VGroup(canvas, sun, hill)


class Scene09Limitations(VideoScene):
    def construct(self) -> None:
        title = Text("limitations & future work", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.45)
        self.play(FadeIn(title))

        def row(num, glyph, head, sub):
            n = Text(num, font_size=FS_BODY, color=MUTED)
            h = Text(head, font_size=FS_BODY, color=INK)
            s = Text(sub, font_size=FS_CHIP, color=MUTED)
            text = VGroup(h, s).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
            g = VGroup(glyph)
            return VGroup(n, g, text)

        rows = [
            row("1", wire_cube(),
                "the 2D plane only — for now",
                "nothing in the quadtree, the SDFs, or the optimizer assumes a dimensionality; N-D is straightforward"),
            row("2", VGroup(op_chip(r"+", True), op_chip(r"-", False),
                            op_chip(r"\leftrightarrow", False)).arrange(RIGHT, buff=0.18),
                "insertion only",
                "no deletion, no movement"),
            row("3", frame_glyph(),
                "basic drawing included — bring your own",
                "intended to be compatible with any drawing API out of the box"),
        ]

        # align columns: number at x0, glyph centred at x1, text left-aligned at x2
        x0, x1, x2 = -6.1, -4.9, -3.6
        ys = [1.55, -0.05, -1.65]
        for r, y in zip(rows, ys):
            num, g, text = r
            num.move_to(np.array([x0, y, 0.0]))
            g.move_to(np.array([x1, y, 0.0]))
            text.next_to(np.array([x2, y, 0.0]), RIGHT, buff=0).set_y(y)

        for r in rows:
            self.play(FadeIn(r[0]), FadeIn(r[1], scale=0.9),
                      FadeIn(r[2], shift=UP * 0.12), run_time=0.9)
            self.wait(1.6)

        self.wait(1.4)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])
