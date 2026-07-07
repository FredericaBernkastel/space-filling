"""Scene 9 — limitations and future work.

Two beats: (1) four numbered points, each with a small glyph — 2D only today;
insertion only; bring-your-own drawing; a GPU port, assessed. (2) the GPU
assessment expanded: a CPU/GPU split diagram (the device as a read-only field
oracle, every mutation staying on the host), the generality price (closures
cannot run on device — shape IR + baked distance fields instead), and the
lopsided theoretical gains.

Render:
    uv run manim -ql scene09_limitations.py Scene09Limitations
"""

import numpy as np
from manim import *

from theme import (
    VideoScene, mono, rich_text, INK, MUTED, ACCENT, COOL, TRAIL,
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


def gpu_chip(color=ACCENT):
    """A tiny IC-package icon: body, die, pins."""
    body = RoundedRectangle(width=0.66, height=0.66, corner_radius=0.07,
                            color=color, stroke_width=2)
    die = Square(side_length=0.3).set_stroke(color, 1.6)
    pins = VGroup()
    for off in (-0.18, 0.0, 0.18):
        pins.add(Line([-0.33, off, 0], [-0.46, off, 0]),
                 Line([0.33, off, 0], [0.46, off, 0]),
                 Line([off, 0.33, 0], [off, 0.46, 0]),
                 Line([off, -0.33, 0], [off, -0.46, 0]))
    pins.set_stroke(color, 1.6, opacity=0.75)
    return VGroup(pins, body, die)


class Scene09Limitations(VideoScene):
    def construct(self) -> None:
        self.beat_list()
        self.beat_gpu()

    # ------------------------------------------------------------------ #
    def beat_list(self) -> None:
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
            row("4", gpu_chip(),
                "A GPU port",
                "the device reads; every mutation stays on the CPU"),
        ]

        # align columns: number at x0, glyph centred at x1, text left-aligned at x2
        x0, x1, x2 = -6.1, -4.9, -3.6
        ys = [2.05, 0.72, -0.61, -1.94]
        for r, y in zip(rows, ys):
            num, g, text = r
            num.move_to(np.array([x0, y, 0.0]))
            g.move_to(np.array([x1, y, 0.0]))
            text.next_to(np.array([x2, y, 0.0]), RIGHT, buff=0).set_y(y)

        for r in rows:
            self.play(FadeIn(r[0]), FadeIn(r[1], scale=0.9),
                      FadeIn(r[2], shift=UP * 0.12), run_time=0.9)
            self.wait(1.6)

        self.wait(1.0)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])

    # ------------------------------------------------------------------ #
    def beat_gpu(self) -> None:
        """The GPU assessment: host/device split, the generality price, the
        lopsided gains (10-100x on ascents, 3-10x end to end)."""
        title = Text("A GPU port", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.45)

        def panel(label, color, cx):
            box = RoundedRectangle(width=4.9, height=3.2, corner_radius=0.14)
            box.set_stroke(color, 2).set_fill(color, 0.04).move_to(np.array([cx, 0.8, 0.0]))
            lab = Text(label, font_size=FS_CAPTION, color=color).next_to(box.get_top(), DOWN, buff=0.2)
            return box, lab

        cpu_box, cpu_lab = panel("CPU — owns every mutation", COOL, -3.7)
        gpu_box, gpu_lab = panel("GPU — read-only field oracle", ACCENT, 3.7)

        cpu_items = VGroup(
            mono("insert / subdivide / prune", font_size=FS_CHIP, color=INK),
            Text("the sound f64 path", font_size=FS_CHIP, color=MUTED),
            Text("dedup + refine the survivors", font_size=FS_CHIP, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(cpu_box.get_center() + DOWN * 0.28)

        arena = VGroup(*[Square(side_length=0.26).set_stroke(ACCENT, 1.4).set_fill(ACCENT, 0.12)
                         for _ in range(11)]).arrange(RIGHT, buff=0.06)
        arena_lbl = mono("nodes[] — 64 B, uploads as-is", font_size=FS_CHIP, color=MUTED)
        darts = VGroup(*[
            Arrow(ORIGIN, 0.42 * np.array([np.cos(a), np.sin(a), 0.0]), buff=0,
                  stroke_width=2.2, color=TRAIL, max_tip_length_to_length_ratio=0.35)
            for a in np.deg2rad([55, 75, 105])
        ]).arrange(RIGHT, buff=0.24, aligned_edge=DOWN)
        ascent_line = VGroup(darts, Text("mega-batches of f32 ascents", font_size=FS_CHIP, color=INK)
                             ).arrange(RIGHT, buff=0.25)
        gpu_items = VGroup(arena, arena_lbl, ascent_line
                           ).arrange(DOWN, buff=0.3).move_to(gpu_box.get_center() + DOWN * 0.28)

        a_up = Arrow(cpu_box.get_right() + UP * 0.62, gpu_box.get_left() + UP * 0.62,
                     buff=0.12, color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.12)
        a_up_lbl = Text("arena deltas", font_size=FS_CHIP, color=MUTED).next_to(a_up, UP, buff=0.08)
        a_dn = Arrow(gpu_box.get_left() + DOWN * 0.62, cpu_box.get_right() + DOWN * 0.62,
                     buff=0.12, color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.12)
        a_dn_lbl = Text("batch maxima — f32", font_size=FS_CHIP, color=MUTED).next_to(a_dn, DOWN, buff=0.08)

        # the price: closures cannot run on device
        closure = mono("Arc<dyn Fn(P2) -> F>", font_size=FS_CHIP, color=RED)
        strike = Line(closure.get_left() + 0.07 * LEFT, closure.get_right() + 0.07 * RIGHT,
                      color=RED, stroke_width=2.4)

        def pill(s):
            t = Text(s, font_size=FS_CHIP, color=GREEN)
            b = SurroundingRectangle(t, color=GREEN, buff=0.14, corner_radius=0.1).set_stroke(width=1.5)
            return VGroup(b, t)

        arrow_g = rich_text("→", font_size=FS_BODY, color=MUTED)
        trade = VGroup(VGroup(closure, strike), arrow_g, pill("shape IR"),
                       pill("baked distance fields")).arrange(RIGHT, buff=0.35)
        trade_cap = Text("the price is generality: the device cannot call function pointers",
                         font_size=FS_CHIP, color=MUTED)
        trade_grp = VGroup(trade, trade_cap).arrange(DOWN, buff=0.2).move_to(np.array([0.0, -1.75, 0.0]))

        # the lopsided gains
        def stat(value, label):
            v = mono(value, font_size=FS_H2, color=ACCENT)
            l = Text(label, font_size=FS_CHIP, color=MUTED)
            return VGroup(v, l).arrange(DOWN, buff=0.1)

        gains = VGroup(stat("10–100×", "the ascent phase"),
                       stat("3–10×", "end to end — Amdahl")).arrange(RIGHT, buff=1.5)
        batch_note = rich_text("a batch's survivors are capped by disjoint free balls — "
                               "batch size scales with fill state, not core count",
                               font_size=FS_CHIP, color=MUTED)
        gains_grp = VGroup(gains, batch_note).arrange(DOWN, buff=0.28).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(title))
        self.play(Create(cpu_box), FadeIn(cpu_lab), Create(gpu_box), FadeIn(gpu_lab))
        self.play(FadeIn(cpu_items, shift=UP * 0.1), FadeIn(gpu_items, shift=UP * 0.1))
        self.play(GrowArrow(a_up), FadeIn(a_up_lbl))
        self.play(GrowArrow(a_dn), FadeIn(a_dn_lbl))
        self.wait(0.8)
        self.play(FadeIn(closure))
        self.play(Create(strike), FadeIn(arrow_g), FadeIn(trade[2], shift=UP * 0.1),
                  FadeIn(trade[3], shift=UP * 0.1), FadeIn(trade_cap))
        self.wait(0.8)
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in gains], lag_ratio=0.25),
                  FadeIn(batch_note))
        self.wait(2.4)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])
