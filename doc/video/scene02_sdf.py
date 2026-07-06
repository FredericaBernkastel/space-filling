"""Scene 2 — from rejection sampling to signed distance fields.

Beats: (1) rejection sampling stalls in a dense packing; (2) a grid of shape
SDFs; (3) the SDF definition + its two key properties; (4) the compound field
g = min sdf_n; (5) a gradient-ascent trajectory to a maximum; (6) the O(N^2)
catch.

Render:
    uv run manim -ql scene02_sdf.py Scene02SDF
"""

import random

import numpy as np
from manim import *

import fields as F
from theme import (
    VideoScene, INK, MUTED, ACCENT, COOL, BG, TRAIL,
    FS_H2, FS_BODY, FS_CAPTION, FS_CHIP, mono_span
)

RED = "#ff5f57"
GREEN = "#3fb950"


class Scene02SDF(VideoScene):
    def construct(self) -> None:
        self.beat_rejection()
        self.beat_grid()
        self.beat_sdf_formula()
        self.beat_compound()
        self.beat_ascent()
        self.beat_complexity()

    # ------------------------------------------------------------------ #
    def beat_rejection(self) -> None:
        """Dense packing of large circles; probes rejected (red) until one is free."""
        random.seed(5)
        xr, yr = 6.3, 3.45
        packed = []
        for _ in range(20000):
            r = 0.32 + 1.45 * (random.random() ** 2)  # bias small, so gaps fill
            x = random.uniform(-xr + r, xr - r)
            y = random.uniform(-yr + r, yr - r)
            if all((x - px) ** 2 + (y - py) ** 2 > (r + pr + 0.02) ** 2 for px, py, pr in packed):
                packed.append((x, y, r))

        circles = VGroup(*[
            Circle(radius=r, color=COOL, stroke_width=2).set_fill(COOL, opacity=0.14).move_to([x, y, 0])
            for x, y, r in packed
        ])
        title = Text("rejection sampling", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)

        self.play(LaggedStart(*[GrowFromCenter(c) for c in circles], lag_ratio=0.015), run_time=2.0)
        self.play(FadeIn(title))

        def inside(x, y):
            return any((x - px) ** 2 + (y - py) ** 2 <= pr * pr for px, py, pr in packed)

        reds = []
        while len(reds) < 12:
            x, y = random.uniform(-xr, xr), random.uniform(-yr, yr)
            if inside(x, y):
                reds.append((x, y))
        green = next(
            (x, y) for x, y in ((random.uniform(-xr, xr), random.uniform(-yr, yr)) for _ in range(40000))
            if not inside(x, y)
        )

        counter = MarkupText(f'attempts: {mono_span("0")}', font_size=FS_BODY, color=INK).to_corner(DR, buff=0.5)
        self.play(FadeIn(counter))
        dots = VGroup()
        for i, (x, y) in enumerate(reds, start=1):
            dot = Dot([x, y, 0], radius=0.08, color=RED)
            self.play(
                FadeIn(dot, scale=1.6),
                Transform(counter, MarkupText(f"attempts: {mono_span(i)}", font_size=FS_BODY, color=INK).to_corner(DR, buff=0.5)),
                run_time=0.16,
            )
            dots.add(dot)

        gdot = Dot([green[0], green[1], 0], radius=0.11, color=GREEN)
        ring = Circle(radius=0.28, color=GREEN, stroke_width=3).move_to([green[0], green[1], 0])
        self.play(FadeIn(gdot, scale=1.6))
        self.play(Create(ring), Flash(gdot, color=GREEN, line_length=0.22))
        self.wait(1.2)
        self.play(FadeOut(VGroup(circles, dots, gdot, ring, counter, title)))

    # ------------------------------------------------------------------ #
    def beat_grid(self) -> None:
        """A 5x3 grid of shape SDFs (most of shapes.rs), each labelled with its
        full field notation, verbatim from the shapes.rs doc comments
        (norms as ||.||, componentwise abs as |.|)."""
        specs = [
            ("Circle", [r"\lVert\vec p\rVert-1"],
             F.circle(0, 0, 1.0)),
            ("Square", [r"\lVert\max(\vec q,\vec 0)\rVert+\min(\max(q_x,q_y),0)",
                        r"\vec q=\lvert\vec p\rvert-1"],
             F.box(0, 0, 0.9, 0.9)),
            ("Triangle", [r"\max_i\,(\vec p\cdot\vec n_i)-\cos(\pi/3)"],
             F.regular_polygon(0, 0, 3, 1.0)),
            ("Pentagon", [r"\max_i\,(\vec p\cdot\vec n_i)-\cos(\pi/5)"],
             F.regular_polygon(0, 0, 5, 1.0)),
            ("Hexagon", [r"\max_i\,(\vec p\cdot\vec n_i)-\cos(\pi/6)"],
             F.regular_polygon(0, 0, 6, 1.0)),
            ("Octagon", [r"\max_i\,(\vec p\cdot\vec n_i)-\cos(\pi/8)"],
             F.regular_polygon(0, 0, 8, 1.0)),
            ("Star", [r"\pm\lVert\vec p\,'-\mathrm{proj}(\vec p\,')\rVert",
                      r"(n,m)=(7,3)"],
             F.star(0, 0, 7, 3.0, 1.0)),
            ("Pentagram", [r"=\mathrm{Star}\bigl(n{=}5,\ m{=}\tfrac{10}{3}\bigr)"],
             F.pentagram(0, 0, 1.0)),
            ("Hexagram", [r"=\mathrm{Star}(n{=}6,\ m{=}3)"],
             F.hexagram(0, 0, 1.0)),
            ("Moon", [r"\max\bigl(\lVert\vec p\rVert-1,\ 1-\lVert\vec p-(d,0)\rVert\bigr)"],
             F.moon(0, 0, 1.0, 0.6)),
            ("Cross", [r"\mathrm{sgn}(k)\,\lVert\max(\vec w,\vec 0)\rVert"],
             F.cross(0, 0, 0.32, 1.0)),
            ("Ring", [r"\max(\lVert\vec p\rVert-1,\ \rho-\lVert\vec p\rVert)"],
             F.ring(0, 0, 0.5, 1.0)),
            ("Kakera", [r"\pm\lVert\vec q-\mathrm{proj}(\vec q)\rVert"],
             F.kakera(0, 0, 0.6, 1.0)),
            ("Line", [r"\mathrm{dist}(\vec p,[\vec a,\vec b])-t/2"],
             F.segment(-0.8, -0.5, 0.8, 0.5, 0.18)),
            ("Polygon", [r"s\cdot\min_i\,\mathrm{dist}(\vec p,\vec e_i)"],
             F.polygon([(-0.9, -0.5), (0.8, -0.7), (0.5, 0.9), (-0.3, 0.4)])),
        ]
        tiles = []
        for name, notation, fn in specs:
            img = F.field_image(fn, height=1.3, res=210, extent=1.5, interval=0.14)
            border = SurroundingRectangle(img, color=MUTED, buff=0.0).set_stroke(width=1)
            name_t = Text(name, font_size=FS_CHIP, color=INK)
            note_t = VGroup(*[MathTex(line, color=MUTED).scale(0.34) for line in notation]
                            ).arrange(DOWN, buff=0.06)
            label = VGroup(name_t, note_t).arrange(DOWN, buff=0.06).next_to(img, DOWN, buff=0.1)
            tiles.append(Group(img, border, label))

        grid = Group(*tiles).arrange_in_grid(rows=3, cols=5, buff=(0.5, 0.35))
        grid.scale_to_fit_height(6.6)
        if grid.width > 13.5:
            grid.scale_to_fit_width(13.5)
        grid.move_to(0.45 * DOWN)
        intro = Text("represent every shape by a signed distance function",
                     font_size=FS_BODY, color=MUTED).to_edge(UP, buff=0.35)

        self.play(FadeIn(intro))
        self.play(LaggedStart(*[FadeIn(t, scale=0.85) for t in tiles], lag_ratio=0.08), run_time=2.6)
        self.wait(1.8)
        self.play(FadeOut(intro), FadeOut(grid))

    # ------------------------------------------------------------------ #
    def beat_sdf_formula(self) -> None:
        """The SDF definition, then its two carrying properties."""
        sdf = MathTex(
            r"\mathrm{sdf}_{S}(\vec v)", r"=",
            r"\bigl(\mathbf{1}_{\vec v \notin S} - \mathbf{1}_{\vec v \in S}\bigr)",
            r"\inf_{\vec u \in \partial S}\lVert \vec v - \vec u\rVert",
            color=INK,
        ).scale(0.85).move_to(0.7 * UP)
        caption = Text("distance to the boundary — negative inside, positive outside",
                       font_size=FS_CAPTION, color=MUTED).next_to(sdf, DOWN, buff=0.45)

        self.play(Write(sdf), run_time=2.2)
        self.play(FadeIn(caption))
        self.wait(0.8)
        self.play(sdf[2].animate.set_color(ACCENT), sdf[3].animate.set_color(COOL))
        self.wait(1.0)

        p1 = MathTex(r"\lVert \nabla\,\mathrm{sdf}\rVert = 1 \ \text{a.e.}", color=INK).scale(0.68)
        p2 = MathTex(r"\text{1-Lipschitz}", color=INK).scale(0.68)

        def chip(m):
            box = SurroundingRectangle(m, color=MUTED, buff=0.22, corner_radius=0.1)
            box.set_stroke(width=1).set_fill(WHITE, opacity=0.03)
            return VGroup(box, m)

        row = VGroup(chip(p1), chip(p2)).arrange(RIGHT, buff=0.7).next_to(caption, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in row], lag_ratio=0.3))
        self.wait(1.6)
        self.play(FadeOut(VGroup(sdf, caption, row)))

    # ------------------------------------------------------------------ #
    def beat_compound(self) -> None:
        """The compound field g = min_n sdf_n over a bounded domain."""
        self.B = 2.6
        H = 6.0
        self.scale_f = H / (2 * self.B)
        self.shapes = [
            F.circle(-1.55, 1.2, 0.7), F.box(1.45, 1.4, 0.55, 0.55),
            F.regular_polygon(1.7, -1.35, 3, 0.72), F.regular_polygon(-1.65, -1.4, 5, 0.6),
            F.segment(-0.5, -0.3, 0.7, 0.4, 0.14), F.circle(0.1, 2.15, 0.38),
        ]
        self.gscalar = F.union(F.frame(self.B), *self.shapes)

        self.field_img = F.field_image(
            self.gscalar, height=H, res=560, extent=self.B, interval=0.12
        ).move_to(ORIGIN)
        self.field_border = SurroundingRectangle(self.field_img, color=MUTED, buff=0.0).set_stroke(width=1.5)

        formula = MathTex(r"g(\vec v) = \min_n \mathrm{sdf}_n(\vec v)", color=INK).scale(0.72)
        panel = SurroundingRectangle(formula, color=MUTED, buff=0.22, corner_radius=0.1)
        panel.set_stroke(width=1).set_fill(BG, opacity=0.85)
        self.compound_formula = VGroup(panel, formula).to_corner(UL, buff=0.4)

        self.play(FadeIn(self.field_img), Create(self.field_border), run_time=1.2)
        self.play(FadeIn(self.compound_formula, shift=DOWN * 0.1))
        self.wait(0.8)

        note = Text("a minimum of 1-Lipschitz functions is 1-Lipschitz", font_size=FS_CAPTION, color=MUTED)
        note.to_edge(DOWN, buff=0.4)
        nbg = SurroundingRectangle(note, buff=0.15).set_fill(BG, opacity=0.85).set_stroke(width=0)
        self.compound_note = VGroup(nbg, note)
        self.play(FadeIn(self.compound_note))
        self.wait(1.4)
        self.play(FadeOut(self.compound_note))

    # ------------------------------------------------------------------ #
    def beat_ascent(self) -> None:
        """A single gradient-ascent trajectory climbing to a maximum."""
        s = self.scale_f
        # start inside the pentagon so the trajectory is long: it climbs out of
        # the shape, crosses the white zero-level boundary, and settles at a pocket
        path = F.ascend(self.gscalar, (-1.55, -1.3))
        pts = [np.array([p[0] * s, p[1] * s, 0.0]) for p in path]

        traj = VMobject(color=TRAIL, stroke_width=5)
        if len(pts) > 2:
            traj.set_points_smoothly(pts)
        else:
            traj.set_points_as_corners(pts)
        start_dot = Dot(pts[0], radius=0.08, color=TRAIL)
        mover = Dot(pts[0], radius=0.10, color=TRAIL)

        label = Text("gradient ascent", font_size=FS_BODY, color=TRAIL)
        lbg = SurroundingRectangle(label, buff=0.15).set_fill(BG, opacity=0.85).set_stroke(width=0)
        lab = VGroup(lbg, label).to_corner(UR, buff=0.4)

        self.play(FadeIn(start_dot), FadeIn(lab))
        self.play(Create(traj), MoveAlongPath(mover, traj), run_time=2.6, rate_func=smooth)

        end = path[-1]
        d = float(self.gscalar(end[0], end[1]))
        ball = Circle(radius=d * s, color=TRAIL, stroke_width=3).move_to(pts[-1])
        center = Dot(pts[-1], radius=0.05, color=TRAIL)
        maxlab = MathTex(r"\vec x^{*}", color=TRAIL).scale(0.7).next_to(pts[-1], UR, buff=0.12)
        mbg = SurroundingRectangle(maxlab, buff=0.08).set_fill(BG, opacity=0.8).set_stroke(width=0)
        self.play(Create(ball), FadeIn(center), FadeIn(VGroup(mbg, maxlab)))
        self.wait(1.6)

        self.play(FadeOut(Group(
            self.field_img, self.field_border, self.compound_formula,
            traj, mover, start_dot, ball, center, mbg, maxlab, lab,
        )))

    # ------------------------------------------------------------------ #
    def beat_complexity(self) -> None:
        """The catch: O(N) per query over N shapes is O(N^2)."""
        line = MathTex(
            r"O(N)\ \text{per query}", r"\times", r"N\ \text{shapes}", r"=", r"O(N^2)",
            color=INK,
        ).scale(0.8)
        self.play(Write(line), run_time=1.8)
        self.wait(0.4)
        self.play(line[4].animate.set_color(ACCENT).scale(1.15))
        self.wait(2.0)
        self.play(FadeOut(line))
