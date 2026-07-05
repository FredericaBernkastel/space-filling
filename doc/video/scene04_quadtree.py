"""Scene 4 — the ADF paper, and a quadtree of buckets.

Beats: (1) scroll the *Adaptively Sampled Distance Fields* paper (per-node
polynomial approximation) and contrast it with this work (each node stores the
primitives themselves); (2) an adaptive quadtree over a field — a query descends
root→leaf and min-reduces over that leaf's short bucket, cost O(depth + β) =
O(log N); (3) a transition to the algebra section.

Render:
    uv run manim -ql scene04_quadtree.py Scene04Quadtree
"""

import numpy as np
from manim import *

import fields as F
from theme import (
    VideoScene, INK, MUTED, ACCENT, COOL, BG, TRAIL, FIELD_HI, asset,
    FS_H2, FS_BODY, FS_CAPTION,
)

B = 2.6
MAXD = 5
SHAPES = [
    F.circle(-1.3, 1.0, 0.62), F.box(1.25, 1.05, 0.5, 0.5),
    F.regular_polygon(1.3, -1.2, 3, 0.62), F.circle(-1.25, -1.15, 0.52),
    F.segment(-0.35, -0.15, 0.55, 0.45, 0.13),
]
CENTERS = [(-1.3, 1.0), (1.25, 1.05), (1.3, -1.2), (-1.25, -1.15), (0.1, 0.15)]


def _near_boundary(cx, cy, size):
    return min(abs(float(fn(cx, cy))) for fn in SHAPES) < size * 0.72


def _leaves():
    out = []

    def rec(cx, cy, size, d):
        if d < MAXD and _near_boundary(cx, cy, size):
            q = size / 4
            for dx, dy in [(-q, q), (q, q), (-q, -q), (q, -q)]:
                rec(cx + dx, cy + dy, size / 2, d + 1)
        else:
            out.append((cx, cy, size, d))

    rec(0.0, 0.0, 2 * B, 0)
    return out


def _descent(Q):
    path = []
    cx, cy, size, d = 0.0, 0.0, 2 * B, 0
    while True:
        path.append((cx, cy, size, d))
        if d < MAXD and _near_boundary(cx, cy, size):
            q = size / 4
            cx += q if Q[0] >= cx else -q
            cy += q if Q[1] >= cy else -q
            size, d = size / 2, d + 1
        else:
            return path


class Scene04Quadtree(VideoScene):
    def construct(self) -> None:
        self.beat_asdf()
        self.beat_quadtree()
        self.beat_transition()

    # ------------------------------------------------------------------ #
    def beat_asdf(self) -> None:
        """Scroll the ADF paper (per-node polynomial fit; we store primitives instead)."""
        self.scroll_image(asset("derived/asdf_paper.png"), width=6.4, run_time=9.0,
                          framed=True, chip_label="Frisken et al. 2000  ·  doi:10.1145/344779.344899")

    # ------------------------------------------------------------------ #
    def beat_quadtree(self) -> None:
        s = 6.0 / (2 * B)
        G = np.array([-1.6, 0.0, 0.0])

        def to_scene(x, y):
            return G + np.array([x * s, y * s, 0.0])

        field = F.field_image(F.union(*SHAPES), height=6.0, res=420, extent=B, interval=0.13).move_to(G)
        dim = Rectangle(width=6.0, height=6.0).set_fill(BG, opacity=0.45).set_stroke(width=0).move_to(G)
        border = SurroundingRectangle(field, color=MUTED, buff=0.0).set_stroke(width=1.5)

        cells = VGroup(*[
            Rectangle(width=size * s, height=size * s).move_to(to_scene(cx, cy))
            .set_stroke(COOL, width=0.8, opacity=0.6).set_fill(opacity=0)
            for (cx, cy, size, d) in _leaves()
        ])

        title = Text("adaptive quadtree of buckets", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)

        # right-hand info column (revealed progressively)
        desc_lbl = Text("descend: root → leaf", font_size=FS_CAPTION, color=ACCENT)
        blabel = Text("bucket B(ℓ): 2 primitives", font_size=FS_CAPTION, color=TRAIL)
        formula = MathTex(r"g(\vec v)=\min_{f \in B(\ell(\vec v))} f(\vec v)", color=INK).scale(0.62)
        cost = MathTex(r"O(\mathrm{depth}+\beta)=O(\log N)", color=FIELD_HI).scale(0.62)
        info = VGroup(desc_lbl, blabel, formula, cost).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        info.to_edge(RIGHT, buff=0.7).shift(UP * 0.2)

        self.play(FadeIn(field), FadeIn(dim), Create(border), run_time=1.0)
        self.play(FadeIn(title))
        self.play(Create(cells, lag_ratio=0.002), run_time=2.6)
        self.wait(0.4)

        # query descent
        Q = (-0.5, 0.5)
        path = _descent(Q)
        qdot = Dot(to_scene(*Q), radius=0.06, color=INK)
        self.play(FadeIn(qdot, scale=1.5), FadeIn(desc_lbl))
        cursor = (Rectangle(width=path[0][2] * s, height=path[0][2] * s)
                  .move_to(to_scene(path[0][0], path[0][1])).set_stroke(ACCENT, 3).set_fill(opacity=0))
        self.play(Create(cursor))
        for (cx, cy, size, d) in path[1:]:
            nxt = (Rectangle(width=size * s, height=size * s)
                   .move_to(to_scene(cx, cy)).set_stroke(ACCENT, 3).set_fill(opacity=0))
            self.play(Transform(cursor, nxt), run_time=0.45)
        self.play(cursor.animate.set_stroke(FIELD_HI, 4).set_fill(FIELD_HI, 0.12))
        self.wait(0.3)

        # bucket: arrows from the leaf to the two nearest primitives
        leaf = path[-1]
        order = sorted(range(len(SHAPES)), key=lambda i: abs(float(SHAPES[i](leaf[0], leaf[1]))))
        arrows = VGroup(*[
            Arrow(to_scene(leaf[0], leaf[1]), to_scene(*CENTERS[i]), buff=0.12,
                  stroke_width=3, color=TRAIL, max_tip_length_to_length_ratio=0.12)
            for i in order[:2]
        ])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.25), FadeIn(blabel))
        self.wait(0.8)

        self.play(FadeIn(formula, shift=UP * 0.1))
        self.play(FadeIn(cost, shift=UP * 0.1))
        self.wait(1.8)

        self.play(FadeOut(Group(
            field, dim, border, cells, qdot, cursor, arrows, title, info,
        )))

    # ------------------------------------------------------------------ #
    def beat_transition(self) -> None:
        line1 = Text("Next: the complete field algebra.", font_size=FS_H2, color=INK)
        line2 = Text("Prefer the results? Jump to 00:00.", font_size=FS_CAPTION, color=MUTED)
        grp = VGroup(line1, line2).arrange(DOWN, buff=0.4).move_to(ORIGIN)
        self.play(FadeIn(line1, shift=UP * 0.2))
        self.play(FadeIn(line2))
        self.wait(2.0)
        self.play(FadeOut(grp))
