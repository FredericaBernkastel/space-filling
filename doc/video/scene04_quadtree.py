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
    VideoScene, INK, MUTED, ACCENT, COOL, BG, TRAIL, FIELD_HI, asset, rich_text,
    FS_H2, FS_BODY, FS_CAPTION,
)

B = 2.6
MAXD = 5   # ADF::new's max_depth (a cap; this field never gets that deep)
BETA = 3   # BUCKET_SIZE in src/solver/adf/mod.rs::insert_where

# insertion order matters: the wall is the root's initial bucket (as in
# ADF::new), the shapes then arrive one at a time
PRIMS = {
    "w": F.frame(B),
    "circle": F.circle(-1.3, 1.0, 0.62),
    "box": F.box(1.25, 1.05, 0.5, 0.5),
    "tri": F.regular_polygon(1.3, -1.2, 3, 0.62),
    "circle2": F.circle(-1.25, -1.15, 0.52),
    "seg": F.segment(-0.35, -0.15, 0.55, 0.45, 0.13),
}
ORDER = ["circle", "box", "tri", "circle2", "seg"]
CENTERS = {"circle": (-1.3, 1.0), "box": (1.25, 1.05), "tri": (1.3, -1.2),
           "circle2": (-1.25, -1.15), "seg": (0.1, 0.15)}


def _union(keys):
    return F.union(*[PRIMS[k] for k in keys])


def _grid(rect, n=44):
    ax, ay, bx, by = rect
    return np.meshgrid(np.linspace(ax, bx, n), np.linspace(ay, by, n))


def _geq_on(fk, gkeys, rect):
    """f_k >= min(gkeys) on rect — numeric stand-in for sdf_geq_everywhere."""
    X, Y = _grid(rect)
    return float(np.min(PRIMS[fk](X, Y) - _union(gkeys)(X, Y))) >= -1e-9


def _bucket_geq(gkeys, fk, rect):
    X, Y = _grid(rect)
    return float(np.min(_union(gkeys)(X, Y) - PRIMS[fk](X, Y))) >= -1e-9


def _prune(keys, rect):
    return [k for k in keys
            if len(keys) == 1 or not _geq_on(k, [j for j in keys if j != k], rect)]


def _quadrants(rect):
    ax, ay, bx, by = rect
    mx, my = (ax + bx) / 2, (ay + by) / 2
    return [(ax, my, mx, by), (mx, my, bx, by), (ax, ay, mx, my), (mx, ay, bx, my)]


def _insert(node, k):
    """Mirror of ADF::insert_where: no-op / replace / append while the bucket
    has room / subdivide once it is full, pruning the combined set per child.
    Fresh children are not revisited within one insertion."""
    if node["kids"] is not None:
        for c in node["kids"]:
            _insert(c, k)
        return
    rect, bucket = node["rect"], node["bucket"]
    if _geq_on(k, bucket, rect):
        return                      # no-op: k never lowers the field here
    if _bucket_geq(bucket, k, rect):
        node["bucket"] = [k]        # k dominates the node
        return
    if node["depth"] == MAXD or len(bucket) < BETA:
        bucket.append(k)            # append (no re-prune, as in the Rust code)
        return
    combined = bucket + [k]
    node["kids"] = [{"rect": q, "depth": node["depth"] + 1,
                     "bucket": _prune(combined, q), "kids": None}
                    for q in _quadrants(rect)]


_TREE = None


def _tree():
    global _TREE
    if _TREE is None:
        _TREE = {"rect": (-B, -B, B, B), "depth": 0, "bucket": ["w"], "kids": None}
        for k in ORDER:
            _insert(_TREE, k)
    return _TREE


def _leaves():
    out = []

    def rec(node):
        if node["kids"] is None:
            out.append(node)
        else:
            for c in node["kids"]:
                rec(c)

    rec(_tree())
    return out


def _descent(Q):
    """Root-to-leaf path of nodes whose rect contains Q."""
    path, node = [], _tree()
    while True:
        path.append(node)
        if node["kids"] is None:
            return path
        node = next(c for c in node["kids"]
                    if c["rect"][0] <= Q[0] <= c["rect"][2]
                    and c["rect"][1] <= Q[1] <= c["rect"][3])


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

        field = F.field_image(_union(["w"] + ORDER), height=6.0, res=420, extent=B,
                              interval=0.13).move_to(G)
        dim = Rectangle(width=6.0, height=6.0).set_fill(BG, opacity=0.45).set_stroke(width=0).move_to(G)
        border = SurroundingRectangle(field, color=MUTED, buff=0.0).set_stroke(width=1.5)

        def cell_rect(rect, **kw):
            ax, ay, bx, by = rect
            return (Rectangle(width=(bx - ax) * s, height=(by - ay) * s, **kw)
                    .move_to(to_scene((ax + bx) / 2, (ay + by) / 2)))

        cells = VGroup(*[
            cell_rect(leaf["rect"]).set_stroke(COOL, width=1.2, opacity=0.75).set_fill(opacity=0)
            for leaf in _leaves()
        ])

        title = Text("adaptive quadtree of buckets", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)

        # query descent through the real tree (built by mirroring ADF::insert_where)
        Q = (0.32, -0.95)
        path = _descent(Q)
        leaf = path[-1]
        bucket = leaf["bucket"]

        # right-hand info column (revealed progressively)
        desc_lbl = rich_text("descend: root → leaf", font_size=FS_CAPTION, color=ACCENT)
        blabel = rich_text(f"bucket B(ℓ): {len(bucket)} primitives", font_size=FS_CAPTION, color=TRAIL)
        formula = MathTex(r"g(\vec v)=\min_{f \in B(\ell(\vec v))} f(\vec v)", color=INK).scale(0.62)
        cost = MathTex(r"O(\mathrm{depth}+\beta)=O(\log N)", color=FIELD_HI).scale(0.62)
        info = VGroup(desc_lbl, blabel, formula, cost).arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        info.to_edge(RIGHT, buff=0.7).shift(UP * 0.2)

        self.play(FadeIn(field), FadeIn(dim), Create(border), run_time=1.0)
        self.play(FadeIn(title))
        self.play(Create(cells, lag_ratio=0.04), run_time=2.2)
        self.wait(0.4)

        qdot = Dot(to_scene(*Q), radius=0.06, color=INK)
        self.play(FadeIn(qdot, scale=1.5), FadeIn(desc_lbl))
        cursor = cell_rect(path[0]["rect"]).set_stroke(ACCENT, 3).set_fill(opacity=0)
        self.play(Create(cursor))
        for node in path[1:]:
            nxt = cell_rect(node["rect"]).set_stroke(ACCENT, 3).set_fill(opacity=0)
            self.play(Transform(cursor, nxt), run_time=0.45)
        self.play(cursor.animate.set_stroke(FIELD_HI, 4).set_fill(FIELD_HI, 0.12))
        self.wait(0.3)

        # the leaf's actual bucket: arrows to every member (wall -> nearest edge)
        lx = (leaf["rect"][0] + leaf["rect"][2]) / 2
        ly = (leaf["rect"][1] + leaf["rect"][3]) / 2

        def member_target(k):
            if k != "w":
                return CENTERS[k]
            edges = [(B, ly), (-B, ly), (lx, B), (lx, -B)]
            return min(edges, key=lambda e: np.hypot(e[0] - lx, e[1] - ly))

        arrows = VGroup(*[
            Arrow(to_scene(lx, ly), to_scene(*member_target(k)), buff=0.12,
                  stroke_width=3, color=TRAIL, max_tip_length_to_length_ratio=0.12)
            for k in bucket
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
        line1 = Text("Next: the complete algebraic formalism.", font_size=FS_H2, color=INK)
        line2 = Text("Prefer the results? Jump to 00:00.", font_size=FS_CAPTION, color=MUTED)
        grp = VGroup(line1, line2).arrange(DOWN, buff=0.4).move_to(ORIGIN)
        self.play(FadeIn(line1, shift=UP * 0.2))
        self.play(FadeIn(line2))
        self.wait(2.0)
        self.play(FadeOut(grp))
