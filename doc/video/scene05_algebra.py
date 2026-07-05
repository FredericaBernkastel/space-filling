"""Scene 5 — the field algebra.

Five beats: (1) the flat arena node layout; (2) Lipschitz continuity as a cone;
(3) the sdf_geq_everywhere branch-and-bound over f-g; (4) the four insertion
cases + prune; (5) the D* insertion domains (global vs local maximum).

Render:
    uv run manim -ql scene05_algebra.py Scene05Algebra
"""

import numpy as np
from manim import *

import fields as F
from theme import (
    VideoScene, plot_deco, INK, MUTED, ACCENT, COOL, BG, TRAIL, FIELD_HI,
    FS_TITLE, FS_H2, FS_BODY, FS_CAPTION, FS_CHIP,
)

GREEN = "#3fb950"
RED = "#ff6b6b"
BLUE = "#58a6ff"


class Scene05Algebra(VideoScene):
    def construct(self) -> None:
        self.beat_arena()
        self.beat_lipschitz()
        self.beat_bnb()
        self.beat_insert()
        self.beat_domains()

    # ------------------------------------------------------------------ #
    def beat_arena(self) -> None:
        """Flat arena: a quadtree stored in one array; 2-D cell <-> array index."""
        title = Text("flat arena — the quadtree in one contiguous array",
                     font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)
        cols = ["#58a6ff", "#3fb0d6", "#2dd4bf", "#39c98a", "#3fb950",
                "#7fc74a", "#c9c247", "#e6b23a", "#f0b429"]

        # --- 2-D quadtree (left): root 0; children 1..4; node 1 split into 5..8 ---
        QC = np.array([-4.1, 0.5, 0.0])
        QS = 2.8
        h = QS / 2
        root = Square(side_length=QS, color=cols[0], stroke_width=3).move_to(QC).set_fill(opacity=0)
        root_lbl = Text("0", font_size=FS_CHIP, color=cols[0]).next_to(root, UL, buff=0.06)
        qpos = {1: (-1, 1), 2: (1, 1), 3: (-1, -1), 4: (1, -1)}
        cells2d, lbl2d = {}, {}
        for i in [2, 3, 4]:
            cx, cy = QC[0] + qpos[i][0] * h / 2, QC[1] + qpos[i][1] * h / 2
            cells2d[i] = Square(side_length=h, stroke_width=1.5).set_stroke(MUTED).set_fill(cols[i], 0.4).move_to([cx, cy, 0])
            lbl2d[i] = Text(str(i), font_size=FS_CHIP, color=INK).move_to([cx, cy, 0])
        n1 = np.array([QC[0] - h / 2, QC[1] + h / 2, 0.0])
        node1 = Square(side_length=h, color=cols[1], stroke_width=2.5).move_to(n1).set_fill(opacity=0)
        n1_lbl = Text("1", font_size=14, color=cols[1]).next_to(node1, UL, buff=0.03)
        for j, i in enumerate([5, 6, 7, 8]):
            dx, dy = qpos[j + 1]
            cx, cy = n1[0] + dx * h / 4, n1[1] + dy * h / 4
            cells2d[i] = Square(side_length=h / 2, stroke_width=1.2).set_stroke(MUTED).set_fill(cols[i], 0.4).move_to([cx, cy, 0])
            lbl2d[i] = Text(str(i), font_size=13, color=INK).move_to([cx, cy, 0])
        tree2d = VGroup(root, node1, *cells2d.values(), *lbl2d.values(), root_lbl, n1_lbl)

        # --- array (bottom) with a growth cell "..." ---
        aw = 0.78
        acells, albl = {}, {}
        for i in range(9):
            acells[i] = Square(side_length=aw, stroke_width=1.5).set_stroke(MUTED).set_fill(cols[i], 0.4)
            albl[i] = Text(str(i), font_size=FS_CHIP, color=INK)
        grow = Square(side_length=aw, stroke_width=1.2).set_stroke(MUTED, opacity=0.5)
        arow = VGroup(*acells.values(), grow).arrange(RIGHT, buff=0.0).to_edge(DOWN, buff=1.4).shift(RIGHT * 0.7)
        for i in range(9):
            albl[i].move_to(acells[i].get_center())
        grow_txt = Text("…", font_size=FS_BODY, color=MUTED).move_to(grow.get_center())

        # correspondence: colour-matched connectors, top-level cells -> array slots
        conns = VGroup(*[
            DashedLine((node1 if i == 1 else cells2d[i]).get_bottom(), acells[i].get_top(),
                       color=cols[i], stroke_width=1.5, dash_length=0.09)
            for i in [1, 2, 3, 4]
        ])

        def link(a, b, color):
            return CurvedArrow(acells[a].get_bottom() + DOWN * 0.03, acells[b].get_bottom() + DOWN * 0.03,
                               angle=PI / 2.2, color=color, stroke_width=3, tip_length=0.16)
        arr0 = link(0, 1, ACCENT)
        arr1 = link(1, 5, TRAIL)
        arr_lbl = Text("child link = index of first child", font_size=FS_CHIP, color=ACCENT).next_to(arr0, DOWN, buff=0.1)
        brace1 = Brace(VGroup(acells[1], acells[4]), UP, color=MUTED)
        bt = Text("4 siblings, contiguous", font_size=FS_CHIP, color=MUTED).next_to(brace1, UP, buff=0.05)

        notes = VGroup(
            Text("single allocation; index-based ⇒ lock-free", font_size=FS_CHIP, color=INK),
            MathTex(r"\texttt{Option<NonZeroU32>} \Rightarrow 64\text{ B} = 1\text{ cache line}", color=INK).scale(0.5),
            Text("None = leaf", font_size=FS_CHIP, color=INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UR, buff=0.5).shift(DOWN * 0.4)

        self.play(FadeIn(title))
        self.play(Create(root), FadeIn(root_lbl))
        self.play(LaggedStart(FadeIn(cells2d[2]), FadeIn(cells2d[3]), FadeIn(cells2d[4]), FadeIn(node1),
                              FadeIn(cells2d[5]), FadeIn(cells2d[6]), FadeIn(cells2d[7]), FadeIn(cells2d[8]),
                              lag_ratio=0.15),
                  FadeIn(VGroup(*lbl2d.values(), n1_lbl)), run_time=2.0)
        self.play(LaggedStart(*[FadeIn(acells[i]) for i in range(9)], FadeIn(grow), lag_ratio=0.05),
                  FadeIn(VGroup(*albl.values(), grow_txt)), run_time=1.5)
        self.play(LaggedStart(*[Create(c) for c in conns], lag_ratio=0.15), run_time=1.4)
        self.play(Create(arr0), Create(arr1), FadeIn(arr_lbl))
        self.play(GrowFromCenter(brace1), FadeIn(bt))
        self.play(FadeIn(notes, shift=UP * 0.1))
        self.wait(1.8)
        self.play(FadeOut(Group(title, tree2d, arow, VGroup(*albl.values()), grow_txt,
                                conns, arr0, arr1, arr_lbl, brace1, bt, notes)))

    # ------------------------------------------------------------------ #
    def beat_lipschitz(self) -> None:
        """A 1-D field + sliding cone with shaded forbidden regions; then L=4."""
        formula = MathTex(r"\lvert f(\vec a)-f(\vec b)\rvert \;\le\; L\,\lVert \vec a-\vec b\rVert",
                          color=INK).scale(0.8).to_edge(UP, buff=0.55)
        caption = Text("the graph can never enter the shaded cone",
                       font_size=FS_CAPTION, color=MUTED).next_to(formula, DOWN, buff=0.25)

        # --- L = 1 field ---
        pts = [-2.2, -0.3, 1.4, 2.6]

        def f1d(x):
            return min(abs(x - p) for p in pts)

        ax = Axes(x_range=[-3, 3, 1], y_range=[0, 3, 1], x_length=9.0, y_length=2.8, tips=False,
                  axis_config={"stroke_color": MUTED, "stroke_width": 2}).to_edge(DOWN, buff=1.2)
        curve = ax.plot(f1d, color=COOL, stroke_width=3)
        clabel = Text("1-D distance field  (L = 1)", font_size=FS_CAPTION, color=COOL).next_to(ax, UP, buff=0.1).to_edge(LEFT, buff=1.0)
        c = ValueTracker(-2.0)

        def cone_edges():
            cx, cy, r = c.get_value(), f1d(c.get_value()), 1.7
            e1 = Line(ax.c2p(cx - r, cy + r), ax.c2p(cx + r, cy - r), color=ACCENT, stroke_width=2)
            e2 = Line(ax.c2p(cx - r, cy - r), ax.c2p(cx + r, cy + r), color=ACCENT, stroke_width=2)
            return VGroup(e1, e2, Dot(ax.c2p(cx, cy), radius=0.06, color=ACCENT))

        def forbidden():
            cx, cy = c.get_value(), f1d(c.get_value())
            wt, wb = 3.0 - cy, cy - 0.0
            top = Polygon(ax.c2p(cx, cy), ax.c2p(cx - wt, 3.0), ax.c2p(cx + wt, 3.0),
                          color=RED, fill_opacity=0.16, stroke_width=0)
            bot = Polygon(ax.c2p(cx, cy), ax.c2p(cx - wb, 0.0), ax.c2p(cx + wb, 0.0),
                          color=RED, fill_opacity=0.16, stroke_width=0)
            return VGroup(top, bot)

        cone = always_redraw(cone_edges)
        forb = always_redraw(forbidden)

        self.play(Write(formula), FadeIn(caption))
        self.play(Create(ax), Create(curve), FadeIn(clabel))
        self.add(forb, cone)
        self.play(c.animate.set_value(2.0), run_time=4.5, rate_func=linear)
        self.wait(0.4)
        self.play(FadeOut(Group(ax, curve, clabel, cone, forb)))

        # --- L = 4 estimator: mostly smooth, with one steep (4-Lipschitz) spike ---
        note = Text("a real estimator: mostly smooth, but one 4-Lipschitz spike",
                    font_size=FS_CAPTION, color=FIELD_HI).move_to(caption)
        self.play(ReplacementTransform(caption, note))

        x_spike = 0.4

        def base(x):
            return 1.1 + 0.4 * np.sin(1.3 * x + 0.5)

        def f4(x):
            return min(base(x), 4 * abs(x - x_spike))  # a sharp V-notch to 0 at the spike

        ax2 = Axes(x_range=[-3, 3, 1], y_range=[0, 2, 1], x_length=9.0, y_length=2.6, tips=False,
                   axis_config={"stroke_color": MUTED, "stroke_width": 2}).to_edge(DOWN, buff=1.2)
        curve2 = ax2.plot(f4, color="#e6b23a", stroke_width=3)
        clabel2 = Text("distance estimator  (L = 4)", font_size=FS_CAPTION, color="#e6b23a").next_to(ax2, UP, buff=0.1).to_edge(LEFT, buff=1.0)
        c2 = ValueTracker(-2.0)

        def cone4():
            cx, cy, r = c2.get_value(), f4(c2.get_value()), 0.5
            e1 = Line(ax2.c2p(cx - r, cy + 4 * r), ax2.c2p(cx + r, cy - 4 * r), color=ACCENT, stroke_width=2)
            e2 = Line(ax2.c2p(cx - r, cy - 4 * r), ax2.c2p(cx + r, cy + 4 * r), color=ACCENT, stroke_width=2)
            return VGroup(e1, e2, Dot(ax2.c2p(cx, cy), radius=0.06, color=ACCENT))

        def forbidden4():
            cx, cy = c2.get_value(), f4(c2.get_value())
            wt, wb = (2.0 - cy) / 4.0, cy / 4.0
            top = Polygon(ax2.c2p(cx, cy), ax2.c2p(cx - wt, 2.0), ax2.c2p(cx + wt, 2.0),
                          color=RED, fill_opacity=0.16, stroke_width=0)
            bot = Polygon(ax2.c2p(cx, cy), ax2.c2p(cx - wb, 0.0), ax2.c2p(cx + wb, 0.0),
                          color=RED, fill_opacity=0.16, stroke_width=0)
            return VGroup(top, bot)

        cone2 = always_redraw(cone4)
        forb2 = always_redraw(forbidden4)
        self.play(Create(ax2), Create(curve2), FadeIn(clabel2))
        self.add(forb2, cone2)
        self.play(c2.animate.set_value(1.8), run_time=4.0, rate_func=linear)
        self.wait(1.2)
        self.play(FadeOut(Group(formula, note, ax2, curve2, clabel2, cone2, forb2)))

    # ------------------------------------------------------------------ #
    def beat_bnb(self) -> None:
        """sdf_geq_everywhere: branch-and-bound over f-g; proving f>=g prunes f."""
        Bd, LSUM, K = 2.0, 2.0, 6
        # f | g | f-g as three equal panels sharing one mapping (u, row height):
        # identical scale and origin, so features line up across the three plots
        PH = 3.4
        u = PH / (2 * Bd)
        PC = {"f": np.array([-4.55, -0.35, 0.0]),
              "g": np.array([-0.15, -0.35, 0.0]),
              "fg": np.array([4.25, -0.35, 0.0])}

        def to_scene(x, y):
            return PC["fg"] + np.array([x * u, y * u, 0.0])

        ff = F.circle(0.55, 0.35, 0.4)     # f: a small circle, disk(f) c disk(g)
        gg = F.circle(-0.35, -0.2, 1.75)   # g: a large circle enclosing it

        def D(x, y):
            # f - g >= r_g - r_f - |c_f - c_g| = 0.295 > 0 on all of Omega,
            # so f never wins the pointwise min: f is the redundant one
            return ff(x, y) - gg(x, y)

        def classify(cx, cy, size, depth):
            h = (size / 2) * np.sqrt(2)
            delta = float(D(cx, cy))
            if delta < 0:
                return "red"
            if delta >= LSUM * h:
                return "green"
            if depth >= K:
                return "red"
            return "amber"

        fill = {"green": (GREEN, 0.22), "amber": (ACCENT, 0.15), "red": (RED, 0.30)}

        def rect(cx, cy, size, v):
            col, op = fill[v]
            return (Rectangle(width=size * u, height=size * u).move_to(to_scene(cx, cy))
                    .set_stroke(col, 2).set_fill(col, op))

        # one colour scale across f, g and f-g, so brightness is comparable
        # (f-g renders visibly above the zero colour everywhere)
        samples = [F.sample(fn, 340, Bd) for fn in (ff, gg, D)]
        vmin = float(min(s.min() for s in samples))
        vmax = float(max(s.max() for s in samples))
        cscale = {"vmin": vmin, "vmax": vmax}

        # colour-bar legend above the row: a linear field through the same
        # colorizer, so striping and the white zero line match the panels
        CB_W, CB_C = 7.5, np.array([-0.15, 2.6, 0.0])
        cbar_img = ImageMobject(F.colorize(np.tile(np.linspace(vmin, vmax, 680), (20, 1)),
                                           interval=0.3, **cscale))
        cbar_img.width = CB_W
        cbar_img.move_to(CB_C)
        cbar_box = SurroundingRectangle(cbar_img, color=MUTED, buff=0.0).set_stroke(width=1.2)

        def bar_x(v):
            return CB_C[0] - CB_W / 2 + (v - vmin) / (vmax - vmin) * CB_W

        ybot = cbar_img.get_bottom()[1]
        cbar_marks = VGroup()
        for v in range(int(np.ceil(vmin)), int(np.floor(vmax)) + 1):
            cbar_marks.add(Line([bar_x(v), ybot, 0], [bar_x(v), ybot - 0.07, 0]).set_stroke(MUTED, 1.4),
                           MathTex(str(v), color=MUTED, font_size=15)
                           .next_to([bar_x(v), ybot - 0.07, 0], DOWN, buff=0.05))
        cbar_lbl = Text("shared colour scale", font_size=FS_CHIP, color=MUTED).next_to(cbar_box, LEFT, buff=0.3)
        cbar = Group(cbar_img, cbar_box, cbar_marks, cbar_lbl)

        field = F.field_image(D, height=PH, res=340, extent=Bd, interval=0.3, **cscale).move_to(PC["fg"])
        dim = Rectangle(width=PH, height=PH).set_fill(BG, opacity=0.5).set_stroke(width=0).move_to(PC["fg"])
        border = SurroundingRectangle(field, color=MUTED, buff=0.0).set_stroke(width=1.5)
        fg_lbl = MathTex(r"f-g", color=INK, font_size=20).next_to(border, UP, buff=0.14)
        fg_deco = plot_deco(PC["fg"], u)
        ineq = MathTex(r"(f-g)(\vec v)\ge (f-g)(\vec c)-(L_f+L_g)\,h(R)", color=INK).scale(0.6).to_edge(UP, buff=0.3)

        f_img = F.field_image(ff, height=PH, res=340, extent=Bd, interval=0.3, **cscale).move_to(PC["f"])
        f_box = SurroundingRectangle(f_img, color=MUTED, buff=0.0).set_stroke(width=1.5)
        f_lbl = MathTex("f", color=COOL, font_size=20).next_to(f_box, UP, buff=0.14)
        f_panel = Group(f_img, f_box, f_lbl, plot_deco(PC["f"], u))
        g_img = F.field_image(gg, height=PH, res=340, extent=Bd, interval=0.3, **cscale).move_to(PC["g"])
        g_box = SurroundingRectangle(g_img, color=MUTED, buff=0.0).set_stroke(width=1.5)
        g_lbl = MathTex("g", color=COOL, font_size=20).next_to(g_box, UP, buff=0.14)
        g_panel = Group(g_img, g_box, g_lbl, plot_deco(PC["g"], u))

        legend = VGroup(
            Text("green = f ≥ g proven (discard)", font_size=FS_CHIP, color=GREEN),
            Text("amber = undecided (split)", font_size=FS_CHIP, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).to_edge(DOWN, buff=0.45)

        self.play(FadeIn(f_panel), FadeIn(g_panel), FadeIn(ineq), FadeIn(cbar))
        self.play(FadeIn(field), FadeIn(dim), Create(border), FadeIn(fg_deco), FadeIn(fg_lbl), FadeIn(legend))

        # animate the recursion, level by level (f-g > 0 everywhere => all green)
        amber = [(rect(0.0, 0.0, 2 * Bd, "amber"), (0.0, 0.0, 2 * Bd, 0))]
        persist = []
        self.play(FadeIn(amber[0][0]))
        for _ in range(K):
            outs, ins, new_amber = [], [], []
            for mob, (cx, cy, size, depth) in amber:
                outs.append(FadeOut(mob))
                q = size / 4
                for dx, dy in [(-q, q), (q, q), (-q, -q), (q, -q)]:
                    ccx, ccy, csz, cd = cx + dx, cy + dy, size / 2, depth + 1
                    v = classify(ccx, ccy, csz, cd)
                    child = rect(ccx, ccy, csz, v)
                    ins.append(FadeIn(child))
                    (new_amber if v == "amber" else persist).append(
                        (child, (ccx, ccy, csz, cd)) if v == "amber" else child)
            self.play(*outs, *ins, run_time=0.7)
            amber = new_amber
            if not amber:
                break
        self.wait(0.5)

        # f >= g proven everywhere => f is dominated, so prune it: blink and delete f
        proven = Text("f ≥ g proven ⇒ f redundant, pruned", font_size=FS_CHIP, color=FIELD_HI).move_to(f_img)
        self.play(Indicate(f_panel, color=RED, scale_factor=1.12))
        self.play(FadeOut(f_panel))
        self.play(FadeIn(proven))
        self.wait(1.2)

        leftover = [m for m, _ in amber]
        self.play(FadeOut(Group(field, dim, border, fg_lbl, ineq, fg_deco, g_panel,
                                legend, proven, cbar, *persist, *leftover)))

        # the full case rule + verdict
        cases = MathTex(
            r"""\begin{aligned}
            \delta < 0 &\Rightarrow \textbf{false} && (\exists\,\vec v:\ f<g)\\
            \delta \ge (L_f+L_g)\,h(R) &\Rightarrow \text{discard } R && (f\ge g \text{ on } R)\\
            d \ge k &\Rightarrow \textbf{false} && (\text{undecided})\\
            \text{else} &\Rightarrow \text{split into 4 quadrants}
            \end{aligned}""",
            color=INK,
        ).scale(0.7)
        sub = Text("sound: true only when f ≥ g truly holds — else conservative",
                   font_size=FS_CAPTION, color=FIELD_HI).next_to(cases, DOWN, buff=0.5)
        self.play(Write(cases), run_time=2.0)
        self.play(FadeIn(sub))
        self.wait(2.0)
        self.play(FadeOut(Group(cases, sub)))

    # ------------------------------------------------------------------ #
    def beat_insert(self) -> None:
        """insert_primitive_domain, live: a real ADF over wall + 3 circles, a
        new circle f at the field's argmax; every leaf resolves to one of the
        four cases, and one subdividing leaf is magnified to show prune per
        child. All set predicates are computed honestly (numerically)."""
        Bd, BETA, DMAX = 2.0, 2, 3
        PH = 5.0
        u = PH / (2 * Bd)
        DC = np.array([-3.8, -0.15, 0.0])

        def ts(x, y):
            return DC + np.array([x * u, y * u, 0.0])

        # the live field: boundary wall + three circles; f lands at the argmax
        prims = {"w": F.frame(Bd), "c1": F.circle(-1.05, 0.95, 0.62),
                 "c2": F.circle(1.15, 1.05, 0.48), "c3": F.circle(-0.95, -1.15, 0.55)}
        old_keys = list(prims)

        def union(keys):
            return F.union(*[prims[k] for k in keys])

        gx = np.linspace(-Bd, Bd, 801)
        GX, GY = np.meshgrid(gx, gx)
        V = union(old_keys)(GX, GY)
        am = np.unravel_index(int(np.argmax(V)), V.shape)
        x0, y0, d = float(GX[am]), float(GY[am]), float(V[am])
        prims["f"] = F.circle(x0, y0, d)

        def sample_rect(rect, n=48):
            ax, ay, bx, by = rect
            return np.meshgrid(np.linspace(ax, bx, n), np.linspace(ay, by, n))

        def geq_on(fk, gkeys, rect):
            """f_k >= min(gkeys) on rect (numeric stand-in for the sound test)."""
            QX, QY = sample_rect(rect)
            return float(np.min(prims[fk](QX, QY) - union(gkeys)(QX, QY))) >= -1e-9

        def prune(keys, rect):
            return [k for k in keys
                    if len(keys) == 1 or not geq_on(k, [j for j in keys if j != k], rect)]

        def quadrants(rect):
            ax, ay, bx, by = rect
            mx, my = (ax + bx) / 2, (ay + by) / 2
            return [(ax, my, mx, by), (mx, my, bx, by), (ax, ay, mx, my), (mx, ay, bx, my)]

        def build(rect, depth):
            bucket = prune(old_keys, rect)
            if len(bucket) <= BETA or depth == DMAX:
                return [(rect, depth, bucket)]
            return sum((build(q, depth + 1) for q in quadrants(rect)), [])

        leaves = build((-Bd, -Bd, Bd, Bd), 0)

        def classify(rect, depth, bucket):
            if geq_on("f", bucket, rect):
                return "no-op"
            QX, QY = sample_rect(rect)
            if float(np.min(union(bucket)(QX, QY) - prims["f"](QX, QY))) >= -1e-9:
                return "replace"
            if depth == DMAX or len(bucket) < BETA:
                return "append"
            return "subdivide"

        cases = [classify(*leaf) for leaf in leaves]

        CASE = {"no-op": MUTED, "replace": BLUE, "append": GREEN, "subdivide": ACCENT}
        ID = {"w": MUTED, "c1": "#d2a8ff", "c2": TRAIL, "c3": "#f778ba", "f": ACCENT}
        SYM = {"w": "w", "c1": "c_1", "c2": "c_2", "c3": "c_3", "f": "f"}

        # --- domain panel: before/after fields on one shared colour scale ---
        Vb = F.sample(union(old_keys), 340, Bd)
        Va = F.sample(union(old_keys + ["f"]), 340, Bd)
        vmin, vmax = float(min(Vb.min(), Va.min())), float(max(Vb.max(), Va.max()))
        img_b = ImageMobject(F.colorize(Vb, interval=0.3, vmin=vmin, vmax=vmax))
        img_a = ImageMobject(F.colorize(Va, interval=0.3, vmin=vmin, vmax=vmax))
        for img in (img_b, img_a):
            img.height = PH
            img.move_to(DC).set_z_index(0)
        dimr = Rectangle(width=PH, height=PH).set_fill(BG, 0.45).set_stroke(width=0).move_to(DC).set_z_index(1)
        border = SurroundingRectangle(img_b, color=MUTED, buff=0.0).set_stroke(width=1.5).set_z_index(2)
        deco = plot_deco(DC, u, grid=False).set_z_index(2)
        clabels = VGroup(*[MathTex(SYM[k], color=ID[k], font_size=16).move_to(ts(*pos)).set_z_index(4)
                           for k, pos in [("c1", (-1.05, 0.95)), ("c2", (1.15, 1.05)),
                                          ("c3", (-0.95, -1.15)), ("w", (-1.8, -1.8))]])

        def cell_rect(rect):
            ax, ay, bx, by = rect
            return Rectangle(width=(bx - ax) * u, height=(by - ay) * u).move_to(ts((ax + bx) / 2, (ay + by) / 2))

        def digit(n, rect, fs=13):
            ax, ay, bx, by = rect
            return Text(str(n), font_size=fs, color=INK).move_to(ts(ax, by) + np.array([0.16, -0.16, 0.0])).set_z_index(4)

        cells = VGroup(*[cell_rect(r).set_stroke(MUTED, 1.3) for r, _, _ in leaves]).set_z_index(2)
        digits = {i: digit(len(bk), r) for i, (r, _, bk) in enumerate(leaves)}

        title = Text("insert f — every overlapping leaf decides independently, in parallel",
                     font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)
        sub = Text("digits = bucket size |Bₙ|", font_size=FS_CHIP, color=MUTED).next_to(title, DOWN, buff=0.12)

        def rule_row(col, tex):
            return VGroup(Square(0.16).set_stroke(width=0).set_fill(col, 1.0),
                          MathTex(tex, color=INK, font_size=17)).arrange(RIGHT, buff=0.18)
        rules = VGroup(
            rule_row(CASE["no-op"], r"f \ge g_{B_n}\ \text{on } R_n \;\Rightarrow\; \text{no-op}"),
            rule_row(CASE["replace"], r"g_{B_n} \ge f\ \text{on } R_n \;\Rightarrow\; B_n \leftarrow \{f\}"),
            rule_row(CASE["append"], r"\mathrm{depth}(n)=D \,\lor\, \lvert B_n\rvert<\beta \;\Rightarrow\; B_n \leftarrow B_n \cup \{f\}"),
            rule_row(CASE["subdivide"], r"\text{else} \;\Rightarrow\; \text{subdivide},\;\; B_c \leftarrow \mathrm{prune}(B_n \cup \{f\},\, R_c)"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24).to_edge(RIGHT, buff=0.55).shift(UP * 1.85)

        fc = DashedVMobject(Circle(radius=d * u, color=ACCENT, stroke_width=2.5),
                            num_dashes=64).move_to(ts(x0, y0)).set_z_index(5)
        fdot = Dot(ts(x0, y0), radius=0.045, color=ACCENT).set_z_index(5)
        flabel = MathTex("f", color=ACCENT, font_size=20).next_to(fdot, UR, buff=0.05).set_z_index(5)

        # case overlays (no-op leaves dim out; the rest tint in their case colour)
        fills = {}
        for i, (r, _, _) in enumerate(leaves):
            col = CASE[cases[i]]
            fills[i] = (cell_rect(r).set_fill(BG, 0.5).set_stroke(col, 1.0) if cases[i] == "no-op"
                        else cell_rect(r).set_fill(col, 0.18).set_stroke(col, 2.0)).set_z_index(3)
        order = sorted(range(len(leaves)),
                       key=lambda i: np.hypot((leaves[i][0][0] + leaves[i][0][2]) / 2 - x0,
                                              (leaves[i][0][1] + leaves[i][0][3]) / 2 - y0))

        # --- intro ---
        self.play(FadeIn(title), FadeIn(sub), FadeIn(img_b), FadeIn(dimr), Create(border),
                  FadeIn(deco), FadeIn(clabels))
        self.play(LaggedStart(*[Create(c) for c in cells], lag_ratio=0.04, run_time=1.6),
                  FadeIn(VGroup(*digits.values()), run_time=1.2))
        self.play(LaggedStart(*[FadeIn(row, shift=LEFT * 0.15) for row in rules], lag_ratio=0.15), run_time=1.5)
        self.wait(0.4)
        self.play(Create(fc), FadeIn(fdot), FadeIn(flabel), run_time=1.2)
        self.wait(0.5)

        # --- the read-only pass: every leaf classifies, rippling out from f ---
        self.play(LaggedStart(*[FadeIn(fills[i]) for i in order], lag_ratio=0.05), run_time=2.4)
        self.wait(0.8)

        # --- structural edits: subdividing leaves split; children get pruned buckets ---
        child_data, split_anims, split_new = {}, [], []
        for i, (r, _, bk) in enumerate(leaves):
            if cases[i] != "subdivide":
                continue
            ax, ay, bx, by = r
            mx, my = (ax + bx) / 2, (ay + by) / 2
            cross = VGroup(Line(ts(mx, ay), ts(mx, by)),
                           Line(ts(ax, my), ts(bx, my))).set_stroke(ACCENT, 1.4).set_z_index(3)
            kids = [(q, prune(bk + ["f"], q)) for q in quadrants(r)]
            child_data[i] = kids
            kd = VGroup(*[digit(len(kept), q, fs=11) for q, kept in kids])
            split_anims.append(AnimationGroup(Create(cross), FadeOut(digits.pop(i)), FadeIn(kd)))
            split_new.append(cross)
        self.play(LaggedStart(*split_anims, lag_ratio=0.12), run_time=1.8)

        # --- magnify one subdividing leaf: prune(B ∪ {f}, R_c) per child ---
        zi = next(i for i, (r, _, _) in enumerate(leaves) if np.allclose(r, (1.0, 0.0, 2.0, 1.0)))
        zrect, _, zbucket = leaves[zi]
        members = zbucket + ["f"]
        hl = cell_rect(zrect).set_stroke(INK, 2.5).set_z_index(6)
        P0 = np.array([1.75, -1.3, 0.0])
        parent_sq = Square(1.05).set_stroke(ACCENT, 1.8).move_to(P0)

        def chipcol(keys):
            return VGroup(*[MathTex(SYM[k], color=ID[k], font_size=16) for k in keys]).arrange(DOWN, buff=0.09)

        parent_chips = chipcol(members).move_to(P0)
        parent_lbl = MathTex(r"B_n \cup \{f\}", color=MUTED, font_size=16).next_to(parent_sq, DOWN, buff=0.12)
        arrow = Arrow(parent_sq.get_right(), parent_sq.get_right() + RIGHT * 1.0, buff=0.08,
                      color=MUTED, stroke_width=2.5, max_tip_length_to_length_ratio=0.22)
        C0 = np.array([4.55, -1.3, 0.0])
        cell_pos = [C0 + 0.56 * np.array([sx, sy, 0.0]) for sx, sy in [(-1, 1), (1, 1), (-1, -1), (1, -1)]]
        child_sqs = VGroup(*[Square(1.05).set_stroke(ACCENT, 1.2).move_to(p) for p in cell_pos])
        child_chips, strikes, drops = VGroup(), VGroup(), VGroup()
        for (q, kept), p in zip(child_data[zi], cell_pos):
            col = chipcol(members).move_to(p)
            for k, chip in zip(members, col):
                if k not in kept:
                    strikes.add(Line(chip.get_left() + 0.05 * LEFT, chip.get_right() + 0.05 * RIGHT,
                                     color=RED, stroke_width=2.2))
                    drops.add(chip)
            child_chips.add(col)
        child_lbl = MathTex(r"B_c = \mathrm{prune}(B_n \cup \{f\},\, R_c)",
                            color=MUTED, font_size=16).next_to(child_sqs, DOWN, buff=0.14)
        conn = DashedLine(hl.get_corner(DR), parent_sq.get_corner(UL), color=MUTED,
                          stroke_width=1.5, dash_length=0.09)
        prune_eq = MathTex(r"\mathrm{prune}(B,R)=\{\,(f_i,L_i)\in B \;:\; \lnot\,(f_i \ge \min_{j\ne i} f_j\ \text{on } R)\,\}",
                           color=INK).scale(0.6).to_edge(DOWN, buff=0.35).shift(RIGHT * 2.2)

        self.play(Create(hl), Create(conn), Create(parent_sq), FadeIn(parent_chips), FadeIn(parent_lbl))
        self.play(GrowArrow(arrow), Create(child_sqs), FadeIn(child_chips), FadeIn(child_lbl))
        self.play(Write(prune_eq), run_time=1.4)
        self.play(LaggedStart(*[Create(s) for s in strikes], lag_ratio=0.08), run_time=1.0)
        self.play(drops.animate.set_opacity(0.22), strikes.animate.set_opacity(0.4), run_time=0.8)
        self.wait(1.2)

        # --- commit: field updates to min(g, f); tints lift, tree stays refined ---
        commit = [FadeOut(img_b), FadeIn(img_a), FadeOut(fc)]
        for i, (r, _, bk) in enumerate(leaves):
            if cases[i] == "replace":
                nd = digit(1, r)
            elif cases[i] == "append":
                nd = digit(len(bk) + 1, r)
            else:
                continue
            commit += [FadeOut(digits.pop(i)), FadeIn(nd)]
            digits[i] = nd
        commit += [FadeOut(fills[i]) for i in fills]
        commit += [cross.animate.set_stroke(MUTED, 1.3) for cross in split_new]
        sound = Text("every ≥ on R runs the sound predicate — a primitive is dropped only\n"
                     "when provably redundant; the stored field never deviates from the true min",
                     font_size=FS_CHIP, color=FIELD_HI, line_spacing=0.9
                     ).to_edge(DOWN, buff=0.25).set_x(0.5)
        self.play(*commit, run_time=1.4)
        self.play(FadeOut(prune_eq), FadeIn(sound))
        self.wait(1.8)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])

    # ------------------------------------------------------------------ #
    def beat_domains(self) -> None:
        """D* insertion domains: global (4d square, optimal) vs local (unbounded)."""
        title = Text("where can an insertion change the field?", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.3)
        dstar = MathTex(r"D^{*}=\{\, \vec v : g(\vec v) > \lVert \vec v-\vec x_0\rVert - d\,\}", color=INK).scale(0.6)
        dstar.next_to(title, DOWN, buff=0.18)
        self.play(FadeIn(title), Write(dstar))

        B, Hd = 2.0, 4.4
        sc = Hd / (2 * B)
        Cc = np.array([-2.3, -0.5, 0.0])

        def ts(x, y):
            return Cc + np.array([x * sc, y * sc, 0.0])

        def dirp(deg):
            return np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg)), 0.0])

        # ===== GLOBAL: field before -> after; change stays within B(x0, 2d) =====
        ring = [F.circle(1.25 * np.cos(a), 1.25 * np.sin(a), 0.45) for a in np.deg2rad([18, 90, 162, 234, 306])]
        beforeG = F.union(F.frame(B), *ring)
        dG = float(beforeG(0.0, 0.0))
        afterG = F.union(beforeG, F.circle(0.0, 0.0, dG))
        fbG = F.field_image(beforeG, height=Hd, res=340, extent=B).move_to(Cc).set_z_index(0)
        faG = F.field_image(afterG, height=Hd, res=340, extent=B).move_to(Cc).set_z_index(0)
        dimG = Rectangle(width=Hd, height=Hd).set_fill(BG, 0.4).set_stroke(width=0).move_to(Cc).set_z_index(1)
        borderG = SurroundingRectangle(fbG, color=MUTED, buff=0.0).set_stroke(width=1.2).set_z_index(2)
        x0 = Dot(ts(0, 0), radius=0.05, color=INK).set_z_index(3)
        free = Circle(radius=dG * sc, color=FIELD_HI, stroke_width=2.5).move_to(ts(0, 0)).set_z_index(3)
        b2d = DashedVMobject(Circle(radius=2 * dG * sc, color=COOL, stroke_width=2).move_to(ts(0, 0)), num_dashes=44).set_z_index(3)
        square = Square(side_length=4 * dG * sc, color=ACCENT, stroke_width=2).move_to(ts(0, 0)).set_z_index(3)
        tangent = Circle(radius=dG * sc, color=MUTED, stroke_width=2).move_to(ts(2 * dG, 0)).set_fill(MUTED, 0.06).set_z_index(3)
        gl_lbls = VGroup(
            Text("global max", font_size=FS_CAPTION, color=INK),
            Text("cover: 4d square", font_size=FS_CHIP, color=ACCENT),
            Text("(4 optimal — two tangent balls)", font_size=FS_CHIP, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_edge(RIGHT, buff=0.5).shift(UP * 1.2)
        chg = Text("insert at x₀: field changes\nonly within D* ⊆ B(x₀, 2d)",
                   font_size=FS_CHIP, color=COOL, line_spacing=0.8).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.9)

        self.play(FadeIn(fbG), FadeIn(dimG), Create(borderG))
        self.play(FadeIn(x0), Create(free))
        self.play(Create(b2d), Create(square), Create(tangent))
        self.play(FadeIn(gl_lbls))
        self.wait(0.4)
        self.play(FadeOut(fbG), FadeIn(faG), Indicate(free, color=INK), FadeIn(chg))
        self.wait(1.4)
        self.play(FadeOut(Group(fbG, faG, dimG, borderG, x0, free, b2d, square, tangent, gl_lbls, chg)))

        # ===== LOCAL: field with 3 contacts; before -> after change streaks out =====
        loc = [F.circle(1.15 * np.cos(a), 1.15 * np.sin(a), 0.5) for a in np.deg2rad([90, 210, 330])]
        beforeL = F.union(F.frame(B), *loc)
        dL = float(beforeL(0.0, 0.0))
        afterL = F.union(beforeL, F.circle(0.0, 0.0, dL))
        fbL = F.field_image(beforeL, height=Hd, res=340, extent=B).move_to(Cc).set_z_index(0)
        faL = F.field_image(afterL, height=Hd, res=340, extent=B).move_to(Cc).set_z_index(0)
        dimL = Rectangle(width=Hd, height=Hd).set_fill(BG, 0.4).set_stroke(width=0).move_to(Cc).set_z_index(1)
        borderL = SurroundingRectangle(fbL, color=MUTED, buff=0.0).set_stroke(width=1.2).set_z_index(2)
        x0L = Dot(ts(0, 0), radius=0.05, color=INK).set_z_index(3)
        freeL = Circle(radius=dL * sc, color=FIELD_HI, stroke_width=2.5).move_to(ts(0, 0)).set_z_index(3)
        ray = Arrow(ts(0, 0), ts(0, 0) + 2.4 * sc * dirp(30), buff=dL * sc, color=TRAIL,
                    stroke_width=3, max_tip_length_to_length_ratio=0.09).set_z_index(3)
        streak = Polygon(ts(0, 0) + dL * sc * dirp(16), Cc + 2.6 * sc * dirp(22),
                         Cc + 2.6 * sc * dirp(40), ts(0, 0) + dL * sc * dirp(46),
                         color=TRAIL).set_fill(TRAIL, 0.20).set_stroke(width=0).set_z_index(2)
        lr_lbls = VGroup(
            Text("local max", font_size=FS_CAPTION, color=INK),
            Text("D* unbounded —", font_size=FS_CHIP, color=TRAIL),
            Text("no c·d square works", font_size=FS_CHIP, color=TRAIL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_edge(RIGHT, buff=0.5).shift(UP * 1.1)
        chgL = Text("insert at x₀: the change\nstreaks out along D*",
                    font_size=FS_CHIP, color=COOL, line_spacing=0.8).to_edge(RIGHT, buff=0.5).shift(DOWN * 0.9)
        walk = MathTex(r"\hat g(\vec c_R)+L_B\,h(R)\;\le\;\mathrm{dist}(R,\vec x_0)-d", color=FIELD_HI).scale(0.56)
        wcap = Text("insert_at_maximum prunes the tree walk, not a fixed box", font_size=FS_CHIP, color=MUTED)
        VGroup(wcap, walk).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.35)

        self.play(FadeIn(fbL), FadeIn(dimL), Create(borderL))
        self.play(FadeIn(x0L), Create(freeL))
        self.play(FadeIn(streak), GrowArrow(ray), FadeIn(lr_lbls))
        self.wait(0.3)
        self.play(FadeOut(fbL), FadeIn(faL), Indicate(streak, color=TRAIL), FadeIn(chgL))
        self.wait(0.6)
        self.play(FadeIn(wcap), Write(walk))
        self.wait(1.8)
        self.play(FadeOut(Group(
            title, dstar, fbL, faL, dimL, borderL, x0L, freeL, ray, streak, lr_lbls, walk, wcap, chgL,
        )))
