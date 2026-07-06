"""Scene 6 — the optimizer.

Three beats: (1) GD-ADF banner — lossless, 10-100x smaller, differentiable,
so an iterative optimizer returns; (2) LineSearch::optimize on the
optimize_precision test field — raw-gradient zigzag vs momentum along the
medial-axis ridge, with the h growth/decay history; (3) find_max_parallel —
a parallel batch of ascents, sequential free-ball dedup, lock-free commit.

Render:
    uv run manim -ql scene06_optimizer.py Scene06Optimizer
"""

import numpy as np
from manim import *

import fields as F
from theme import (
    VideoScene, plot_deco, INK, MUTED, ACCENT, COOL, BG, TRAIL, FIELD_HI,
    FS_TITLE, FS_H2, FS_BODY, FS_CAPTION, FS_CHIP, mono_span, rich_text,
)

GREEN = "#3fb950"
RED = "#ff6b6b"


def ascend_ls(fn, start, momentum=True, h0=0.5, growth=1.25, decay=0.35,
              tol=5e-3, eps=1e-6, limit=400):
    """Mirror of ``LineSearch::optimize``: forward-difference gradient,
    dir = normalize(unit gradient + momentum), momentum RESET on rejection,
    h capped at h0, growth on accept / decay on reject, stop once h < tol.
    Returns (endpoint, steps, evals); steps = (from, trial, accepted, h_used).
    """
    p = np.array(start, float)
    fp = float(fn(p[0], p[1]))
    mom = np.zeros(2)
    h = h0
    evals = 1
    steps = []
    for _ in range(limit):
        if h < tol:
            break
        g = np.array([fn(p[0] + eps, p[1]) - fp, fn(p[0], p[1] + eps) - fp]) / eps
        evals += 2
        n = float(np.hypot(*g))
        if not n > 0:
            break
        d = g / n + (mom if momentum else 0)
        dn = float(np.hypot(*d))
        d = d / dn if dn > 0 else g / n
        trial = p + h * d
        fc = float(fn(trial[0], trial[1]))
        evals += 1
        if fc > fp:
            steps.append((p.copy(), trial.copy(), True, h))
            p, fp = trial, fc
            mom = d
            h = min(h * growth, h0)
        else:
            steps.append((p.copy(), trial.copy(), False, h))
            mom = np.zeros(2)
            h *= decay
    return p, steps, evals


class Scene06Optimizer(VideoScene):
    def construct(self) -> None:
        self.beat_intro()
        self.beat_ascent()
        self.beat_batch()

    # ------------------------------------------------------------------ #
    def beat_intro(self) -> None:
        """GD-ADF banner: why an iterative optimizer is back on the table."""
        title = Text("GD-ADF — reintroducing the optimizer", font_size=FS_TITLE, color=INK)
        chips = VGroup(
            Text("lossless — the field is exact, not sampled", font_size=FS_BODY, color=MUTED),
            rich_text("10–100× less memory than the bitmap", font_size=FS_BODY, color=MUTED),
            rich_text("continuous field ⇒ gradient ascent works", font_size=FS_BODY, color=MUTED),
            rich_text("⇒ GD-ADF is a local-maximum method", font_size=FS_BODY, color=FIELD_HI),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        VGroup(title, chips).arrange(DOWN, buff=0.7).move_to(0.2 * UP)

        self.play(FadeIn(title, shift=UP * 0.1))
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in chips], lag_ratio=0.25), run_time=2.2)
        self.wait(1.8)
        self.play(FadeOut(Group(title, chips)))

    # ------------------------------------------------------------------ #
    def beat_ascent(self) -> None:
        """LineSearch::optimize on the optimize_precision test field (mapped to
        [-2,2]^2): raw gradient zigzags across the medial axis; momentum runs
        along it. The h-history charts show growth on accept, decay on reject.
        """
        Bd = 2.0
        PH = 5.0
        u = PH / (2 * Bd)
        DC = np.array([-3.8, -0.15, 0.0])

        def ts(x, y):
            return DC + np.array([x * u, y * u, 0.0])

        # the optimize_precision test, x4: two point obstacles + the walls;
        # the kink apex (0,-0.64) is witnessed by both points and the wall
        fn = F.union(F.frame(Bd), F.circle(-1.2, 0.0, 0.0), F.circle(1.2, 0.0, 0.0))
        seed = (-0.36, -1.1)
        apex = np.array([0.0, -0.64])
        H0, TOL = 0.5, 5e-3
        runs = {m: ascend_ls(fn, seed, momentum=m, h0=H0, tol=TOL) for m in (False, True)}

        # --- panel ---
        img = F.field_image(fn, height=PH, res=340, extent=Bd, interval=0.3).move_to(DC).set_z_index(0)
        dim = Rectangle(width=PH, height=PH).set_fill(BG, 0.3).set_stroke(width=0).move_to(DC).set_z_index(1)
        border = SurroundingRectangle(img, color=MUTED, buff=0.0).set_stroke(width=1.5).set_z_index(2)
        deco = plot_deco(DC, u, grid=False).set_z_index(2)
        title = Text("adaptive gradient ascent — momentum along the medial-axis ridge",
                     font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)
        chip = self.source_chip("src/solver/line_search.rs — LineSearch::optimize")

        d_eq = MathTex(r"\vec d_k \;=\; \mathrm{normalize}\!\left(\frac{\nabla g(\vec p_k)}{\lVert \nabla g(\vec p_k)\rVert} + \vec d_{k-1}\right)",
                       color=INK).scale(0.62)
        p_eq = MathTex(r"\vec p_{k+1} = \begin{cases}\vec p_k + h_k\,\vec d_k & \text{if } g \text{ improves}\\"
                       r"\vec p_k & \text{otherwise}\end{cases}"
                       r"\quad h \leftarrow \begin{cases} h\cdot\text{growth}\\ h\cdot\text{decay}\end{cases}",
                       color=INK).scale(0.52)
        VGroup(d_eq, p_eq).arrange(DOWN, buff=0.3).to_edge(RIGHT, buff=0.55).shift(UP * 2.15)

        # --- h-history charts (log scale, baseline = tolerance) ---
        CH_W, CH_H = 3.4, 0.72
        n_bars = max(len(runs[False][1]), len(runs[True][1]))

        def chart(origin, label, color):
            base = Line(origin, origin + RIGHT * CH_W).set_stroke(MUTED, 1.4)
            tol_lbl = MathTex(r"\Delta", color=MUTED, font_size=15).next_to(base.get_start(), DOWN, buff=0.08)
            h_lbl = MathTex("h", color=MUTED, font_size=16).next_to(origin + UP * CH_H, LEFT, buff=0.12)
            name = Text(label, font_size=FS_CHIP, color=color).next_to(base.get_start() + UP * CH_H, RIGHT, buff=0.0).shift(UP * 0.2)
            return VGroup(base, tol_lbl, h_lbl, name)

        def bar(origin, k, h, accepted):
            frac = max(np.log(h / TOL), 0.0) / np.log(H0 / TOL)
            w = CH_W / n_bars
            r = Rectangle(width=w * 0.7, height=max(frac * CH_H, 0.02))
            r.set_stroke(width=0).set_fill(GREEN if accepted else RED, 0.9)
            return r.move_to(origin + RIGHT * (k + 0.5) * w, aligned_edge=DOWN)

        ch_raw_o = np.array([2.2, 0.35, 0.0])
        ch_mom_o = np.array([2.2, -1.35, 0.0])
        ch_raw = chart(ch_raw_o, "raw  (no momentum)", RED)
        ch_mom = chart(ch_mom_o, "momentum", TRAIL)

        # caption lane sits above the source chip's line (the mono chip is wide)
        cap = Text("a trial step is kept only if g improves — the iterate is monotone",
                   font_size=FS_CHIP, color=MUTED).to_edge(DOWN, buff=0.78).set_x(2.2)

        self.play(FadeIn(title), FadeIn(img), FadeIn(dim), Create(border), FadeIn(deco),
                  FadeIn(chip), FadeIn(d_eq), FadeIn(p_eq), FadeIn(cap))

        # --- one ascent, step by step, feeding its h-chart ---
        def play_ascent(momentum, color, ch_origin, rt=0.24):
            _, steps, _ = runs[momentum]
            dot = Dot(ts(*steps[0][0]), radius=0.055, color=color).set_z_index(6)
            self.play(FadeIn(dot, scale=2.5), run_time=0.4)
            trail = []
            for k, (p0, trial, acc, h) in enumerate(steps):
                b = bar(ch_origin, k, h, acc)
                if acc:
                    seg = Line(ts(*p0), ts(*trial), color=color, stroke_width=3.0).set_z_index(4)
                    self.play(Create(seg), dot.animate.move_to(ts(*trial)), FadeIn(b), run_time=rt)
                else:
                    seg = DashedLine(ts(*p0), ts(*trial), dash_length=0.05, color=RED,
                                     stroke_width=1.8).set_stroke(opacity=0.4).set_z_index(3)
                    self.play(Create(seg), FadeIn(b), run_time=rt * 0.8)
                trail.append(seg)
            return dot, trail

        self.play(FadeIn(ch_raw))
        raw_dot, raw_trail = play_ascent(False, RED, ch_raw_o)
        raw_note = Text("the raw gradient alternates sides of the ridge",
                        font_size=FS_CHIP, color=RED).to_edge(DOWN, buff=0.78).set_x(2.2)
        self.play(ReplacementTransform(cap, raw_note))
        self.wait(0.8)

        self.play(FadeIn(ch_mom), *[m.animate.set_stroke(opacity=0.25) for m in raw_trail],
                  raw_dot.animate.set_opacity(0.45))
        mom_dot, mom_trail = play_ascent(True, TRAIL, ch_mom_o, rt=0.28)
        mom_note = Text("momentum cancels the across-ridge component — travel runs along the ridge",
                        font_size=FS_CHIP, color=TRAIL).to_edge(DOWN, buff=0.78).set_x(2.2)
        self.play(ReplacementTransform(raw_note, mom_note))
        self.wait(0.6)

        # --- verdict: evals + distance to the kink apex, straight from the sim ---
        apex_dot = Dot(ts(*apex), radius=0.045, color=FIELD_HI).set_z_index(6)
        apex_lbl = Text("kink maximum", font_size=FS_CHIP, color=FIELD_HI
                        ).next_to(apex_dot, DOWN, buff=0.1).set_z_index(6)

        def verdict(momentum, color):
            p_end, _, ev = runs[momentum]
            err = float(np.hypot(*(p_end - apex)))
            return Text(f"{ev} field evals — ends {err:.0e} from the apex",
                        font_size=FS_CHIP, color=color)
        v_raw = verdict(False, RED).next_to(ch_raw[0], DOWN, buff=0.32)
        v_mom = verdict(True, TRAIL).next_to(ch_mom[0], DOWN, buff=0.32)
        end_note = rich_text("h < Δ stops the ascent — the kink is refined bisection-style",
                             font_size=FS_CHIP, color=FIELD_HI).to_edge(DOWN, buff=0.78).set_x(2.2)

        self.play(FadeIn(apex_dot, scale=2.0), FadeIn(apex_lbl), FadeIn(v_raw), FadeIn(v_mom))
        self.play(ReplacementTransform(mom_note, end_note))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])

    # ------------------------------------------------------------------ #
    def beat_batch(self) -> None:
        """find_max_parallel: a batch of seeds ascends one field snapshot in
        parallel; a sequential sweep keeps a maximum only if its free ball is
        disjoint from every kept one; survivors commit together, lock-free."""
        Bd = 2.0
        PH = 5.0
        u = PH / (2 * Bd)
        DC = np.array([-3.8, -0.15, 0.0])

        def ts(x, y):
            return DC + np.array([x * u, y * u, 0.0])

        circles = [(-1.1, 0.95, 0.5), (1.05, 1.1, 0.45), (-1.0, -1.0, 0.42),
                   (1.1, -0.85, 0.48), (0.05, 0.1, 0.38)]
        fb = F.union(F.frame(Bd), *[F.circle(*c) for c in circles])
        seeds = [(-1.7, 1.55), (0.1, 1.55), (1.7, 1.6), (-1.65, -0.2), (1.75, 0.3),
                 (-1.6, -1.7), (0.2, -1.6), (1.55, -1.7), (-0.4, 0.9), (-0.25, -1.45)]

        runs = [ascend_ls(fb, s, momentum=True, tol=1e-3) for s in seeds]
        maxima = [(p, float(fb(p[0], p[1]))) for p, _, _ in runs]
        kept_flags, kept = [], []
        for m, d in maxima:  # batch order, as in find_max_parallel
            ok = all(float(np.hypot(*(m - mk))) > d + dk for mk, dk in kept)
            kept_flags.append(ok)
            if ok:
                kept.append((m, d))

        # --- panels: before/after on one shared colour scale ---
        after = F.union(fb, *[F.circle(m[0], m[1], d) for m, d in kept])
        Vb = F.sample(fb, 340, Bd)
        Va = F.sample(after, 340, Bd)
        cscale = {"vmin": float(min(Vb.min(), Va.min())), "vmax": float(max(Vb.max(), Va.max()))}
        img_b = ImageMobject(F.colorize(Vb, interval=0.3, **cscale))
        img_a = ImageMobject(F.colorize(Va, interval=0.3, **cscale))
        for img in (img_b, img_a):
            img.height = PH
            img.move_to(DC).set_z_index(0)
        dim = Rectangle(width=PH, height=PH).set_fill(BG, 0.3).set_stroke(width=0).move_to(DC).set_z_index(1)
        border = SurroundingRectangle(img_b, color=MUTED, buff=0.0).set_stroke(width=1.5).set_z_index(2)
        deco = plot_deco(DC, u, grid=False).set_z_index(2)
        title = MarkupText(
            f'{mono_span("find_max_parallel")} — one snapshot, a whole batch of ascents',
            font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)
        chip = self.source_chip("src/util.rs")  # the fn name is already in the title
        rule = MathTex(r"\text{accept } \vec m_j \iff \lVert \vec m_i - \vec m_j\rVert > d_i + d_j"
                       r"\quad \forall\ \text{accepted } i", color=INK).scale(0.6)
        rule.to_edge(RIGHT, buff=0.55).shift(UP * 2.3)

        seed_dots = VGroup(*[Dot(ts(*s), radius=0.05, color=TRAIL).set_z_index(6) for s in seeds])
        paths = VGroup()
        for (s, (p_end, steps, _)) in zip(seeds, runs):
            pts = [ts(*s)] + [ts(*t) for _, t, acc, _ in steps if acc]
            vm = VMobject().set_points_as_corners(pts if len(pts) > 1 else pts * 2)
            paths.add(vm.set_stroke(TRAIL, 2.0, opacity=0.85).set_z_index(4))
        balls = VGroup(*[Circle(radius=d * u, color=FIELD_HI, stroke_width=2.2)
                         .move_to(ts(*m)).set_z_index(5) for m, d in maxima])

        cap1 = Text("read-only ascents on one snapshot — embarrassingly parallel",
                    font_size=FS_CHIP, color=MUTED).to_edge(DOWN, buff=0.35).set_x(0.6)
        self.play(FadeIn(title), FadeIn(img_b), FadeIn(dim), Create(border), FadeIn(deco), FadeIn(chip))
        self.play(LaggedStart(*[FadeIn(d, scale=2.0) for d in seed_dots], lag_ratio=0.06), run_time=1.0)
        self.play(FadeIn(cap1))
        self.play(*[Create(p) for p in paths],
                  *[MoveAlongPath(d, p) for d, p in zip(seed_dots, paths)],
                  run_time=2.8, rate_func=linear)
        self.play(LaggedStart(*[Create(b) for b in balls], lag_ratio=0.06), run_time=1.2)
        self.wait(0.4)

        # --- sequential dedup sweep, in batch order ---
        cap2 = Text("sequential sweep: keep a maximum only if its free ball is disjoint",
                    font_size=FS_CHIP, color=MUTED).to_edge(DOWN, buff=0.35).set_x(0.6)
        self.play(FadeIn(rule), ReplacementTransform(cap1, cap2))
        for j, ok in enumerate(kept_flags):
            if ok:
                self.play(balls[j].animate.set_stroke(GREEN), Flash(balls[j], color=GREEN, line_length=0.12),
                          run_time=0.45)
            else:
                self.play(balls[j].animate.set_stroke(RED), Flash(balls[j], color=RED, line_length=0.12),
                          run_time=0.45)
                self.play(FadeOut(balls[j]), FadeOut(seed_dots[j]),
                          paths[j].animate.set_stroke(opacity=0.15), run_time=0.5)
        self.wait(0.5)

        # --- commit: survivors inserted together; next batch reads afresh ---
        cap3 = rich_text("disjoint free balls ⇒ the shapes cannot overlap, in any insertion order —\n"
                    "the batch commits lock-free; the next batch reads the field afresh",
                    font_size=FS_CHIP, color=FIELD_HI, line_spacing=0.9).to_edge(DOWN, buff=0.3).set_x(0.6)
        live = [b for b, ok in zip(balls, kept_flags) if ok]
        self.play(FadeOut(img_b), FadeIn(img_a),
                  *[b.animate.set_stroke(INK, opacity=0.9) for b in live],
                  *[p.animate.set_stroke(opacity=0.2) for p in paths],
                  ReplacementTransform(cap2, cap3), run_time=1.4)
        sub = Text("the tree update has the same shape:\na parallel read-only pass, then brief sequential edits",
                   font_size=FS_CHIP, color=MUTED, line_spacing=0.9).next_to(rule, DOWN, buff=0.5).set_x(3.8)
        self.play(FadeIn(sub))
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])
