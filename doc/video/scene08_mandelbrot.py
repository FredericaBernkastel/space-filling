"""Scene 8 — a Mandelbrot distance estimator, 20k times.

Beats: (1) the single Mandelbrot DE field with a grid of gradient arrows —
unit-ish gradients far away, unbounded spikes near the boundary filaments, and
a flat (clamped-to-0) interior; the declared L = 4 propagates through every
combinator. (2) the real `06_custom_primitive` run (20k instances) embedded as
footage with a live counter, closing on the quoted figures: 7 s, 4.74 MiB, no
obvious errors.

Render:
    uv run manim -ql scene08_mandelbrot.py Scene08Mandelbrot
"""

import numpy as np
from manim import *

import fields as F
from theme import (
    VideoScene, asset, INK, MUTED, ACCENT, COOL, TRAIL, FIELD_HI, BG,
    FS_TITLE, FS_H2, FS_BODY, FS_CAPTION, FS_CHIP,
)
from video import load_frames, VideoMobject

RED = "#ff6b6b"
GREEN = "#3fb950"

# mirror of the emission schedule in render/src/bin/custom_primitive.rs: frame 0
# is the seed shape alone, then one frame per geometric threshold — thresholds
# below the next integer collapse, so the exact frame -> count table is rebuilt
# here with the same loop the binary runs
COUNT, FRAMES, FIRST = 20_000, 300, 1.0
RATIO = (COUNT / FIRST) ** (1.0 / FRAMES)


def frame_counts() -> list[int]:
    table, nxt = [0], FIRST
    for drawn in range(1, COUNT + 1):
        if drawn >= nxt:
            table.append(drawn)
            while nxt <= drawn:
                nxt *= RATIO
    table.append(COUNT)  # the binary's explicit final frame
    return table

# window over the c-plane (matches MandlelDE's bounding box, roughly centred)
CX, CY, EXT = -0.7, 0.0, 1.5


def mandel_de(X, Y, iters=256):
    """Vectorized Hubbard-Douady estimator — verbatim maths of the example's
    ``MandlelDE``: d = |z|·ln|z| / |z'|, interior clamped to 0."""
    C = np.asarray(X, dtype=float) + 1j * np.asarray(Y, dtype=float)
    Z = np.zeros_like(C)
    DZ = np.ones_like(C)
    alive = np.ones(C.shape, dtype=bool)  # not escaped yet
    for _ in range(iters):
        Zn = Z * Z + C
        DZn = 2.0 * Z * DZ + 1.0
        Z = np.where(alive, Zn, Z)
        DZ = np.where(alive, DZn, DZ)
        alive &= np.abs(Z) ** 2 <= 1e9
    r = np.abs(Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = r * np.log(r) / np.abs(DZ)
    return np.where(alive, 0.0, np.nan_to_num(d))


class Scene08Mandelbrot(VideoScene):
    def construct(self) -> None:
        self.beat_estimator()
        self.beat_fill()

    # ------------------------------------------------------------------ #
    def beat_estimator(self) -> None:
        """The single DE field + gradient arrows; the declared L = 4."""
        title = Text("a Mandelbrot distance estimator", font_size=FS_H2, color=MUTED).to_edge(UP, buff=0.35)

        H = 5.9
        s = H / (2 * EXT)
        G = np.array([-2.9, -0.25, 0.0])

        def to_scene(x, y):
            return G + np.array([(x - CX) * s, (y - CY) * s, 0.0])

        # interior (d = 0) is nudged slightly negative for display only, so the
        # colorizer reads it as "inside" and the white line hugs the boundary
        def display(X, Y):
            d = mandel_de(X, Y)
            return np.where(d <= 0.0, -0.045, d)

        field = F.field_image(display, height=H, res=460, extent=EXT, center=(CX, CY),
                              interval=0.1).move_to(G)
        border = SurroundingRectangle(field, color=MUTED, buff=0.0).set_stroke(width=1.5)

        self.play(FadeIn(title))
        self.play(FadeIn(field), Create(border), run_time=1.2)

        # gradient arrows on a coarse grid (central differences, eps in c-units)
        eps = 1e-3
        n = 19
        # the far-field estimator gradient drifts to ~1.9 at the window corners;
        # only the filament probes (picked below) should read as "hot", so the
        # blanket grid threshold sits above that drift
        HOT = 2.0
        PROBE_MIN = 1.5   # floor for the filament probes (band max ~1.96)
        PROBE_BAND = 0.08  # probes only within a hair of the set, i.e. on filaments
        margin = EXT / n
        gx = np.linspace(CX - EXT + margin, CX + EXT - margin, n)
        gy = np.linspace(CY - EXT + margin, CY + EXT - margin, n)
        GX, GY = np.meshgrid(gx, gy)
        VX = (mandel_de(GX + eps, GY) - mandel_de(GX - eps, GY)) / (2 * eps)
        VY = (mandel_de(GX, GY + eps) - mandel_de(GX, GY - eps)) / (2 * eps)
        MAG = np.hypot(VX, VY)

        def arrow(px, py, vx, vy, m, hot):
            ux, uy = vx / m, vy / m
            ln = float(np.clip(0.14 * m, 0.08, 0.34))
            return Arrow(to_scene(px, py), to_scene(px, py) + ln * np.array([ux, uy, 0.0]),
                         buff=0, stroke_width=1.8, max_tip_length_to_length_ratio=0.45,
                         color=RED if hot else TRAIL)

        arrows_unit = VGroup()   # |∇| ≈ 1 — the true-SDF-like far field
        arrows_hot = VGroup()    # |∇| ≫ 1 — hugging the boundary filaments
        interior_dots = VGroup()
        for i in range(n):
            for j in range(n):
                m = float(MAG[j, i])
                p = to_scene(gx[i], gy[j])
                if m < 0.05:
                    interior_dots.add(Dot(p, radius=0.018, color=MUTED, fill_opacity=0.8))
                    continue
                hot = m > HOT
                a = arrow(gx[i], gy[j], float(VX[j, i]), float(VY[j, i]), m, hot)
                (arrows_hot if hot else arrows_unit).add(a)

        # the blow-up lives ON the filaments, which a coarse grid misses — find
        # the hottest well-separated exterior points on a fine grid and probe there
        res = 520
        fx = np.linspace(CX - EXT, CX + EXT, res)
        fy = np.linspace(CY - EXT, CY + EXT, res)
        FX, FY = np.meshgrid(fx, fy)
        Df = mandel_de(FX, FY)
        gyf, gxf = np.gradient(Df, fx[1] - fx[0])
        Mf = np.hypot(gxf, gyf)
        # keep only the near-boundary band — far-field drift (window corners
        # reach ~1.96 too) must not masquerade as filament heat
        Mf[(Df <= 0.0) | (Df >= PROBE_BAND)] = 0.0
        hot_pts = []
        for flat in np.argsort(Mf.ravel())[::-1]:
            j, i = divmod(int(flat), res)
            if Mf[j, i] < PROBE_MIN or len(hot_pts) >= 9:
                break
            p = np.array([fx[i], fy[j]])
            if all(np.hypot(*(p - q)) > 0.28 for q in hot_pts):
                hot_pts.append(p)
                arrows_hot.add(arrow(fx[i], fy[j], float(gxf[j, i]), float(gyf[j, i]),
                                     float(Mf[j, i]), hot=True))

        self.play(LaggedStart(*[GrowArrow(a) for a in [*arrows_unit, *arrows_hot]],
                              lag_ratio=0.012),
                  FadeIn(interior_dots), run_time=2.2)

        # right column: the estimate, the two failure modes, the declared bound
        est = MathTex(r"d(c) \;=\; \frac{\lvert z\rvert \ln\lvert z\rvert}{\lvert z'\rvert}",
                      color=INK).scale(0.72)
        b1 = VGroup(
            Line(ORIGIN, RIGHT * 0.42, color=RED, stroke_width=3).add_tip(tip_length=0.12, tip_width=0.12),
            Text("‖∇d‖ unbounded near the filaments", font_size=FS_CHIP, color=RED),
        ).arrange(RIGHT, buff=0.18)
        b2 = VGroup(
            Dot(radius=0.045, color=MUTED),
            Text("interior clamped to 0 — no gradient", font_size=FS_CHIP, color=MUTED),
        ).arrange(RIGHT, buff=0.18)
        b3 = VGroup(
            Line(ORIGIN, RIGHT * 0.42, color=TRAIL, stroke_width=3).add_tip(tip_length=0.12, tip_width=0.12),
            Text("‖∇d‖ ≈ 1 in the far field", font_size=FS_CHIP, color=TRAIL),
        ).arrange(RIGHT, buff=0.18)
        not_sdf = Text("⇒ not a true SDF: L = 1 no longer applies", font_size=FS_CAPTION, color=INK)

        ldecl = MathTex(r"L_{\mathrm{MandelDE}} \;=\; 4", color=FIELD_HI).scale(0.8)
        lbox = SurroundingRectangle(ldecl, color=FIELD_HI, buff=0.2, corner_radius=0.1).set_stroke(width=1.5)
        lnote = VGroup(
            Text("declared once, on the type —", font_size=FS_CHIP, color=MUTED),
            Text("rotate · scale · translate propagate it", font_size=FS_CHIP, color=MUTED),
            Text("soundness intact; pruning merely relaxed", font_size=FS_CHIP, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        col = VGroup(est, b3, b1, b2, not_sdf, VGroup(lbox, ldecl), lnote)
        col.arrange(DOWN, aligned_edge=LEFT, buff=0.38).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.15)

        self.play(Write(est))
        self.play(LaggedStart(FadeIn(b3), FadeIn(b1), FadeIn(b2), lag_ratio=0.25))
        self.play(Indicate(arrows_hot, color=RED, scale_factor=1.06), FadeIn(not_sdf, shift=UP * 0.1))
        self.wait(0.4)
        self.play(Create(lbox), Write(ldecl))
        self.play(FadeIn(lnote, shift=UP * 0.1))
        self.wait(2.0)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])

    # ------------------------------------------------------------------ #
    def beat_fill(self) -> None:
        """The 20k-instance fill, embedded; counter follows the schedule; verdict."""
        title = Text("20 000 instances — can GD-ADF handle it?",
                     font_size=FS_H2, color=INK).to_edge(UP, buff=0.4)

        frames = load_frames(asset("derived/custom_primitive.mp4"))
        video = VideoMobject(frames).set_height(5.9)
        vbox = SurroundingRectangle(video, color=MUTED, buff=0.0).set_stroke(width=1.5)
        panel = Group(video, vbox).move_to(np.array([-2.2, -0.35, 0.0]))
        chip = self.source_chip("examples/gd_adf/06_custom_primitive.rs — ADF max depth 7, L = 4")

        counter_lbl = Text("mandelbrots inserted", font_size=FS_CAPTION, color=MUTED)
        counter = Integer(0, color=ACCENT, font_size=46)
        target = Text("target: 20 000", font_size=FS_CHIP, color=MUTED)
        readout = VGroup(counter_lbl, counter, target).arrange(DOWN, buff=0.3)
        readout.to_edge(RIGHT, buff=1.0).shift(UP * 0.6)
        counter_pos = counter.get_center().copy()  # pin: set_value re-typesets

        prog = ValueTracker(0.0)
        n = video.n_frames

        def count_at(alpha: float) -> int:
            k = alpha * (n - 1)
            if k < 1.0:      # frame 0: the seed shape alone, nothing inserted yet
                return 0
            if k >= n - 2:   # the final emitted frame is the full 20k
                return COUNT
            return min(int(np.ceil(FIRST * RATIO ** (k - 1.0))), COUNT)

        self.play(FadeIn(title), FadeIn(video), Create(vbox), FadeIn(chip), FadeIn(readout))
        video.add_updater(lambda m: m.set_frame(prog.get_value() * (m.n_frames - 1)))
        counter.add_updater(lambda m: m.set_value(count_at(prog.get_value())).move_to(counter_pos))
        self.play(prog.animate.set_value(1.0), run_time=n / 30.0, rate_func=linear)
        video.clear_updaters()
        counter.clear_updaters()
        counter.set_value(COUNT)
        self.wait(0.6)

        # the quoted figures (script-verbatim: "7 seconds, 4.74 MiB, no obvious errors.")
        def stat(value: str, label: str, color=ACCENT) -> VGroup:
            v = Text(value, font_size=FS_H2, color=color)
            l = Text(label, font_size=FS_CHIP, color=MUTED)
            return VGroup(v, l).arrange(RIGHT, buff=0.25)

        stats = VGroup(
            stat("7 s", "— insertion time"),
            stat("4.74 MiB", "— ADF size"),
            stat("✓", "no obvious errors", color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32)
        stats.to_edge(RIGHT, buff=1.0).shift(DOWN * 1.6)

        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.15) for s in stats], lag_ratio=0.22),
                  run_time=1.6)
        self.wait(2.4)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])
