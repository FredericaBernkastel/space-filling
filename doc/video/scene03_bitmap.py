"""Scene 3 — Argmax2D: a bitmap on a Z-order curve.

Beats: (1) rasterize the compound field onto an N×N bitmap, thread the cells by
a Z-order (Morton) curve so neighbours stay close in memory, and linearly scan
for the global maximum; (2) the catch — O(1) lookup but quadratic time *and*
memory (a placeholder slot stands in for the Rust example render).

Render:
    uv run manim -ql scene03_bitmap.py Scene03Bitmap
"""

import numpy as np
from manim import *

import fields as F
from theme import (
    VideoScene, INK, MUTED, ACCENT, BG, TRAIL, FIELD_HI, asset, mono, mono_span, rich_text,
    FS_TITLE, FS_H2, FS_BODY, FS_CAPTION, FS_CHIP,
)
from video import load_frames, VideoMobject

GREEN = "#3fb950"


def morton_order(n):
    """Cell coordinates (x, y) in Z-order for an n×n grid (n a power of 2)."""
    bits = n.bit_length() - 1
    order = []
    for m in range(n * n):
        x = y = 0
        for b in range(bits):
            x |= ((m >> (2 * b)) & 1) << b
            y |= ((m >> (2 * b + 1)) & 1) << b
        order.append((x, y))
    return order


class Scene03Bitmap(VideoScene):
    def construct(self) -> None:
        self.beat_bitmap()
        self.beat_tradeoff()

    # ------------------------------------------------------------------ #
    def beat_bitmap(self) -> None:
        N, extent, s = 8, 1.2, 0.56
        G = np.array([-1.8, 0.55, 0.0])

        def cc(x, y):
            return G + np.array([(x - (N - 1) / 2) * s, ((N - 1) / 2 - y) * s, 0.0])

        # compound field -> one sample per cell
        shapes = [
            F.circle(-0.55, 0.45, 0.32), F.circle(0.45, 0.55, 0.26),
            F.box(0.5, -0.45, 0.24, 0.24), F.regular_polygon(-0.55, -0.45, 3, 0.32),
        ]
        g = F.union(F.frame(1.25), *shapes)
        vals = np.zeros((N, N))
        for y in range(N):
            for x in range(N):
                fx = -extent + (x + 0.5) / N * 2 * extent
                fy = extent - (y + 0.5) / N * 2 * extent
                vals[y, x] = g(fx, fy)
        t = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        rgb = F._ramp(t)

        def hexof(x, y):
            r, gg, b = (rgb[y, x] * 255).astype(int)
            return "#%02x%02x%02x" % (r, gg, b)

        cells = VGroup(*[
            Square(side_length=s).move_to(cc(x, y)).set_fill(hexof(x, y), 1.0).set_stroke(BG, width=0.5)
            for y in range(N) for x in range(N)
        ])

        title = Text("rasterize the field — one sample per cell", font_size=FS_H2, color=MUTED)
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title))
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.008), run_time=1.6)

        # Z-order (Morton) curve threading the cells
        order = morton_order(N)
        pts = [cc(x, y) for (x, y) in order]
        curve = VMobject(stroke_color=MUTED, stroke_width=2).set_points_as_corners(pts)
        zlabel = VGroup(
            Text("Z-order (Morton) curve", font_size=FS_CAPTION, color=INK),
            rich_text("→ cache-local", font_size=FS_CHIP, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        zlabel.next_to(cells, RIGHT, buff=0.6).shift(UP * 1.3)
        self.play(Create(curve), run_time=2.2)
        self.play(FadeIn(zlabel))

        # 1-D memory strip: the same cells unrolled in Morton order
        W, ch = 11.0, 0.42
        cw = W / (N * N)

        def strip_pos(i):
            return np.array([-W / 2 + (i + 0.5) * cw, -2.7, 0.0])

        strip = VGroup(*[
            Rectangle(width=cw, height=ch).move_to(strip_pos(i))
            .set_fill(hexof(*order[i]), 1.0).set_stroke(width=0)
            for i in range(N * N)
        ])
        strip_lbl = Text("linear memory", font_size=FS_CHIP, color=MUTED).next_to(strip, DOWN, buff=0.18)
        self.play(FadeIn(strip, shift=UP * 0.1), FadeIn(strip_lbl))

        # linear scan: cursor sweeps grid + strip; best marker tracks the running max
        order_vals = [float(vals[y, x]) for (x, y) in order]
        alpha = ValueTracker(0.0)
        last = len(order) - 1

        def idx(a):
            return min(last, int(a * last))

        def best_xy(a):
            count = min(len(order), int(a * len(order)) + 1)
            return order[int(np.argmax(order_vals[:count]))]

        cursor = Square(side_length=s * 1.02).set_stroke(INK, 3).set_fill(opacity=0).move_to(cc(*order[0]))
        best = Square(side_length=s * 1.06).set_stroke(ACCENT, 4).set_fill(opacity=0).move_to(cc(*best_xy(0)))
        strip_cursor = Rectangle(width=cw, height=ch).set_stroke(INK, 2).set_fill(opacity=0).move_to(strip_pos(0))
        trail = always_redraw(lambda: VMobject(stroke_color=TRAIL, stroke_width=4)
                              .set_points_as_corners(pts[:max(2, idx(alpha.get_value()) + 1)]))

        cursor.add_updater(lambda m: m.move_to(cc(*order[idx(alpha.get_value())])))
        best.add_updater(lambda m: m.move_to(cc(*best_xy(alpha.get_value()))))
        strip_cursor.add_updater(lambda m: m.move_to(strip_pos(idx(alpha.get_value()))))

        scan_lbl = Text("scan for the global maximum", font_size=FS_CAPTION, color=ACCENT)
        scan_lbl.next_to(cells, RIGHT, buff=0.6).shift(DOWN * 0.2)

        self.add(trail)
        self.play(FadeIn(cursor), FadeIn(best), FadeIn(strip_cursor), FadeIn(scan_lbl))
        self.play(alpha.animate.set_value(1.0), run_time=6.0, rate_func=linear)

        for m in (cursor, best, strip_cursor):
            m.clear_updaters()
        gx, gy = order[int(np.argmax(order_vals))]
        result = Text("global maximum — O(1) lookup", font_size=FS_CAPTION, color=FIELD_HI)
        result.next_to(cells, RIGHT, buff=0.6).shift(DOWN * 0.2)
        self.play(FadeOut(scan_lbl), FadeOut(cursor), Flash(cc(gx, gy), color=ACCENT, line_length=0.3))
        self.play(FadeIn(result))
        self.wait(1.4)

        self.play(FadeOut(VGroup(cells, curve, best, trail, strip, strip_cursor,
                                 strip_lbl, zlabel, result, title)))

    # ------------------------------------------------------------------ #
    def beat_tradeoff(self) -> None:
        # the actual Rust example render (built by build_assets pipeline), playing
        frames = load_frames(asset("derived/fractal_distribution.mp4"))
        video = VideoMobject(frames).set_height(4.7)
        vbox = SurroundingRectangle(video, color=MUTED, buff=0.0).set_stroke(width=1.2)
        panel = Group(video, vbox).to_edge(LEFT, buff=1.3).shift(UP * 0.2)
        cap = mono("01_fractal_distribution.rs", font_size=FS_CHIP, color=MUTED).next_to(vbox, DOWN, buff=0.2)

        ok = rich_text("lookup  O(1)   ✓", font_size=FS_CAPTION, color=GREEN)
        head = Text("increasing precision costs:", font_size=FS_BODY, color=INK)
        s_time = rich_text("time      O(N²)", font_size=FS_BODY, color=ACCENT)
        s_mem = rich_text("memory  O(N²)", font_size=FS_BODY, color=ACCENT)
        d_grid = MarkupText(f'{mono_span("4096 × 4096 grid")}', font_size=FS_CAPTION, color=MUTED)
        d_err = MarkupText(mono_span("avg. error  2.44 × 10⁻⁴"), font_size=FS_CAPTION, color=MUTED)
        d_mem = MarkupText(mono_span("64 MB"), font_size=FS_CAPTION, color=FIELD_HI)
        stats = VGroup(ok, head, s_time, s_mem, d_grid, d_err, d_mem).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        stats.to_edge(RIGHT, buff=1.3)

        prog = ValueTracker(0.0)
        self.play(FadeIn(video), Create(vbox), FadeIn(cap))
        video.add_updater(lambda m: m.set_frame(prog.get_value() * (video.n_frames - 1)))
        self.play(
            prog.animate.set_value(0.45),
            LaggedStart(*[FadeIn(m, shift=UP * 0.1) for m in stats], lag_ratio=0.18),
            run_time=3.2, rate_func=linear,
        )
        self.play(prog.animate.set_value(1.0), run_time=3.0, rate_func=linear)
        video.clear_updaters()
        self.wait(0.4)

        verdict = Text("a dead end.", font_size=FS_TITLE, color=ACCENT, weight=BOLD).to_edge(DOWN, buff=0.55)
        self.play(Write(verdict))
        self.wait(1.6)
        self.play(FadeOut(Group(panel, cap, stats, verdict)))
