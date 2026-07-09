"""The YouTube thumbnail — one static frame, the full scope of the research.

Hero: the signature contour field — library shapes (triangle, hexagon, an
arbitrary polygon, a cross, a SmoothMin pair, star, moon, box) plus the
Mandelbrot estimator — textured with streamplot-style gradient flow lines and
an adaptive quadtree, one exponential-decay ascent ending in its free ball.
Left column: equation (1), the GD-ADF title, the O(N^2) -> O(log N) hook, the
Implementation-section algebra in badges, and the method strip.

Render (1920x1080; YouTube downscales to 1280x720):
    uv run manim -r 1920,1080 -s thumbnail.py Thumbnail
"""

import numpy as np
from manim import *

import fields as F
from scene08_mandelbrot import mandel_de
from theme import (
    VideoScene, rich_text, BG, INK, MUTED, ACCENT, COOL, TRAIL, FIELD_HI, mono
)

RED = "#ff6b6b"

# field window, 16:9 — y in [-B, B], x in [-B*16/9, B*16/9]
B = 3.4
WX = B * 16.0 / 9.0
RX, RY = 960, 540


def mandel_placed(cx, cy, s):
    """The example's normalized Mandelbrot ([-1,1] box), placed at (cx, cy), scale s."""
    def f(X, Y):
        return mandel_de(1.5 * (X - cx) / s - 1.0, 1.5 * (Y - cy) / s) * s / 1.5
    return f


def wide_frame(X, Y):
    """True distance to the 16:9 window border — positive inside, the whole image."""
    return np.minimum(WX - np.abs(X), B - np.abs(Y))


SHAPES = [
    F.star(-5.1, -0.6, 7, 3.0, 0.66),
    F.cross(-4.45, 1.75, 0.34, 0.66),
    F.polygon([(-3.75, -2.35), (-2.55, -2.6), (-2.3, -1.75), (-2.95, -1.15), (-3.7, -1.5)]),
    F.box(-1.35, 2.15, 0.5, 0.5),
    F.moon(-1.7, -0.35, 0.62, 0.38),
    F.circle(0.4, -2.3, 0.55),
    F.regular_polygon(1.15, 1.9, 3, 0.66),                     # triangle
    F.smooth_min(F.circle(4.9, 2.35, 0.44), F.circle(5.75, 1.9, 0.38), k=6.0),
    F.regular_polygon(5.6, -2.1, 6, 0.62),                     # hexagon
]
MANDEL = mandel_placed(3.35, -0.15, 1.55)
ALL = SHAPES + [MANDEL]
G = F.union(wide_frame, *ALL)  # bounded over the WHOLE window, not a centre square

# one shared raster of the field: the colorized hero AND the flow-line vector
# grid both come from it (streamplot-style grid-sampled gradient)
_XS = np.linspace(-WX, WX, RX)
_YS = np.linspace(B, -B, RY)  # row 0 = +y
_D = G(*np.meshgrid(_XS, _YS))
_dGrow, _dGcol = np.gradient(_D)
_GX = _dGcol / (2 * WX / (RX - 1))
_GY = _dGrow / (-2 * B / (RY - 1))


def _interp(A, x, y):
    """Bilinear sample of grid ``A`` at field coordinates (x, y)."""
    fx = np.clip((x + WX) / (2 * WX) * (RX - 1), 0, RX - 1 - 1e-9)
    fy = np.clip((B - y) / (2 * B) * (RY - 1), 0, RY - 1 - 1e-9)
    i0, j0 = int(fx), int(fy)
    tx, ty = fx - i0, fy - j0
    return float((A[j0, i0] * (1 - tx) + A[j0, i0 + 1] * tx) * (1 - ty)
                 + (A[j0 + 1, i0] * (1 - tx) + A[j0 + 1, i0 + 1] * tx) * ty)


def flow_dir(x, y):
    """Unit gradient from the sampled grid (smooth, streamplot-style)."""
    gx, gy = _interp(_GX, x, y), _interp(_GY, x, y)
    n = float(np.hypot(gx, gy))
    return (gx / n, gy / n) if n > 1e-6 else (0.0, 0.0)


def grad(x, y, eps=1e-3):
    """Exact central-difference gradient (used by the hero ascent)."""
    gx = (float(G(x + eps, y)) - float(G(x - eps, y))) / (2 * eps)
    gy = (float(G(x, y + eps)) - float(G(x, y - eps))) / (2 * eps)
    return gx, gy


def g_wide():
    return F.colorize(_D, interval=0.11)


def flow_lines(to_scene):
    """Gradient streamlines over the whole window — RK2 on the grid field,
    bezier-smoothed, one crisp notched arrowhead per line (streamplot look)."""
    LINE_OP, TIP_OP = 0.5, 0.5
    LINE_LEN = 1.5

    def tip(at, direction, scale=0.07):
        a = np.arctan2(direction[1], direction[0])
        pts = [np.array([1.1, 0.0, 0.0]), np.array([-0.5, 0.62, 0.0]),
               np.array([-0.18, 0.0, 0.0]), np.array([-0.5, -0.62, 0.0])]
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        return (Polygon(*[at + scale * (rot @ p) for p in pts])
                .set_fill(RED, opacity=TIP_OP).set_stroke(width=0))

    lines = VGroup()
    nx, ny = 32, 32
    h = 0.05
    for i in range(nx):
        for j in range(ny):
            # deterministic jitter, no RNG
            jx = 0.29 * np.sin(12.9898 * i + 78.233 * j)
            jy = 0.29 * np.sin(39.3468 * j + 11.135 * i)
            x = -WX + (i + 0.5) / nx * 2 * WX + jx
            y = -B + (j + 0.5) / ny * 2 * B + jy
            if _interp(_D, x, y) < 0.06:  # inside / on a shape: no flow to show
                continue
            p = np.array([x, y])
            pts = [p.copy()]
            prev = None
            for _ in range(44):
                d1 = np.array(flow_dir(*p))
                if not d1.any():
                    break
                d2 = np.array(flow_dir(*(p + LINE_LEN * h * d1)))  # RK2 midpoint
                step = d2 if d2.any() else d1
                if prev is not None and float(np.dot(step, prev)) < 0.15:
                    break  # the ridge: stop cleanly instead of wiggling
                p = p + h * step
                if abs(p[0]) > WX - 0.04 or abs(p[1]) > B - 0.04:
                    break
                pts.append(p.copy())
                prev = step
            if len(pts) < 8:
                continue
            spts = [to_scene(*q) for q in pts]
            m = VMobject(stroke_color=RED, stroke_width=1.6, stroke_opacity=LINE_OP)
            m.set_points_smoothly(spts[::3] + [spts[-1]])
            lines.add(m)
            mid = len(pts) // 2
            lines.add(tip(spts[mid], np.array(flow_dir(*pts[mid]))))
    return lines


def quadtree_cells(to_scene):
    """Adaptive cells over the right 2/3 of the window (square root domain)."""
    side = 2 * B
    cx0, cy0 = WX - B, 0.0  # flush with the right edge
    fns = ALL

    def near(cx, cy, size):
        return min(abs(float(f(cx, cy))) for f in fns) < size * 0.72

    out = []

    def rec(cx, cy, size, d):
        if d < 5 and near(cx, cy, size):
            q = size / 4
            for dx, dy in [(-q, q), (q, q), (-q, -q), (q, -q)]:
                rec(cx + dx, cy + dy, size / 2, d + 1)
        else:
            out.append((cx, cy, size))

    rec(cx0, cy0, side, 0)
    u = 8.0 / (2 * B)  # scene units per field unit
    return VGroup(*[
        Rectangle(width=s * u, height=s * u).move_to(to_scene(cx, cy))
        .set_stroke(COOL, width=1.1, opacity=0.42).set_fill(opacity=0)
        for (cx, cy, s) in out
    ])


class Thumbnail(VideoScene):
    def construct(self) -> None:
        u = 8.0 / (2 * B)

        def ts(x, y):
            return np.array([x * u, y * u, 0.0])

        # --- hero field, full bleed ---
        img = ImageMobject(g_wide())
        img.height = config.frame_height
        img.width = config.frame_width
        self.add(img)

        # --- gradient flow lines + adaptive quadtree ---
        self.add(flow_lines(ts))
        self.add(quadtree_cells(ts))

        # --- one ascent into the pocket: the ORIGINAL fixed exponential-decay
        # schedule (always step, h·γ^k) — its ridge zigzag is the instantly
        # recognizable gradient-method signature; corners kept sharp on purpose
        def exp_ascend(start, h0=0.6, gamma=0.88, steps=30):
            p = np.array(start, dtype=float)
            out = [p.copy()]
            for k in range(steps):
                gx, gy = grad(*p)
                n = float(np.hypot(gx, gy))
                if n < 1e-6:
                    break
                p = p + (h0 * gamma ** k) * np.array([gx, gy]) / n
                out.append(p.copy())
            return np.array(out)

        path = exp_ascend((0.55, -2.12))
        pts = [ts(*p) for p in path]
        traj = VMobject(stroke_color=TRAIL, stroke_width=6)
        traj.set_points_as_corners(pts)
        end = path[-1]
        d = float(G(end[0], end[1]))
        ball = Circle(radius=d * u, color=INK, stroke_width=4).move_to(pts[-1])
        xstar = MathTex(r"\vec x^{*}", color=FIELD_HI).scale(0.95)
        ang = 0.9
        xstar.move_to(ball.get_center() + (d * u + 0.33) * np.array([np.cos(ang), np.sin(ang), 0.0]))
        self.add(traj, Dot(pts[0], radius=0.08, color=TRAIL),
                 Dot(pts[-1], radius=0.07, color=INK), ball, xstar)

        # --- left column: dark overlay for legibility ---
        overlay = Rectangle(width=7.6, height=config.frame_height)
        overlay.set_fill(BG, opacity=0.80).set_stroke(width=0)
        overlay.to_edge(LEFT, buff=0)
        soft = Rectangle(width=1.8, height=config.frame_height)
        soft.set_fill(BG, opacity=0.38).set_stroke(width=0)
        soft.next_to(overlay, RIGHT, buff=0)
        self.add(overlay, soft)

        # --- the research, top to bottom ---
        eq = MathTex(r"\vec{x}^{*} \;=\; \arg\max_{\vec v \in \Omega}\ \min_{n}\ \mathrm{sdf}_{n}(\vec v)",
                     color=INK).scale(0.82)

        title = Text("GD-ADF", weight=BOLD, font_size=96, color=INK,
                     t2c={"ADF": ACCENT})

        hook = MathTex(r"O(N^2)", r"\;\longrightarrow\;", r"O(\log N)").scale(1.5)
        hook[0].set_color(RED)
        hook[1].set_color(MUTED)
        hook[2].set_color(FIELD_HI)

        # the Implementation-section algebra (readme.md), badge-boxed
        def badge(mob, color):
            box = SurroundingRectangle(mob, color=color, buff=0.16, corner_radius=0.1)
            box.set_stroke(width=2).set_fill(color, opacity=0.10)
            return VGroup(box, mob)

        algebra = VGroup(
            VGroup(
                badge(MathTex(r"(f-g)(\vec v)\;\ge\;(f-g)(\vec c)-(L_f+L_g)\,h(R)",
                            color=INK).scale(0.33), COOL),
                badge(MathTex(r"\mathrm{prune}(B,R)=\{\,f_i \,:\, \lnot(f_i\ge \min_{j\ne i} f_j\ \text{on } R)\,\}",
                            color=INK).scale(0.33), "#dfa4ff"),   
            ).arrange(DOWN, aligned_edge=ORIGIN, buff=0.22),
            badge(MathTex(r"D^{*}=\{\,\vec v \,:\, g(\vec v) > \lVert \vec v-\vec x_0\rVert - d\,\}",
                color=INK).scale(0.33), ACCENT)
        ).arrange(DOWN, buff=0.22)

        methods = VGroup(
            rich_text("k-d trees • Lipschitz Continuity", font_size=22, color=MUTED),
            rich_text("branch-and-bound • momentum ascent", font_size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)

        col = VGroup(eq, title, hook, algebra, methods).arrange(DOWN, buff=0.52)
        col.move_to(overlay.get_center()).align_to(overlay, LEFT).shift(RIGHT * 0.55)
        self.add(col)
