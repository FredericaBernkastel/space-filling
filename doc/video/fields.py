"""Numpy signed-distance primitives and a contour-plot colorizer.

Reused by every scene that shows a distance field. SDFs are vectorized: each
returns a *closure* ``f(X, Y)`` taking meshgrid arrays (or scalars) and
returning signed distance, **negative inside** the shape. Unit shapes match
``src/geometry/shapes.rs`` and are inscribed in the unit circle; a ``(cx, cy, r)``
wrapper places and scales them.

The colorizer keys brightness to the field *value* (so maxima glow) and overlays
thin, resolution-independent contour lines — no colour banding.
"""

from __future__ import annotations

import numpy as np
from manim import ImageMobject

# --- primitives (return a closure; negative inside) ----------------------


def circle(cx: float = 0.0, cy: float = 0.0, r: float = 0.62):
    def f(X, Y):
        return np.hypot(X - cx, Y - cy) - r
    return f


def box(cx: float = 0.0, cy: float = 0.0, bx: float = 0.55, by: float = 0.55):
    def f(X, Y):
        dx = np.abs(X - cx) - bx
        dy = np.abs(Y - cy) - by
        outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
        inside = np.minimum(np.maximum(dx, dy), 0.0)
        return outside + inside
    return f


def polygon(verts):
    """Signed distance to an arbitrary simple polygon (matches Polygon)."""
    V = [np.asarray(p, dtype=float) for p in verts]
    N = len(V)

    def f(X, Y):
        px = np.asarray(X, dtype=float)
        py = np.asarray(Y, dtype=float)
        d = (px - V[0][0]) ** 2 + (py - V[0][1]) ** 2
        s = np.ones_like(d)
        j = N - 1
        for i in range(N):
            vi, vj = V[i], V[j]
            ex, ey = vj[0] - vi[0], vj[1] - vi[1]
            wx, wy = px - vi[0], py - vi[1]
            t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
            d = np.minimum(d, (wx - ex * t) ** 2 + (wy - ey * t) ** 2)
            c1 = py >= vi[1]
            c2 = py < vj[1]
            c3 = (ex * wy - ey * wx) > 0.0
            s = np.where((c1 & c2 & c3) | (~c1 & ~c2 & ~c3), -s, s)
            j = i
        return s * np.sqrt(d)
    return f


def regular_polygon(cx=0.0, cy=0.0, n=5, r=0.7, rot=np.pi / 2):
    verts = [
        (cx + r * np.cos(rot + 2 * np.pi * k / n),
         cy + r * np.sin(rot + 2 * np.pi * k / n))
        for k in range(n)
    ]
    return polygon(verts)


def segment(ax, ay, bx, by, th=0.14):
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby

    def f(X, Y):
        wx, wy = X - ax, Y - ay
        t = np.clip((wx * abx + wy * aby) / L2, 0.0, 1.0)
        return np.hypot(wx - abx * t, wy - aby * t) - th
    return f


def moon(cx=0.0, cy=0.0, r=0.72, off=0.42):
    outer = circle(cx, cy, r)
    inner = circle(cx + off, cy + 0.10, r)

    def f(X, Y):
        return np.maximum(outer(X, Y), -inner(X, Y))
    return f


# --- unit shapes matching shapes.rs (inscribed in the unit circle) -------


def _star_unit(px, py, n, m):
    an = np.pi / n
    en = np.pi / m
    acs = (np.cos(an), np.sin(an))
    ecs = (np.cos(en), np.sin(en))
    ang = np.mod(np.arctan2(px, py), 2 * an) - an
    L = np.hypot(px, py)
    qx = np.cos(ang) * L - acs[0]
    qy = np.abs(np.sin(ang)) * L - acs[1]
    cl = np.clip(-(qx * ecs[0] + qy * ecs[1]), 0.0, acs[1] / ecs[1])
    qx = qx + ecs[0] * cl
    qy = qy + ecs[1] * cl
    return np.hypot(qx, qy) * np.sign(qx)


def _cross_unit(px, py, th):
    ax, ay = np.abs(px), np.abs(py)
    sx = np.where(ay > ax, ay, ax)
    sy = np.where(ay > ax, ax, ay)
    qx, qy = sx - 1.0, sy - th
    k = np.maximum(qx, qy)
    wx = np.where(k > 0, qx, th - sx)
    wy = np.where(k > 0, qy, -k)
    return np.sign(k) * np.hypot(np.maximum(wx, 0.0), np.maximum(wy, 0.0))


def _ring_unit(px, py, inner_r):
    L = np.hypot(px, py)
    return np.maximum(L - 1.0, inner_r - L)


def _kakera_unit(px, py, width):
    bx, by = width, 1.0
    qx, qy = np.abs(px), np.abs(py)
    h = np.clip((-2 * (qx * bx - qy * by) + (bx * bx - by * by)) / (bx * bx + by * by), -1.0, 1.0)
    d = np.hypot(qx - (1.0 - h) * bx / 2.0, qy - (1.0 + h) * by / 2.0)
    return d * np.sign(qx * by + qy * bx - bx * by)


def _placed(unit_fn, cx, cy, r):
    def f(X, Y):
        return unit_fn((X - cx) / r, (Y - cy) / r) * r
    return f


def star(cx=0.0, cy=0.0, n=5, m=3.0, r=1.0):
    return _placed(lambda x, y: _star_unit(x, y, n, m), cx, cy, r)


def cross(cx=0.0, cy=0.0, th=0.32, r=1.0):
    return _placed(lambda x, y: _cross_unit(x, y, th), cx, cy, r)


def ring(cx=0.0, cy=0.0, inner_r=0.5, r=1.0):
    return _placed(lambda x, y: _ring_unit(x, y, inner_r), cx, cy, r)


def kakera(cx=0.0, cy=0.0, width=0.6, r=1.0):
    return _placed(lambda x, y: _kakera_unit(x, y, width), cx, cy, r)


def pentagram(cx=0.0, cy=0.0, r=1.0):
    return star(cx, cy, 5, 10.0 / 3.0, r)


def hexagram(cx=0.0, cy=0.0, r=1.0):
    return star(cx, cy, 6, 3.0, r)


def frame(B=2.6, cx=0.0, cy=0.0):
    """Domain walls: Euclidean distance to a square boundary, positive inside.

    This is the ``boundary_rect`` primitive — always the first one inserted.
    It bounds the compound field so interior maxima (largest empty pockets)
    actually exist; without it, distance grows without bound into empty space.
    """
    def f(X, Y):
        return B - np.maximum(np.abs(X - cx), np.abs(Y - cy))
    return f


def union(*fns):
    """Compound field: pointwise minimum of the operands."""
    def f(X, Y):
        d = fns[0](X, Y)
        for g in fns[1:]:
            d = np.minimum(d, g(X, Y))
        return d
    return f


# --- rasterization + colour ---------------------------------------------


def sample(fn, res=320, extent=1.2, center=(0.0, 0.0)):
    xs = np.linspace(center[0] - extent, center[0] + extent, res)
    ys = np.linspace(center[1] + extent, center[1] - extent, res)  # row 0 = +y (top)
    X, Y = np.meshgrid(xs, ys)
    return fn(X, Y)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# perceptual dark -> bright ramp: deep indigo, blue, teal, green, gold, near-white
_RAMP = (
    (0.00, (0.043, 0.055, 0.118)),
    (0.32, (0.086, 0.196, 0.360)),
    (0.56, (0.114, 0.447, 0.500)),
    (0.76, (0.298, 0.686, 0.522)),
    (0.90, (0.902, 0.780, 0.376)),
    (1.00, (1.000, 0.965, 0.860)),
)


def _ramp(t):
    t = np.clip(t, 0.0, 1.0)
    r = np.full_like(t, _RAMP[0][1][0])
    g = np.full_like(t, _RAMP[0][1][1])
    b = np.full_like(t, _RAMP[0][1][2])
    for (t0, c0), (t1, c1) in zip(_RAMP[:-1], _RAMP[1:]):
        m = (t >= t0) & (t <= t1)
        f = (t[m] - t0) / (t1 - t0)
        r[m] = c0[0] + f * (c1[0] - c0[0])
        g[m] = c0[1] + f * (c1[1] - c0[1])
        b[m] = c0[2] + f * (c1[2] - c0[2])
    return np.stack([r, g, b], axis=-1)


def colorize(d, interval=0.12, line_strength=0.5, vmin=None, vmax=None):
    """Field -> RGB: brightness rises with the value (maxima glow); thin,
    resolution-independent contour lines every ``interval`` distance units."""
    d = np.asarray(d, dtype=float)
    if vmin is None:
        vmin = float(d.min())
    if vmax is None:
        vmax = float(d.max())
    t = (d - vmin) / (vmax - vmin + 1e-12)
    col = _ramp(t)
    # contour lines: ~1.2 px wide regardless of resolution (fwidth trick)
    f = d / interval
    gy, gx = np.gradient(f)
    fw = np.hypot(gx, gy) + 1e-6
    tri = np.abs(np.mod(f - 0.5, 1.0) - 0.5)
    line = 1.0 - _smoothstep(0.0, 1.2 * fw, tri)
    col = col * (1.0 - line_strength * line[..., None])
    # a crisp white line at the zero level, so inside/outside read clearly
    bnd = 1.0 - _smoothstep(0.0, 1.6 * fw * interval, np.abs(d))
    white = np.array([0.96, 0.98, 1.0])
    col = col * (1.0 - bnd[..., None]) + white * bnd[..., None]
    return (np.clip(col, 0.0, 1.0) * 255).astype(np.uint8)


def field_image(fn, height=2.3, res=320, extent=1.2, center=(0.0, 0.0), **kw):
    """An ImageMobject of ``fn`` rasterized over ``[±extent]^2`` about ``center``."""
    img = ImageMobject(colorize(sample(fn, res, extent, center), **kw))
    img.height = height
    return img


# --- optimizer -----------------------------------------------------------


def ascend(fn, start, step=0.05, iters=800, eps=1e-3):
    """Gradient ascent on a scalar field; returns the path as an (M, 2) array.

    Steps along the normalized gradient (a distance field has unit-magnitude
    gradient), halving the step when it would overshoot the ridge, until the
    step collapses — i.e. it converges onto the medial-axis maximum.
    """
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(iters):
        gx = (fn(p[0] + eps, p[1]) - fn(p[0] - eps, p[1])) / (2 * eps)
        gy = (fn(p[0], p[1] + eps) - fn(p[0], p[1] - eps)) / (2 * eps)
        n = float(np.hypot(gx, gy))
        if n < 1e-6:
            break
        nxt = p + step * np.array([gx, gy]) / n
        if float(fn(nxt[0], nxt[1])) <= float(fn(p[0], p[1])):
            step *= 0.5
            if step < 1e-3:
                break
            continue
        p = nxt
        path.append(p.copy())
    return np.array(path)
