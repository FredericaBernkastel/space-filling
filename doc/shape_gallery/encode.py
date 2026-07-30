"""Colour the raw fields dumped by `shape-gallery` and write doc/shapes/*.webp.

The colouriser is lifted verbatim from ``doc/video/fields.py`` so the API docs
and the explainer video speak the same visual language: brightness rises with
the field value (so maxima glow), thin resolution-independent contour lines every
``interval`` distance units, and a crisp white line on the zero level.

Multi-frame entries are 2D slices swept through a shape that has no planar
analogue, written as animated WebP. The sweep is already a full cosine cycle, so
the frames loop seamlessly as they stand.

    cd doc/video && uv run python ../shape_gallery/encode.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "shapes"
RAW = ROOT / "_raw"


# --- doc/video/fields.py, unchanged --------------------------------------


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


# --- encoding ------------------------------------------------------------


def load(name):
    blob = (RAW / f"{name}.bin").read_bytes()
    frames, w, h = struct.unpack_from("<III", blob, 0)
    a = np.frombuffer(blob, dtype="<f4", count=frames * w * h, offset=12)
    return a.reshape(frames, h, w).astype(np.float64)


def main():
    manifest = (RAW / "manifest.txt").read_text().split("\n")
    total = 0
    for line in manifest:
        if not line.strip():
            continue
        name, frames, interval = line.split()
        frames, interval = int(frames), float(interval)
        stack = load(name)
        out = ROOT / f"{name}.webp"

        if frames == 1:
            img = Image.fromarray(colorize(stack[0], interval=interval))
            img.save(out, "WEBP", quality=92, method=6)
        else:
            # one scale for the whole sweep, so brightness does not pump between
            # frames as the cross-section grows and shrinks
            vmin, vmax = float(stack.min()), float(stack.max())
            imgs = [
                Image.fromarray(colorize(s, interval=interval, vmin=vmin, vmax=vmax))
                for s in stack
            ]
            imgs[0].save(
                out, "WEBP", save_all=True, append_images=imgs[1:],
                duration=80, loop=0, quality=80, method=4,
            )
        kb = out.stat().st_size / 1024
        total += out.stat().st_size
        print(f"{name:<22} {frames:>3} frame(s)  {kb:>7.1f} KiB")
    print(f"\n{total / 1024 / 1024:.2f} MiB total -> {ROOT}")


if __name__ == "__main__":
    sys.exit(main())
