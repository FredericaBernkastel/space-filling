"""Stage three: write doc/shapes/*.avif from what the first two stages produced.

Planar shapes arrive as raw field values (`_raw/`) and are coloured here. The
colouriser is lifted verbatim from ``doc/video/fields.py`` so the API docs and the
explainer video speak the same visual language: brightness rises with the field
value (so maxima glow), thin resolution-independent contour lines every
``interval`` distance units, and a crisp white line on the zero level. Those are
ordinary sRGB images and are written as still AVIF.

Shapes that do not fit in the plane arrive as a directory of Blender frames
(`_frames/<name>/NNN.avif`), 10-bit HDR in Rec.2100 HLG, and are muxed into one
animated AVIF. Nothing here touches those pixels. Any Python image library on hand
decodes them to 8 bits, which would throw away exactly the range they were rendered
for, so everything that needed doing to them — the bloom above all — happens in
Blender's compositor while the values are still scene-linear floats. What is left is
a container job, and ffmpeg does it.

The frames are decoded to raw planar 4:4:4 and re-encoded rather than being passed
through as-is, for two reasons: the published animation is half the rendered size,
and ffmpeg cannot give a sequence of stills sensible timing on its own — every
route through its image demuxers either refuses `.avif` or invents a duration per
frame and then duplicates frames to fill it. Raw video has no timestamps to
misinterpret, so `-framerate` means what it says. Both kinds of sequence are already
closed loops — a full turntable, or a cosine sweep out and back — so the frames need
no ping-ponging.

    cd doc/video && uv run python ../shape_gallery/encode.py
"""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "shapes"
RAW = ROOT / "_raw"
FRAMES = ROOT / "_frames"
MESH = ROOT / "_mesh"

SIZE = 512      # published edge length, in pixels
FPS = 12.5      # 80 ms a frame
CRF = 24        # libaom quality; AVIF is efficient enough that this is cheap
CPU_USED = 4    # libaom speed/quality dial, 0 slowest

# Rec.2100 HLG, matching what Blender wrote into the frames. Raw video carries no
# such tags, so they are stamped back on explicitly — without them a browser reads
# the result as ordinary sRGB and the whole point of the HDR output is lost.
HDR = {
    "color_primaries": "bt2020",
    "color_trc": "arib-std-b67",
    "colorspace": "bt2020nc",
    "color_range": "pc",
}
# 4:4:4 rather than 4:2:0: the glowing creases are one- to two-pixel lines of
# saturated warm light on a cool body, which is precisely what chroma subsampling
# smears.
PIX_FMT = "yuv444p10le"


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


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True)


def planar(name, interval):
    """One contour plot of a field sampled in the plane."""
    out = ROOT / f"{name}.avif"
    Image.fromarray(colorize(load(name)[0], interval=interval)).save(
        out, "AVIF", quality=90, subsampling="4:4:4"
    )
    return out, 1


def rendered(name, frames):
    """A Blender frame sequence, scaled down and muxed into animated AVIF."""
    out = ROOT / f"{name}.avif"
    with tempfile.TemporaryDirectory() as tmp:
        stack = Path(tmp) / "frames.yuv"
        with open(stack, "wb") as sink:
            for i in range(frames):
                # Downscaled here, a frame at a time. The renders are at double size
                # on purpose — the meshes alias an acute crease by about a grid cell,
                # and averaging four pixels into one puts what is left of that under
                # a pixel.
                decoded = subprocess.run(
                    ["ffmpeg", "-v", "error",
                     "-i", str(FRAMES / name / f"{i:03d}.avif"),
                     "-vf", f"scale={SIZE}:{SIZE}:flags=lanczos",
                     "-pix_fmt", PIX_FMT, "-f", "rawvideo", "-"],
                    check=True, stdout=subprocess.PIPE,
                )
                sink.write(decoded.stdout)

        # `setparams` stamps the frames, and the output flags tag the stream; both
        # are needed, since raw video arrives with nothing to propagate.
        stamp = "setparams=" + ":".join(
            f"{'range' if k == 'color_range' else k}={v}" for k, v in HDR.items()
        )
        ffmpeg([
            "-f", "rawvideo", "-pixel_format", PIX_FMT,
            "-video_size", f"{SIZE}x{SIZE}", "-framerate", str(FPS),
            "-i", str(stack),
            "-vf", stamp,
            "-c:v", "libaom-av1", "-crf", str(CRF), "-cpu-used", str(CPU_USED),
            "-pix_fmt", PIX_FMT,
            *[arg for k, v in HDR.items() for arg in (f"-{k}", v)],
            "-loop", "0", "-f", "avif", str(out),
        ])
    return out, frames


def main():
    jobs = []
    for line in (RAW / "manifest.txt").read_text().split("\n"):
        if line.strip():
            name, _, interval = line.split()
            jobs.append((name, planar, (name, float(interval))))
    for line in (MESH / "manifest.txt").read_text().split("\n"):
        if line.strip():
            name, _kind, frames, _radius = line.split()
            jobs.append((name, rendered, (name, int(frames))))

    total = 0
    for name, fn, argv in jobs:
        out, frames = fn(*argv)
        size = out.stat().st_size
        total += size
        kind = "plot" if fn is planar else "render"
        print(f"{name:<22} {kind:<7} {frames:>3} frame(s)  {size / 1024:>7.1f} KiB",
              flush=True)
    print(f"\n{len(jobs)} images, {total / 1024 / 1024:.2f} MiB total -> {ROOT}")


if __name__ == "__main__":
    sys.exit(main())
