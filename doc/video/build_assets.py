"""Rasterize source PDFs in ``assets/`` into tall PNG strips in ``assets/derived/``.

Manim can't load PDFs, so webpage/paper exports are pre-rendered once into a
single tall PNG that scenes then scroll. Re-run after changing a source PDF:

    uv run python build_assets.py

Requires poppler's ``pdftocairo`` on PATH (ships with MiKTeX).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets"
DERIVED = ASSETS / "derived"


def pdf_to_strip(pdf: Path, out: Path, dpi: int = 150, width: int = 820,
                 gap: int = 0, max_pages: int | None = None) -> None:
    """Rasterize ``pdf`` (optionally only the first ``max_pages``) and stack it."""
    DERIVED.mkdir(exist_ok=True)
    if out.exists() and out.stat().st_mtime >= pdf.stat().st_mtime:
        print(f"skip {out.name} (up to date)")
        return
    with tempfile.TemporaryDirectory() as td:
        cmd = ["pdftocairo", "-png", "-r", str(dpi)]
        if max_pages:
            cmd += ["-f", "1", "-l", str(max_pages)]
        cmd += [str(pdf), str(Path(td) / "pg")]
        subprocess.run(cmd, check=True)
        pages = sorted(Path(td).glob("pg-*.png"))
        scaled = []
        for p in pages:
            im = Image.open(p).convert("RGB")
            h = round(im.height * width / im.width)
            scaled.append(im.resize((width, h), Image.LANCZOS))
        total_h = sum(im.height for im in scaled) + gap * (len(scaled) - 1)
        strip = Image.new("RGB", (width, total_h), (0, 0, 0))
        y = 0
        for im in scaled:
            strip.paste(im, (0, y))
            y += im.height + gap
        strip.save(out)
        print(f"wrote {out.name}  {width}x{total_h}  ({len(pages)} pages)")


def build_stream_mp4(bin_name: str, out: Path, fps: int = 30, size: int = 640) -> None:
    """Build a frame-streaming binary from the `render` crate and pipe it to ffmpeg.

    The binaries live in the `doc/video/render` crate (which imports space-filling
    via a local path dependency). Each streams a PNG per frame to stdout; we feed
    that straight into ffmpeg. The pipe is wired with subprocess (no shell), so
    the binary frames are not mangled the way a PowerShell pipe would mangle them.
    """
    DERIVED.mkdir(exist_ok=True)
    render_dir = Path(__file__).resolve().parent / "render"
    src = render_dir / "src" / "bin" / f"{bin_name}.rs"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"skip {out.name} (up to date)")
        return
    exe = render_dir / "target" / "release" / (f"{bin_name}.exe" if os.name == "nt" else bin_name)
    subprocess.run(["cargo", "build", "--release", "--bin", bin_name],
                   cwd=render_dir, check=True)
    gen = subprocess.Popen([str(exe)], stdout=subprocess.PIPE)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps),
         "-i", "-", "-vf", f"scale={size}:{size}:flags=lanczos", "-pix_fmt", "yuv420p",
         "-crf", "20", str(out)],
        stdin=gen.stdout, check=True,
    )
    gen.stdout.close()
    gen.wait()
    print(f"wrote {out.name}")


def build_kenburns_mp4(src: Path, out: Path, size: int = 1280, fps: int = 30) -> None:
    """Pre-render a zoom/pan (Ken Burns) pass over a large still as an mp4.

    1M.png is 8192x8192 (64 MP) — far too large for manim to pan live (it
    reprocesses the full raster every frame), so the camera move is baked here:
    crop-and-resize per frame along a keyframed (centre, zoom) path, piped to
    ffmpeg. Zoom interpolates in log space; centres are clamped to the image.
    """
    DERIVED.mkdir(exist_ok=True)
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"skip {out.name} (up to date)")
        return
    im = Image.open(src).convert("RGB")
    W = im.width
    keys = [  # (t seconds, cx, cy, zoom) — normalized centre, zoom >= 1
        (0.0, 0.50, 0.50, 1.0),
        (1.0, 0.50, 0.50, 1.0),
        (4.2, 0.30, 0.72, 6.4),
        (7.6, 0.72, 0.35, 6.4),
        (11.0, 0.50, 0.50, 1.0),
        (11.6, 0.50, 0.50, 1.0),
    ]

    def state(t: float):
        for (t0, x0, y0, z0), (t1, x1, y1, z1) in zip(keys, keys[1:]):
            if t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                a = a * a * (3 - 2 * a)  # smoothstep easing
                return (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, z0 * (z1 / z0) ** a)
        return keys[-1][1:]

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps),
         "-i", "-", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
        stdin=subprocess.PIPE,
    )
    n = int(keys[-1][0] * fps) + 1
    for k in range(n):
        cx, cy, z = state(k / fps)
        half = 0.5 / z
        cx = min(max(cx, half), 1 - half)
        cy = min(max(cy, half), 1 - half)
        box = (round((cx - half) * W), round((cy - half) * W),
               round((cx + half) * W), round((cy + half) * W))
        im.crop(box).resize((size, size), Image.LANCZOS).save(enc.stdin, format="PNG")
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {out.name}  ({n} frames)")


if __name__ == "__main__":
    # Show only the first third of the 20-page export (the intro + technique +
    # first gallery), at full width — crisper than downscaling the whole strip,
    # and ~6.6 MP so manim can scroll it without reprocessing a huge surface.
    pdf_to_strip(
        ASSETS / "Random space filling of the plane.pdf",
        DERIVED / "bourke_webpage.png",
        dpi=150,
        width=820,
        max_pages=7,
    )
    # Adaptively Sampled Distance Fields paper (Scene 4), all 6 A5 pages
    pdf_to_strip(
        ASSETS / "Adaptively sampled distance fields.pdf",
        DERIVED / "asdf_paper.png",
        dpi=150,
        width=820,
    )
    build_stream_mp4("fractal_distribution", DERIVED / "fractal_distribution.mp4")
    # Scene 7: 02_random_distribution at 1M circles (ADF depth 10), then the
    # pre-baked zoom/pan over the final 8192^2 render
    build_stream_mp4("random_distribution", DERIVED / "random_distribution.mp4", size=1024)
    build_kenburns_mp4(ASSETS / "1M.png", DERIVED / "million_zoom.mp4")
    # Scene 8: 06_custom_primitive — 20k Mandelbrot estimators (L = 4)
    build_stream_mp4("custom_primitive", DERIVED / "custom_primitive.mp4", size=1024)
