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


def build_fractal_mp4(out: Path, fps: int = 30, size: int = 640) -> None:
    """Build the `fractal_distribution` render binary and pipe it to ffmpeg.

    The binary lives in the `doc/video/render` crate (which imports space-filling
    via a local path dependency). It streams a PNG per frame to stdout; we feed
    that straight into ffmpeg. The pipe is wired with subprocess (no shell), so
    the binary frames are not mangled the way a PowerShell pipe would mangle them.
    """
    DERIVED.mkdir(exist_ok=True)
    render_dir = Path(__file__).resolve().parent / "render"
    src = render_dir / "src" / "bin" / "fractal_distribution.rs"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        print(f"skip {out.name} (up to date)")
        return
    exe = render_dir / "target" / "release" / (
        "fractal_distribution.exe" if os.name == "nt" else "fractal_distribution"
    )
    subprocess.run(["cargo", "build", "--release", "--bin", "fractal_distribution"],
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
    build_fractal_mp4(DERIVED / "fractal_distribution.mp4")
