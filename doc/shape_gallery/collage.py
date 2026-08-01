"""Build doc/shapes/_collage.avif — every shape in one animated HDR sheet.

A 7x7 grid of all 47 gallery images, each labelled with its name and the field it
draws, set in real LaTeX. The two cells the grid leaves over carry the notation
the rest of the sheet is written in.

The interesting part is that the sources are not alike. Thirty-one are still
contour plots in ordinary sRGB; sixteen are 128-frame HDR animations in Rec.2100
HLG with BT.2020 primaries. Pasting one into the other unconverted is the usual
way a sheet like this goes wrong — sRGB values read as an HLG signal come out
washed and far too bright. So the stills are put through a real conversion
(sRGB -> linear -> BT.2020 -> HLG) on the way in, and from then on everything is
one encoding and compositing is just copying pixels.

Three smaller traps, all found the hard way:

  - an AVIF written by `encode.py` holds *two* video streams, a primary still
    and the animation, and ffmpeg picks the still by default. The animation is
    stream 1; without `-map 0:v:1` every animated cell is a freeze frame.
  - `npl` has to sit on the zscale that *leaves* linear light. On the one that
    enters it the option is accepted and silently ignored, and the plots come out
    at a tenth the brightness of the renders beside them.
  - raw video carries no colour tags at all, so they are stamped back on at the
    end. Without that a browser reads the result as sRGB.

Needs LaTeX and dvipng on PATH; MiKTeX supplies both, and the video project
already depends on them through manim.

    cd doc/video && uv run python ../shape_gallery/collage.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "shapes"
OUT = ROOT / "_collage.avif"

# Canvas edge, overridable from the command line.
#
# The byte budget is not what limits this. At CRF 20 the cost came out at 5.52 MiB
# for 2048, 6.88 for 2400, 7.20 for 2560, 7.56 for 2688 and 7.87 for 2752 —
# sublinear in pixel count, so extrapolating from one point understates it, and
# 2752 was the last step under 8 MiB. But 2752 decodes at about 3 fps in Chrome
# against the 12.5 it is authored at, which makes it useless as a page element.
# 2048 plays in real time, so that is the size: the limit is the decoder, not the
# file. Cells stay under the 512 px sources either way, so none of this upscales.
SIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
GRID = 7
FPS = 12.5
CRF = 20
CPU_USED = 4
PIX_FMT = "yuv444p10le"
HDR = {
    "color_primaries": "bt2020",
    "color_trc": "arib-std-b67",
    "colorspace": "bt2020nc",
    "color_range": "pc",
}

ALEF_BOLD = Path("C:/Windows/Fonts/Alef-Bold.ttf")

# Label colours, as sRGB — `hlg()` puts them through the same conversion the
# contour plots get, so a swatch here and a pixel there mean the same thing.
ACCENT = (240, 180, 41)     # theme.py's warm highlight, for the names
INK = (255, 255, 255)       # the formulas, and the panel's border
# The panel keeps a backing; the cell labels sit straight on the image.
PANEL_SCRIM = 0.02
PANEL_SCRIM_ALPHA = 0.72
# With no plate behind them the labels have to survive whatever they land on, and
# the contour plots are brightest exactly along their bottom edge. This is a
# shadow on the glyphs themselves rather than a box behind them — set it to 0 to
# have the labels bare.
GLYPH_SHADOW = 2.2

# The design was drawn at 1440; every length below scales with the canvas so a
# larger sheet is the same sheet, not the same sheet with smaller writing.
DESIGN = 1440
NOTATION_SCALE = 0.8

# A second, tone-mapped file, for viewers that cannot show HDR.
#
# These cannot be one file. A browser without HDR support applies the HLG inverse
# OETF and then treats scene light as display light, with no reference-white
# mapping: white at signal 0.746 comes back as #8d8d8d. Correcting for that inside
# the HDR file means writing values that are wrong for a display which *can* show
# HDR — an earlier attempt here lifted the signal and did exactly that, fixing
# Firefox and blowing out Chrome. The pixels cannot serve both.
#
# So this file keeps its sources untouched, and the fallback is a separate tone
# map, to be chosen by the page with `<picture>` and `(dynamic-range: high)`.
#
# 203 nits because that is BT.2408's reference white — the level diffuse white
# sits at in an HDR grade — so mapping it to SDR white is a conversion rather than
# a taste. Normalising by the 1000-nit peak instead is precisely what makes every
# naive HDR-to-SDR path land at about half brightness.
SDR_OUT = ROOT / "_collage_sdr.avif"
SDR_REFERENCE_NITS = 203

# The field each cell draws, in the notation of its own documentation.
NOTATION = {
    "hypersphere": r"\|p\| - 1",
    "hyperrect": r"\|q^{+}\| + \min(\max_a q_a, 0)",
    "hypersquare": r"\mathrm{Hyperrect}(2)",
    "line": r"d(p, \overline{ab}) - t/2",
    "ring": r"\max(\|p\|-1,\ r-\|p\|)",
    "moon": r"\max(\|p\|-1,\ 1-\|p-d\|)",
    "kakera": r"\pm\|q - \mathrm{proj}\,q\|",
    "cross": r"\min_a d(p, \mathrm{arm}_a) - t",
    "orthoplex": r"(\textstyle\sum_a |p_a| - 1)/\sqrt{D}",
    "lp_ball": r"\|p\|_p - 1",
    "polyline": r"\min_i d(p, s_i) - t/2",
    "simplex": r"\max_i (n_i\!\cdot\!p - d_i)",
    "permutohedron": r"\max_i (n_i\!\cdot\!p - d_i)",
    "polytope": r"\max_i (n_i\!\cdot\!p - d_i)",
    "gyroid": r"\tfrac{1}{2k\sqrt{D}}\textstyle\sum_a \sin kx_a \cos kx_{a+1}",
    "boundary_rect": r"\min_a \min(p_a,\ 1-p_a)",
    "ngon_c": r"\max_i (p\!\cdot\!n_i) - \cos\tfrac{\pi}{N}",
    "ngon_r": r"\max_i (p\!\cdot\!n_i) - \cos\tfrac{\pi}{N}",
    "star": r"\pm\|p' - \mathrm{proj}\,p'\|",
    "pentagram": r"\mathrm{Star}(5, \tfrac{10}{3})",
    "hexagram": r"\mathrm{Star}(6, 3)",
    "polygon": r"\pm\|p' - \mathrm{proj}\,p'\|",
    "holy_cross": r"R_1 \cup R_2",
    "translation": r"f(p - t)",
    "rotation": r"f(R(p-c) + c)",
    "scale": r"s\,f((p-c)/s + c)",
    "union": r"\min(f_1, f_2)",
    "subtraction": r"\max(f_1, -f_2)",
    "intersection": r"\max(f_1, f_2)",
    "smooth_min": r"-\log_2(2^{-kf_1} + 2^{-kf_2})/k",
    "shell": r"|f| - w",
    "offset": r"f - r",
    "extrude": r"\mathrm{box}(f,\ |p_D| - h)",
    "revolve": r"f(p_0,\ \|p_\perp\| - r)",
    "torus": r"\|(\|p_\perp\|-R,\ p_0)\| - r",
    "product_ball": r"\max_i (\|p_{B_i}\| - r_i)",
    "dodecahedron": r"\max_i (n_i\!\cdot\!p - d_i)",
    "icosahedron": r"\max_i (n_i\!\cdot\!p - d_i)",
    "truncated_octahedron": r"\max_i (n_i\!\cdot\!p - d_i)",
    "rhombic_dodecahedron": r"\max_i (n_i\!\cdot\!p - d_i)",
    "torus_knot": r"\min_i d(p, s_i) - t",
    "cell_24": r"\max_i (n_i\!\cdot\!p - d_i)",
    "cell_120": r"\max_i (n_i\!\cdot\!p - d_i)",
    "cell_600": r"\max_i (n_i\!\cdot\!p - d_i)",
    "duocylinder": r"\max(\|p_{01}\|-r_1,\ \|p_{23}\|-r_2)",
    "clifford_torus": r"\|(\|p_{01}\|-r_1, \|p_{23}\|-r_2)\| - \tfrac{t}{2}",
    "rotation_3d": r"f(R(p-c) + c)",
}

# What the sheet is written in, for the space the grid leaves over: where the
# fields live, how the solids were got out of them, and where the light along a
# crease comes from — the Dirichlet energy density of the normal field, which is
# what separates an edge from a merely tight curve.
PANEL = r"""\begin{aligned}
&p \in \mathbb{R}^{D},\quad f : \mathbb{R}^{D} \to \mathbb{R},\quad
   |f(a) - f(b)| \le L\|a-b\|,\quad \partial S = f^{-1}(0) \\[3pt]
&\text{plot: contours of } f \text{ every } 0.12
   \qquad \text{4D: } f|_{x_3 = w},\quad w = a\cos 2\pi t \\[3pt]
&\text{mesh: } x^{\star} = \arg\min_{x}
   \sum_{i} \big( n_i \!\cdot\! (x - p_i) \big)^{2},
   \quad n_i = \tfrac{\nabla f}{\|\nabla f\|}(p_i) \\[3pt]
&\text{glow: } \varepsilon_i =
   \big\langle \|n_j - n_i\|^{2} \big\rangle_{j \in N(i)}^{1/2}
   \simeq h\,\|\nabla n\|
   \quad \big( \text{flat } 0,\ \text{crease } \theta \big)
\end{aligned}"""


def entries():
    """Every image, in the order stage one lists them — planar first, then the
    meshed ones, which is already a curated grouping."""
    raw = (ROOT / "_raw" / "manifest.txt").read_text().split("\n")
    mesh = (ROOT / "_mesh" / "manifest.txt").read_text().split("\n")
    out = [(l.split()[0], 1) for l in raw if l.strip()]
    out += [(l.split()[0], int(l.split()[2])) for l in mesh if l.strip()]
    return out


def bounds():
    """Cell edges. 1440 does not divide by 7, so the columns are 205 or 206 wide
    and tile exactly rather than leaving a seam or a margin."""
    return [round(i * SIDE / GRID) for i in range(GRID + 1)]


_HLG_CACHE: dict[tuple, tuple] = {}


def hlg(srgb, tmp):
    """An sRGB colour as the HLG signal the sheet is composited in.

    Measured rather than derived: a swatch is written as an sRGB AVIF and pushed
    through the very filter chain the contour plots take, so the labels cannot
    drift away from the images they sit on. Doing the matrix and the OETF by hand
    would be a second implementation to keep in step, and zscale refuses the
    conversion outright for a plain RGB input — the swatch has to be an AVIF.
    """
    if srgb in _HLG_CACHE:
        return _HLG_CACHE[srgb]
    swatch = Path(tmp) / f"swatch_{'_'.join(map(str, srgb))}.avif"
    Image.new("RGB", (16, 16), srgb).save(
        swatch, "AVIF", quality=100, subsampling="4:4:4")
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(swatch), "-frames:v", "1",
         "-vf", "zscale=tin=iec61966-2-1:min=709:pin=709:rin=full:t=linear"
                ",zscale=p=2020:t=arib-std-b67:m=2020_ncl:r=full:npl=203",
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "-"],
        check=True, stdout=subprocess.PIPE,
    )
    a = np.frombuffer(done.stdout, dtype="<u2").reshape(-1, 3).mean(axis=0) / 65535.0
    _HLG_CACHE[srgb] = tuple(float(x) for x in a)
    return _HLG_CACHE[srgb]


_TEX_CACHE: dict[str, Image.Image] = {}


def tex(body, tmp, display=False, dpi=600):
    """A LaTeX fragment as a tight RGBA image, white on transparent.

    Rendered far larger than it is used and scaled down later, which is where the
    labels get their antialiasing. Cached by source, because most of the
    polytopes share one formula and there is no sense running LaTeX forty-seven
    times for a dozen distinct labels.
    """
    key = f"{display}|{dpi}|{body}"
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    math = f"\\[{body}\\]" if display else f"${body}$"
    stem = Path(tmp) / hashlib.sha1(key.encode()).hexdigest()[:16]
    stem.with_suffix(".tex").write_text(
        "\\documentclass[12pt]{article}\n"
        "\\pagestyle{empty}\n"
        "\\usepackage{amsmath,amssymb}\n"
        "\\begin{document}\n" + math + "\n\\end{document}\n",
        encoding="utf-8",
    )
    # Run from inside the temp directory, passing bare filenames. The absolute
    # path here contains a short-name tilde (`BERNKA~1`), and TeX reads `~` as an
    # active character — handed the full path it stops dead asking for another
    # input file.
    run = subprocess.run(
        ["latex", "-interaction=nonstopmode", "-halt-on-error", stem.name + ".tex"],
        cwd=tmp, capture_output=True, text=True,
    )
    if not stem.with_suffix(".dvi").exists():
        sys.exit(f"latex failed for {body!r}\n{run.stdout[-1500:]}")
    subprocess.run(
        ["dvipng", "-D", str(dpi), "-T", "tight", "-bg", "Transparent",
         "-fg", "rgb 1 1 1", "-q", "-o", stem.name + ".png", stem.name + ".dvi"],
        cwd=tmp, check=True, capture_output=True,
    )
    img = Image.open(stem.with_suffix(".png")).convert("RGBA")
    _TEX_CACHE[key] = img
    return img


def fit(img, box_w, box_h):
    """Scale to fit inside the box, never up."""
    k = min(box_w / img.width, box_h / img.height, 1.0)
    if k >= 1.0:
        return img
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))),
                      Image.LANCZOS)


def overlay(items, edges, tmp):
    """The labels and the panel, drawn once: they do not move, so they are
    composited from masks rather than redrawn 128 times."""
    k = SIDE / DESIGN
    px = lambda n: max(1, round(n * k))
    scrim = Image.new("L", (SIDE, SIDE), 0)
    names = Image.new("L", (SIDE, SIDE), 0)
    maths = Image.new("L", (SIDE, SIDE), 0)
    ds, dn = ImageDraw.Draw(scrim), ImageDraw.Draw(names)
    font = ImageFont.truetype(str(ALEF_BOLD), px(14))
    pad, gap = px(6), px(5)

    for idx, (name, _) in enumerate(items):
        col, row = idx % GRID, idx // GRID
        x0, x1, y1 = edges[col], edges[col + 1], edges[row + 1]
        formula = fit(tex(NOTATION[name], tmp),
                      ((x1 - x0) - 2 * pad) * NOTATION_SCALE,
                      px(20) * NOTATION_SCALE)
        # no backing here: the labels sit straight on the field
        dn.text((x0 + pad, y1 - px(18) - formula.height - gap), name,
                font=font, fill=255)
        maths.paste(formula.getchannel("A"), (x0 + pad, y1 - formula.height - gap))

    # whatever the grid leaves over, as one panel — this one keeps its backing,
    # and a rule around it so it reads as a legend rather than a forty-eighth cell
    used = len(items)
    if used < GRID * GRID:
        col, row = used % GRID, used // GRID
        x0, y0, y1 = edges[col], edges[row], edges[row + 1]
        panel = fit(tex(PANEL, tmp, display=True, dpi=900),
                    (SIDE - x0) - px(28), (y1 - y0) - px(28))
        ds.rectangle([x0, y0, SIDE - 1, SIDE - 1], fill=round(255 * PANEL_SCRIM_ALPHA))
        maths.paste(panel.getchannel("A"),
                    (x0 + ((SIDE - x0) - panel.width) // 2,
                     y0 + ((y1 - y0) - panel.height) // 2))
        ImageDraw.Draw(maths).rectangle(
            [x0 + px(3), y0 + px(3), SIDE - 1 - px(3), SIDE - 1 - px(3)],
            outline=255, width=px(2))

    as_mask = lambda im: np.asarray(im, dtype=np.float32) / 255.0
    name_a, math_a = as_mask(names), as_mask(maths)

    glyphs = Image.fromarray((np.maximum(name_a, math_a) * 255).astype(np.uint8))
    shadow = as_mask(glyphs.filter(ImageFilter.GaussianBlur(px(2))))
    shadow = np.clip(shadow * GLYPH_SHADOW, 0.0, 1.0)
    return as_mask(scrim), shadow, name_a, math_a


def decode(name, frames, w, h, hdr):
    """One source as raw 16-bit RGB, in whichever space that sheet is built in.

    Each sheet converts only the half that is not already in its own space, which
    is the whole reason there are two. The HDR sheet lifts the plots into HLG and
    leaves the renders alone; the SDR sheet tone-maps the renders and leaves the
    plots alone — so on either sheet, half the cells are their sources untouched
    and no cell is converted twice.

    An earlier attempt made the SDR sheet by tone-mapping the finished HDR one.
    That put the plots through sRGB -> HLG -> sRGB, and no exposure setting could
    undo it: bright enough to bring white back to white also lifted the dark
    interiors, and the fields came out as flat cyan.
    """
    animated = frames > 1
    # stream 1 is the animation; stream 0 is the primary still
    pre = ["-map", "0:v:1"] if animated else []
    if hdr:
        # sRGB -> scene light -> BT.2020 -> HLG. 203 nits is BT.2408's reference
        # level for graphics composited into HDR; at the default 100 the plots
        # read as grey smudges beside the renders.
        convert = "" if animated else (
            ",zscale=tin=iec61966-2-1:min=709:pin=709:rin=full:t=linear"
            ",zscale=p=2020:t=arib-std-b67:m=2020_ncl:r=full:npl=203"
        )
    else:
        # HLG -> scene light -> BT.709 sRGB, mapping reference white to white and
        # rolling the glow above it off with Hable.
        convert = (
            f",zscale=t=linear:npl={SDR_REFERENCE_NITS},tonemap=hable:desat=0"
            ",zscale=p=709:t=iec61966-2-1:m=709:r=full"
        ) if animated else ""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src_of(name)), *pre,
         "-vf", f"scale={w}:{h}:flags=lanczos{convert}",
         "-pix_fmt", "rgb48le", "-f", "rawvideo", "-"],
        check=True, stdout=subprocess.PIPE,
    )
    return np.frombuffer(done.stdout, dtype="<u2").reshape(-1, h, w, 3)


def src_of(name):
    return ROOT / f"{name}.avif"


def build(items, edges, masks, tmp, hdr):
    """One sheet. `hdr` picks the space everything is converted into, which
    decides the cell conversions, the label colours and the output tagging."""
    out = OUT if hdr else SDR_OUT
    scrim_a, shadow_a, name_a, math_a = masks

    cells = {}
    for idx, (name, frames) in enumerate(items):
        col, row = idx % GRID, idx // GRID
        w, h = edges[col + 1] - edges[col], edges[row + 1] - edges[row]
        data = decode(name, frames, w, h, hdr)
        path = Path(tmp) / f"{name}.{'hdr' if hdr else 'sdr'}.raw"
        data.tofile(path)
        cells[idx] = (path, data.shape)

    if hdr:
        tags = HDR
        # the labels have to be lifted into HLG like any other sRGB colour
        accent, ink = hlg(ACCENT, tmp), hlg(INK, tmp)
        pix = PIX_FMT
    else:
        tags = {"color_primaries": "bt709", "color_trc": "iec61966-2-1",
                "colorspace": "bt709", "color_range": "pc"}
        # already sRGB, so they go down verbatim — which is why white here is
        # white, rather than whatever a tone map happens to leave of it
        accent = tuple(c / 255.0 for c in ACCENT)
        ink = tuple(c / 255.0 for c in INK)
        pix = "yuv444p"

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pixel_format", "rgb48le",
         "-video_size", f"{SIDE}x{SIDE}", "-framerate", str(FPS), "-i", "-",
         "-vf", "setparams=" + ":".join(
             f"{'range' if k == 'color_range' else k}={v}" for k, v in tags.items()),
         "-c:v", "libaom-av1", "-crf", str(CRF), "-cpu-used", str(CPU_USED),
         "-pix_fmt", pix,
         *[a for k, v in tags.items() for a in (f"-{k}", v)],
         "-loop", "0", "-f", "avif", str(out)],
        stdin=subprocess.PIPE,
    )

    loaded = {i: np.fromfile(p, dtype="<u2").reshape(s) for i, (p, s) in cells.items()}
    layers = ((scrim_a, (PANEL_SCRIM,) * 3), (shadow_a, (0.0, 0.0, 0.0)),
              (name_a, accent), (math_a, ink))
    for f in range(128):
        canvas = np.zeros((SIDE, SIDE, 3), dtype=np.float32)
        for idx in range(len(items)):
            col, row = idx % GRID, idx // GRID
            src = loaded[idx]
            canvas[edges[row]:edges[row + 1], edges[col]:edges[col + 1]] = \
                src[f % src.shape[0]]
        for mask, rgb in layers:
            a = mask[..., None]
            canvas = canvas * (1 - a) + np.asarray(rgb, np.float32) * 65535.0 * a
        enc.stdin.write(np.clip(canvas, 0, 65535).astype("<u2").tobytes())
        if f % 64 == 0:
            print(f"    frame {f}/128", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    return out


def main():
    items = entries()
    edges = bounds()
    print(f"{len(items)} images into {GRID}x{GRID}, {SIDE}px, 128 frames")

    with tempfile.TemporaryDirectory() as tmp:
        print("  typesetting...", flush=True)
        masks = overlay(items, edges, tmp)
        print(f"  {len(_TEX_CACHE)} distinct LaTeX fragments", flush=True)

        for hdr in (True, False):
            print(f"  building {'HDR' if hdr else 'SDR'} sheet...", flush=True)
            out = build(items, edges, masks, tmp, hdr)
            mib = out.stat().st_size / 1024 / 1024
            print(f"  {out.name}  {mib:.2f} MiB", flush=True)

    if OUT.stat().st_size / 1024 / 1024 > 8:
        print(f"  !! over the 8 MiB budget — raise CRF (currently {CRF})")


if __name__ == "__main__":
    sys.exit(main())
