# space-filling — explainer video

A [Manim Community Edition](https://docs.manim.community/) project that
animates the algorithms behind the `space-filling` crate. The narration and
shot list live in [`script.md`](script.md); source imagery is in
[`assets/`](assets).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From this
directory:

```sh
uv sync                    # create .venv and install manim + deps from uv.lock
uv run manim checkhealth   # verify the install
```

## Rendering

Pre-rendered assets (PDF strips, Rust example footage) are built once with:

```sh
uv run python build_assets.py
```

Preview a single scene (fast, 480p15):

```sh
uv run manim -ql scene05_algebra.py Scene05Algebra
```

Full-quality final (QHD 2560×1440 @ 60 fps, 4 scenes in parallel, caching
disabled; each scene stays a separate mp4):

```sh
uv run python render_all.py            # all scenes
uv run python render_all.py 3 5 8      # a subset
uv run python render_all.py --list     # show the plan only
```

Originals land in `media/videos/<scene>/1440p60/`; ordered copies are
collected in `media/final/NN_SceneName.mp4`, logs in `media/logs/`.

The thumbnail (a single still):

```sh
uv run manim -r 1920,1080 -s thumbnail.py Thumbnail
```

## LaTeX (for formulas)

`Tex` / `MathTex` mobjects require a LaTeX distribution, which is **not**
bundled with manim. On Windows, install MiKTeX:

```powershell
winget install MiKTeX.MiKTeX
```

Then re-run `uv run manim checkhealth` — the `latex` and `dvisvgm` checks
should pass. Until then, use `Text` (Pango) mobjects instead of `Tex`.
