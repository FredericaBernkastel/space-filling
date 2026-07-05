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

```sh
uv run manim -ql scene.py Verify     # 480p15, fast preview
uv run manim -qh scene.py Verify     # 1080p60, final
uv run manim -qh -p scene.py Verify  # render, then play the result
```

Rendered files are written to `media/videos/`.

## LaTeX (for formulas)

`Tex` / `MathTex` mobjects require a LaTeX distribution, which is **not**
bundled with manim. On Windows, install MiKTeX:

```powershell
winget install MiKTeX.MiKTeX
```

Then re-run `uv run manim checkhealth` — the `latex` and `dvisvgm` checks
should pass. Until then, use `Text` (Pango) mobjects instead of `Tex`.
