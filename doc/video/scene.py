"""Starter & verification scene for the *space-filling* explainer video.

The animations described in ``script.md`` are built with Manim Community
Edition (https://docs.manim.community/). This module holds a minimal scene
used to confirm the toolchain renders video end-to-end.

Fast 480p preview::

    uv run manim -ql scene.py Verify

Final 1080p60::

    uv run manim -qh scene.py Verify

Output is written to ``./media/videos/``. ``Tex`` / ``MathTex`` (LaTeX
formulas) need a LaTeX distribution -- see README.md.
"""

from manim import *


class Verify(Scene):
    """End-to-end pipeline check -- uses only Pango text, no LaTeX.

    A stack of shrinking circles evokes the distance-field packing this
    library produces. If this renders to an ``.mp4``, then cairo and ffmpeg
    are wired up correctly.
    """

    def construct(self) -> None:
        title = Text("space-filling").scale(1.3).to_edge(UP)
        subtitle = Text("manim toolchain OK", color=GREY_B).scale(0.5)
        subtitle.next_to(title, DOWN)

        radii = [2.2, 1.5, 1.0, 0.65, 0.4]
        colors = [BLUE_E, BLUE_D, TEAL, GREEN, YELLOW]
        circles = VGroup(
            *(
                Circle(radius=r, color=c, stroke_width=4)
                for r, c in zip(radii, colors)
            )
        ).move_to(ORIGIN)

        self.play(Write(title), FadeIn(subtitle, shift=DOWN))
        self.play(
            LaggedStart(*(Create(c) for c in circles), lag_ratio=0.25),
            run_time=2.0,
        )
        self.play(circles.animate.set_fill(opacity=0.15), run_time=1.0)
        self.wait(0.5)
