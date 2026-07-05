"""Scene 1 — cold open.

Four beats: the governing equation (1); a scroll through the rendered README
"Implementation" section; a scroll through Bourke's 2011 page (exported to
PDF); and the *Million-Circle Fractal* zoom-out that motivates the project.

The README and webpage beats show pre-rendered assets:
  - assets/readme.md.png              (given)
  - assets/derived/bourke_webpage.png (built by build_assets.py from the PDF)

Render:
    uv run manim -ql scene01_intro.py Scene01Intro
"""

from manim import *

from theme import (
    VideoScene, INK, MUTED, ACCENT, COOL, FIELD_HI, asset,
    FS_TITLE, FS_BODY, FS_CAPTION,
)


class Scene01Intro(VideoScene):
    def construct(self) -> None:
        self.beat_equation()
        self.beat_readme()
        self.beat_bourke_webpage()
        self.beat_million_circles()

    # ------------------------------------------------------------------ #
    def beat_equation(self) -> None:
        """'Liserotte here.' + equation (1): the arg-max / min objective."""
        signature = Text("Liserotte", font_size=FS_CAPTION, color=MUTED, slant=ITALIC)
        signature.to_corner(DR, buff=0.4)
        self.play(FadeIn(signature, shift=UP * 0.2))

        eq = MathTex(
            r"\vec{x}^{*}",                    # 0
            r"=",                              # 1
            r"\arg\max_{\vec v \in \Omega}",   # 2
            r"\min_{n}",                       # 3
            r"\mathrm{sdf}_{n}(\vec v)",       # 4
            color=INK,
        ).scale(0.95)

        number = self.eq_number(1, eq)

        self.play(Write(eq), run_time=2.2)
        self.play(FadeIn(number))
        self.wait(0.4)

        self.play(
            eq[2].animate.set_color(ACCENT),
            eq[3].animate.set_color(COOL),
            run_time=1.0,
        )
        self.wait(1.2)

        goal = Tex(
            r"solvable --- but in $O(\log N)$ time, $O(N)$ memory?",
            color=MUTED,
        ).scale(0.62)
        goal.next_to(eq, DOWN, buff=1.0)
        self.play(FadeIn(goal, shift=UP * 0.2))
        self.wait(2.0)

        self.play(
            LaggedStart(
                FadeOut(goal, shift=DOWN * 0.2),
                FadeOut(number),
                FadeOut(eq, shift=UP * 0.3),
                FadeOut(signature),
                lag_ratio=0.12,
            )
        )

    # ------------------------------------------------------------------ #
    def _scroll_image(self, path, width, run_time, framed=False, chip_label=None):
        """Fade in a tall image aligned to the top edge, scroll it up, fade out."""
        img = ImageMobject(path)
        img.width = width
        top_gap = 0.4
        img.move_to([0, (config.frame_height / 2 - top_gap) - img.height / 2, 0])

        grp = Group(img)
        if framed:
            border = SurroundingRectangle(img, color=MUTED, buff=0.0).set_stroke(width=1.5)
            grp.add(border)
        chip = self.source_chip(chip_label) if chip_label else None

        intro = [FadeIn(grp)]
        if chip:
            intro.append(FadeIn(chip))
        self.play(*intro, run_time=0.8)

        scroll = max(img.height - config.frame_height + 2 * top_gap, 0.3)
        self.play(grp.animate.shift(UP * scroll), run_time=run_time, rate_func=linear)
        self.wait(0.3)

        outro = [FadeOut(grp)]
        if chip:
            outro.append(FadeOut(chip))
        self.play(*outro)

    def beat_readme(self) -> None:
        """Scroll the rendered README 'Implementation' section (framed page)."""
        self._scroll_image(asset("readme.md.png"), width=7.0, run_time=7.5,
                           framed=True, chip_label="readme.md")

    def beat_bourke_webpage(self) -> None:
        """Scroll Bourke's 2011 page, exported to PDF and rasterized to a strip."""
        self._scroll_image(asset("derived/bourke_webpage.png"), width=7.6, run_time=8.0,
                           framed=False, chip_label="paulbourke.net/fractals/randomtile")

    # ------------------------------------------------------------------ #
    def beat_million_circles(self) -> None:
        """Zoom out of the fractal, hide it, then title / stat / question on a clean bg."""
        img = ImageMobject(asset("bourke_zoom.png"))
        img.height = config.frame_height
        if img.width > config.frame_width:
            img.width = config.frame_width
        img.move_to(ORIGIN)

        self.camera.frame.save_state()
        self.camera.frame.scale(0.28).move_to(img.get_center() + RIGHT * 1.2 + UP * 0.8)
        self.add(img)
        self.wait(0.5)
        self.play(Restore(self.camera.frame), run_time=6, rate_func=smooth)
        self.wait(0.5)
        self.play(FadeOut(img))  # hide the image before any text

        title = Text("A Million-Circle Fractal", font_size=FS_TITLE, color=INK, weight=BOLD)
        stat = Text("generation time: 14.7 hours", font_size=FS_BODY, color=FIELD_HI)
        group = VGroup(title, stat).arrange(DOWN, buff=0.35).move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP * 0.2))
        self.play(FadeIn(stat, shift=UP * 0.1))
        self.wait(1.8)
        self.play(FadeOut(group))

        question = Text("Can we do better?", font_size=38, color=ACCENT, weight=BOLD)
        self.play(Write(question))
        self.wait(2.0)
        self.play(FadeOut(question))
