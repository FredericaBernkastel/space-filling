"""Scene 10 — outro.

A quiet close over a dim circle packing: the thank-you lines, then an end card
(crate name, repository, and the narrator's signature — bookending Scene 1's
cold open). The narration hands off to the assorted development footage.

Render:
    uv run manim -ql scene10_outro.py Scene10Outro
"""

import random

import numpy as np
from manim import *

from theme import (
    VideoScene, INK, MUTED, ACCENT, COOL, mono,
    FS_TITLE, FS_H2, FS_BODY, FS_CAPTION, FS_CHIP,
)


def dim_packing(seed=11, count=52, xr=6.6, yr=3.6):
    """A faint random circle packing — the subject, whispering in the background."""
    random.seed(seed)
    packed = []
    for _ in range(30000):
        if len(packed) >= count:
            break
        r = 0.25 + 1.15 * (random.random() ** 2)
        x = random.uniform(-xr + r, xr - r)
        y = random.uniform(-yr + r, yr - r)
        if all((x - px) ** 2 + (y - py) ** 2 > (r + pr + 0.03) ** 2 for px, py, pr in packed):
            packed.append((x, y, r))
    return VGroup(*[
        Circle(radius=r, stroke_width=1.2)
        .set_stroke(COOL, opacity=0.22).set_fill(COOL, opacity=0.035)
        .move_to([x, y, 0])
        for x, y, r in packed
    ])


class Scene10Outro(VideoScene):
    def construct(self) -> None:
        packing = dim_packing()
        self.play(LaggedStart(*[GrowFromCenter(c) for c in packing], lag_ratio=0.02),
                  run_time=2.4)

        # the thank-you lines
        l1 = Text("I hope you've learnt something new.", font_size=FS_H2, color=INK)
        l2 = Text("Excited to see what kind of art you will make — feel free to share!",
                  font_size=FS_BODY, color=ACCENT)
        lines = VGroup(l1, l2).arrange(DOWN, buff=0.45).move_to(UP * 0.3)
        self.play(FadeIn(l1, shift=UP * 0.15))
        self.play(FadeIn(l2, shift=UP * 0.15))
        self.wait(2.2)

        # hand-off to the closing footage reel
        reel = Text("up next: assorted visuals from the project's development — enjoy!",
                    font_size=FS_CAPTION, color=MUTED)
        reel.next_to(lines, DOWN, buff=0.8)
        self.play(FadeIn(reel))
        self.wait(1.8)
        self.play(FadeOut(lines), FadeOut(reel))

        # end card: crate, repository, signature (bookends the cold open)
        name = Text("space-filling", font_size=FS_TITLE, color=INK, weight=BOLD)
        tagline = Text("generalized random space filling of the plane, driven by signed distance fields",
                       font_size=FS_CHIP, color=MUTED)
        repo = mono("github.com/FredericaBernkastel/space-filling",
                    font_size=FS_CAPTION, color=COOL)
        card = VGroup(name, tagline, repo).arrange(DOWN, buff=0.32).move_to(UP * 0.15)
        signature = Text("Liserotte", font_size=FS_CAPTION, color=MUTED, slant=ITALIC)
        signature.to_corner(DR, buff=0.4)

        self.play(FadeIn(name, shift=UP * 0.1))
        self.play(FadeIn(tagline), FadeIn(repo))
        self.play(FadeIn(signature, shift=UP * 0.2))
        self.wait(2.6)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=1.4)
