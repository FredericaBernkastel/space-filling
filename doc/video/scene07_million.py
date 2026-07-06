"""Scene 7 — one million circles.

Two beats, both playing pre-rendered footage (built by ``build_assets.py``):
(1) the real `02_random_distribution` run driven to 1M circles at ADF depth 10
(``render/src/bin/random_distribution.rs`` streamed to mp4), with a live
insertion counter synced to the binary's geometric frame schedule; (2) a baked
zoom/pan over the final 8192^2 render (assets/1M.png), closed by the quoted
figures: 49 s on 4 cores, 113k nodes, 76 MiB.

Render:
    uv run manim -ql scene07_million.py Scene07Million
"""

import numpy as np
from manim import *

from theme import (
    VideoScene, asset, mono, INK, MUTED, ACCENT,
    FS_TITLE, FS_H2, FS_BODY, FS_CAPTION, FS_CHIP,
)
from video import load_frames, VideoMobject

# mirror of the emission schedule in render/src/bin/random_distribution.rs:
# frame k is emitted once the insertion count first reaches FIRST * RATIO^k
COUNT, FRAMES, FIRST = 1_000_000, 330, 4.0
RATIO = (COUNT / FIRST) ** (1.0 / FRAMES)


class Scene07Million(VideoScene):
    def construct(self) -> None:
        self.beat_fill()
        self.beat_zoom()

    # ------------------------------------------------------------------ #
    def beat_fill(self) -> None:
        """The 1M-circle fill, embedded; a live counter follows the schedule."""
        title = Text("so — how long does it take to make “A Million-Circle Fractal”?",
                     font_size=FS_H2, color=INK).to_edge(UP, buff=0.4)

        frames = load_frames(asset("derived/random_distribution.mp4"))
        video = VideoMobject(frames).set_height(5.9)
        vbox = SurroundingRectangle(video, color=MUTED, buff=0.0).set_stroke(width=1.5)
        panel = Group(video, vbox).move_to(np.array([-2.2, -0.35, 0.0]))
        chip = self.source_chip("examples/gd_adf/02_random_distribution.rs — ADF max depth 10, batch 32")

        def fmt(n: int) -> str:
            return f"{n:,}".replace(",", " ")

        def counter_text(n: int) -> Text:
            return mono(fmt(n), font_size=40, color=ACCENT)

        counter_lbl = Text("circles inserted", font_size=FS_CAPTION, color=MUTED)
        counter = counter_text(0)
        target = mono("target: 1 000 000", font_size=FS_CHIP, color=MUTED)
        readout = VGroup(counter_lbl, counter, target).arrange(DOWN, aligned_edge=RIGHT, buff=0.3)
        readout.to_edge(RIGHT, buff=1.0).shift(UP * 0.6)
        # pin the units digit: the number grows leftward, odometer-style
        counter_anchor = counter.get_right().copy()

        prog = ValueTracker(0.0)
        n = video.n_frames

        def count_at(alpha: float) -> int:
            k = alpha * (n - 1)
            if k >= n - 2:  # the final emitted frame is the full million
                return COUNT
            return min(int(np.ceil(FIRST * RATIO ** k)), COUNT)

        self.play(FadeIn(title), FadeIn(video), Create(vbox), FadeIn(chip),
                  FadeIn(readout))
        video.add_updater(lambda m: m.set_frame(prog.get_value() * (m.n_frames - 1)))
        counter.add_updater(lambda m: m.become(
            counter_text(count_at(prog.get_value())).move_to(counter_anchor, aligned_edge=RIGHT)))
        self.play(prog.animate.set_value(1.0), run_time=n / 30.0, rate_func=linear)
        video.clear_updaters()
        counter.clear_updaters()
        counter.become(counter_text(COUNT).move_to(counter_anchor, aligned_edge=RIGHT))
        self.wait(0.8)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])

    # ------------------------------------------------------------------ #
    def beat_zoom(self) -> None:
        """The final 8192^2 render, panned and zoomed (baked clip); the verdict."""
        frames = load_frames(asset("derived/million_zoom.mp4"))
        video = VideoMobject(frames).set_height(7.1).move_to(np.array([-0.9, 0.1, 0.0]))
        vbox = SurroundingRectangle(video, color=MUTED, buff=0.0).set_stroke(width=1.5)
        chip = self.source_chip("assets/1M.png — the finished render, 8192×8192")

        prog = ValueTracker(0.0)
        self.play(FadeIn(video), Create(vbox), FadeIn(chip))
        video.add_updater(lambda m: m.set_frame(prog.get_value() * (m.n_frames - 1)))
        self.play(prog.animate.set_value(1.0), run_time=video.n_frames / 30.0, rate_func=linear)
        video.clear_updaters()

        # the quoted figures: a right-aligned column beside the artwork
        def stat(value: str, label: str) -> VGroup:
            v = mono(value, font_size=FS_TITLE, color=ACCENT)
            l = Text(label, font_size=FS_CHIP, color=MUTED)
            return VGroup(v, l).arrange(DOWN, buff=0.14, aligned_edge=RIGHT)

        stats = VGroup(
            stat("49 s", "on a 4-core machine"),
            stat("113 k", "ADF tree nodes"),
            stat("76 MiB", "total size"),
        ).arrange(DOWN, buff=0.7, aligned_edge=RIGHT)
        stats.to_edge(RIGHT, buff=0.8).set_y(0.1)

        self.play(LaggedStart(*[FadeIn(s, shift=LEFT * 0.2) for s in stats],
                              lag_ratio=0.22), run_time=1.6)
        self.wait(2.6)
        self.play(*[FadeOut(m) for m in list(self.mobjects)])
