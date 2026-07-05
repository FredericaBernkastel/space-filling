"""Play an mp4 inside a manim scene.

Manim has no native video mobject, so we decode frames with PyAV (a manim
dependency) and blit them through an ImageMobject driven by an updater. Used to
embed Rust example renders (e.g. Scene 3's fractal fill).
"""

from __future__ import annotations

import av
import numpy as np
from manim import ImageMobject
from PIL import Image


def load_frames(path, size=None):
    """Decode ``path`` into a list of HxWx4 uint8 (RGBA) frames."""
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgba")
            if size is not None:
                arr = np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))
            frames.append(np.ascontiguousarray(arr))
    return frames


class VideoMobject(ImageMobject):
    """An ImageMobject that can swap to any decoded frame (drive via an updater)."""

    def __init__(self, frames, **kwargs):
        self._frames = frames
        super().__init__(frames[0], **kwargs)

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def set_frame(self, idx):
        i = int(np.clip(idx, 0, len(self._frames) - 1))
        self.pixel_array = self._frames[i]
        return self
