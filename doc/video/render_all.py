"""Render every scene at full quality — QHD 2560x1440 @ 60 fps — 4 in parallel.

Each scene renders to its own mp4 (no concatenation): the originals land in
``media/videos/<module>/1440p60/`` and an ordered copy of each is collected
into ``media/final/NN_SceneName.mp4`` for the compositing timeline. Per-scene
manim output goes to ``media/logs/<module>.log``.

Caching is disabled throughout: manim's partial-movie cache does not reliably
detect ImageMobject pixel changes (see theme/fields history), and a final
render must never reuse a stale frame.

Usage (from doc/video; latex + ffmpeg must be on PATH):

    uv run python render_all.py            # all scenes
    uv run python render_all.py 3 5 8      # only scenes 03, 05, 08
    uv run python render_all.py --list     # show the render plan and exit
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
FINAL = MEDIA / "final"
LOGS = MEDIA / "logs"

WORKERS = 4
QUALITY = "-qp"          # 2560x1440 @ 60 fps -> media/videos/<module>/1440p60/
QUALITY_DIR = "1440p60"


def discover() -> list[tuple[Path, str]]:
    """(file, scene class) for every sceneNN_*.py defining a VideoScene."""
    scenes = []
    for f in sorted(HERE.glob("scene*.py")):
        m = re.search(r"^class (\w+)\(VideoScene\)", f.read_text(encoding="utf-8"), re.M)
        if m:
            scenes.append((f, m.group(1)))
    return scenes


_print_lock = Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def render(job: tuple[Path, str]) -> tuple[str, bool, float]:
    file, cls = job
    log_path = LOGS / f"{file.stem}.log"
    t0 = time.time()
    log(f"[start] {file.stem} :: {cls}")
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [sys.executable, "-m", "manim", QUALITY, "--disable_caching", file.name, cls],
            cwd=HERE, stdout=lf, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    out = MEDIA / "videos" / file.stem / QUALITY_DIR / f"{cls}.mp4"
    ok = proc.returncode == 0 and out.exists()
    if ok:
        FINAL.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, FINAL / f"{file.stem[5:7]}_{cls}.mp4")
        log(f"[done ] {file.stem}  ({dt / 60:.1f} min)")
    else:
        log(f"[FAIL ] {file.stem}  ({dt / 60:.1f} min)  exit {proc.returncode} — see {log_path}")
    return file.stem, ok, dt


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--list"]
    scenes = discover()
    if args:
        wanted = {f"{int(a):02d}" for a in args}
        scenes = [s for s in scenes if s[0].stem[5:7] in wanted]
    if not scenes:
        print("no scenes matched")
        return 1
    print(f"render plan ({len(scenes)} scenes, {WORKERS}-way parallel, {QUALITY} + --disable_caching):")
    for f, cls in scenes:
        print(f"  {f.stem[5:7]}  {f.name:28s} {cls}")
    if "--list" in sys.argv:
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(render, scenes))

    print(f"\ntotal wall time: {(time.time() - t0) / 60:.1f} min")
    print(f"separate mp4s collected in: {FINAL}")
    failures = [name for name, ok, _ in results if not ok]
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
