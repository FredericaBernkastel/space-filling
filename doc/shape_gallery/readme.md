# Regenerating doc/shapes

Three stages, 128 frames a shape. Both stages 1 and 2 have to be re-run for a frame
count change: the 4D shapes are meshed once *per frame*, so their sweeps live in
stage 1's output.

Budget for it — roughly 45 minutes, 7½ hours and 40 minutes respectively, and about
21 GB of intermediates.

### 1. Mesh the fields (Rust)

```
cd doc/shape_gallery && cargo run --release
```

Evaluates every shape's real SDF. Planar ones become rasters of field values in
`doc/shapes/_raw`; the rest are dual-contoured into OBJ meshes in `doc/shapes/_mesh`.
Add names to do only some of them — `cargo run --release -- dodecahedron gyroid` —
which merges into the manifests rather than truncating them.

About 45 minutes, nearly all of it the five sliced 4D shapes: each is meshed 128
times where a solid is meshed once. That is also where the disk goes — a section is
~30 MB of OBJ, so those five come to some 20 GB. `MORPH_RES` is the dial if that is
too much. A sweep clears its own previous files first, so a changed frame count
cannot leave the tail of an older one behind.

### 2. Render (Blender)

Renders all 16 animations, 2048 frames, to `doc/shapes/_frames/<name>/NNN.avif`:

```
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b --python C:\git\rust\space-filling\doc\shape_gallery\render.py
```

Working directory doesn't matter — the script resolves its paths from its own
location. Roughly 13 s a frame at the defaults (1024², 40 samples), so about 7½
hours for the set. `--samples` is the lever if that is too long.

Frames are 10-bit AVIF through an ACES view transform onto a Rec.2100 HLG display.
The bloom is a Glare node in the compositor, so it works on scene-linear values
where the creases genuinely exceed white — `BLOOM` in `render.py` is the dial, and
those numbers are mine rather than read out of `scene.blend`, so they are the first
thing to change if the glow wants tuning.

Useful flags: `--only dodecahedron,gyroid` to pick entries, `--max-frames 1` for one
frame each, `--samples`, `--res`, `--no-bloom` to isolate the compositor, and
`--blend <path>` to save the scene for opening by hand.

```
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b --python C:\git\rust\space-filling\doc\shape_gallery\render.py -- --only gyroid --max-frames 1
```

### 3. Encode (Python + ffmpeg)

```
cd C:\git\rust\space-filling\doc\video; uv run python ..\shape_gallery\encode.py
```

Colours the planar rasters into still AVIF, and muxes each frame directory into one
animated AVIF — 512², 4:4:4 10-bit, HLG, looping. The `cd` is only because that is
where the uv environment lives; the script finds its own paths.

Around 40 minutes, most of it libaom; `CPU_USED` trades quality for speed. Expect
roughly half a megabyte an animation, so 8–10 MB for the set — extrapolated from a
short test rather than measured, and probably pessimistic, since a slow turntable
predicts well between frames.

**Frame rate.** 128 frames at `FPS = 12.5` is a 10.2 s loop — a quarter of the speed
the 32-frame version turned at. If the intention was a smoother turn rather than a
slower one, `FPS = 50` in `encode.py` restores the old rotation speed at four times
the smoothness. Frame count drives file size either way; the rate does not.

Nothing in Python touches the rendered pixels: every image library to hand decodes
them to 8 bits, which would discard the range they were rendered for. ffmpeg does
the downscale and the container, and the HDR tags are re-stamped on the way out
because raw video carries none.

### Afterwards

The previous generation of images was WebP. Once stage 3 has written the full set,
those are dead:

```
git rm doc/shapes/*.webp
```
