# Regenerating doc/shapes

Three stages. Stage 1 is done and its output (`doc/shapes/_mesh`, 4.3 GB) is current,
so a re-render starts at stage 2.

### 1. Mesh the fields (Rust)

```
cd doc/shape_gallery && cargo run --release
```

Evaluates every shape's real SDF. Planar ones become rasters of field values in
`doc/shapes/_raw`; the rest are dual-contoured into OBJ meshes in `doc/shapes/_mesh`.
Takes about ten minutes. Add names to do only some of them — `cargo run --release --
dodecahedron gyroid` — which merges into the manifests rather than truncating them.

### 2. Render (Blender)

Renders all 16 animations, 452 frames, to `doc/shapes/_frames/<name>/NNN.avif`:

```
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b --python C:\git\rust\space-filling\doc\shape_gallery\render.py
```

Working directory doesn't matter — the script resolves its paths from its own
location. Roughly 13 s a frame at the defaults (1024², 40 samples), so about 100
minutes for the set.

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
