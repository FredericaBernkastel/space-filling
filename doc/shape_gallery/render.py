"""Stage two: render the meshed shapes in Blender, one PNG per frame.

    blender -b --python doc/shape_gallery/render.py -- [options]

Reads `doc/shapes/_mesh/manifest.txt` (written by the Rust stage) and writes
`doc/shapes/_frames/<name>/NNN.avif`, which `encode.py` muxes into animated AVIF.
Rendering is at double the final size and `encode.py` scales down: the meshes alias
acute creases by about one grid cell, and supersampling puts that under a pixel
instead of showing it as a stair-step. Two kinds of entry:

  solid  one mesh, turntabled through a full 2π — the loop closes by construction
  morph  one mesh per frame, already a closed cosine cycle of 4D cross-sections,
         rendered under a camera framed on the *largest* frame, so the section is
         seen to grow and shrink rather than being re-zoomed each frame

Lighting is a sky and nothing else — no lamps. A Nishita sky with the sun near the
horizon supplies every bit of it, which suits these shapes: an environment lights a
translucent body evenly from all round instead of carving it into a lit half and a
dark one, and being an environment rather than a lamp it is independent of how far
away anything is, so one rig serves every shape and only the camera distance
changes. `SKY` and `BACKDROP` were arrived at by hand in Blender and read back out
of the file; the numbers here are that file.

Output is 10-bit HDR: an ACES view transform onto a Rec.2100 HLG display, written
as AVIF. That choice is what puts the bloom in the compositor rather than in
`encode.py`. Post-processing in Python would mean decoding HLG to work on it and
re-encoding afterwards, through a library that hands back 8 bits — so the range the
HDR output exists for would be thrown away to compute a glow. The compositor runs
*before* the view transform, on scene-linear values, where the emissive creases
really are several times over white and a bloom spreads the light they actually
emit. That is also simply the right place for it: the former version had to
threshold pixels that had already been clipped to 1.0.

Colour constants are given as sRGB display values and converted to linear, because
Blender wants linear and the palette this started from — `doc/video/fields.py`,
shared with the explainer video — is written in display values.

The material is translucent, with its creases emitting. The intensity is read from
the mesh, not computed from the view: the Rust stage bakes a per-vertex crease
measure into the OBJ's vertex colour, and `build_material` maps it to emission and
to opacity. See that function for why a Fresnel rim would be the wrong thing.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# --- appearance ----------------------------------------------------------
# Two conventions, labelled, because the constants come from two places. Ones
# inherited from `doc/video/fields.py` are sRGB display values, as that file writes
# them straight into PNG bytes, and go through `rgba`. Ones chosen by hand in
# Blender are already linear — that is how Blender stores a colour — and go through
# `linear_rgba` unchanged, so what is written here is exactly what the file holds
# rather than a value that has been round-tripped through a gamma curve.

# The shape itself: a desaturated blue, dark enough that the creases carry the
# picture. sRGB (0.400, 0.400, 0.592).
SURFACE_LINEAR = (0.13319, 0.13319, 0.30926)
# What the creases emit — the top of the ramp, a shade warmer.
GLOW = (1.000, 0.930, 0.740)
# What the camera sees behind the shape: flat neutral grey, no sky gradient. The
# sky is for lighting only, and a plain field keeps these readable as diagrams.
# sRGB (0.138, 0.138, 0.138).
BACKDROP_LINEAR = (0.01704, 0.01704, 0.01704)
# How see-through the body is away from any crease, and how hard the creases burn.
BODY_ALPHA = 0.60
GLOW_STRENGTH = 4.5
# Exponent applied to the crease measure before it becomes emission. Above 1 it
# narrows the burning core and leaves the dilated skirt as a soft halo, which is
# what stops every edge reading as one fat white stripe.
GLOW_FALLOFF = 3.0

# Nishita sky, sun just under two degrees above the horizon: a warm, low, wrapping
# light. Every value read back out of the hand-edited scene.blend.
SKY = {
    "sky_type": "MULTIPLE_SCATTERING",
    "sun_disc": True,
    "sun_size": 0.009512044489383698,
    "sun_intensity": 1.0,
    "sun_elevation": 0.03316127136349678,
    "sun_rotation": 0.0,
    "altitude": 100.0,
    "air_density": 1.0,
    "aerosol_density": 5.199999809265137,
    "ozone_density": 0.20000000298023224,
    # ignored while sky_type is MULTIPLE_SCATTERING, but part of the file
    "turbidity": 2.200000047683716,
    "ground_albedo": 0.30000001192092896,
}

# Bloom, in the compositor. `THRESHOLD` is in scene-linear light, where the body
# sits below 1 and the creases emit several times it, so a threshold of 1 selects
# the emitters and nothing else — no tuning against a tone-mapped image needed.
BLOOM = {
    "Type": "Bloom",
    "Quality": "High",
    "Threshold": 1.0,
    "Smoothness": 0.4,
    "Strength": 0.55,
    "Size": 0.4,
}

# Colour management. HLG carries the creases as genuinely bright rather than merely
# white, which is the whole point of taking the output to 10 bits.
DISPLAY_DEVICE = "Rec.2100-HLG"
VIEW_TRANSFORM = "ACES 2.0 - HDR 1000 nits"
LOOK = "None"

# where the camera sits, as a direction from the origin
CAM_DIR = (0.95, -1.0, 0.55)
CAM_LENS = 50.0


def srgb_to_linear(c):
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


def rgba(c):
    """An sRGB display colour, as Blender wants it."""
    return (*srgb_to_linear(c), 1.0)


def linear_rgba(c):
    """A colour already in linear light, i.e. read back out of Blender."""
    return (*c, 1.0)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(prog="render.py")
    ap.add_argument("--mesh-dir", type=Path, default=here.parent / "shapes" / "_mesh")
    ap.add_argument("--out", type=Path, default=here.parent / "shapes" / "_frames")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--only", default=None, help="comma-separated names to render")
    ap.add_argument("--blend", type=Path, default=None, help="save the scene here")
    ap.add_argument("--no-bloom", action="store_true",
                    help="skip the compositor, for isolating it")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="render at most this many frames per entry, for iterating")
    return ap.parse_args(argv)


def build_material():
    """A translucent body whose creases light up from the inside.

    Everything about the glow comes from the mesh's own `Color` attribute, which
    the Rust stage filled with a per-vertex crease intensity: the deviation of the
    surface normal over one cell, so flat facets read 0, a tight tube reads a
    little, and an edge or corner saturates. Nothing here consults the view
    direction — a Fresnel-driven rim would trace the silhouette, which moves as
    the shape turns and says nothing about the shape itself, whereas these lines
    are welded to the geometry and rotate with it.

    Being see-through is what makes them worth having: the far side of the shape
    shows through the near side, so a polytope reads as its whole edge graph
    rather than the three faces pointing at the camera.
    """
    mat = bpy.data.materials.new("shape")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = linear_rgba(SURFACE_LINEAR)
    bsdf.inputs["Roughness"].default_value = 0.16
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Emission Color"].default_value = rgba(GLOW)
    # No clearcoat: a second specular lobe on a body this dark reads as a smear of
    # highlight over the crease lines, which are the thing worth looking at.
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.0

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "Color"       # what the OBJ importer names vertex colour
    attr.location = (-800, 0)
    edge = attr.outputs["Fac"]          # the mean of R, G, B: the scalar as written

    # Bias toward the sharp end, so a crease blazes while merely-curved stays a
    # suggestion, then scale into an emission strength.
    bias = nt.nodes.new("ShaderNodeMath")
    bias.operation, bias.location = "POWER", (-600, -150)
    bias.inputs[1].default_value = GLOW_FALLOFF
    strength = nt.nodes.new("ShaderNodeMath")
    strength.operation, strength.location = "MULTIPLY", (-420, -150)
    strength.inputs[1].default_value = GLOW_STRENGTH

    # Opacity rides the same signal: the body is translucent, the creases are not,
    # which keeps the glow from being diluted by everything behind it.
    opacity = nt.nodes.new("ShaderNodeMapRange")
    opacity.location = (-420, 150)
    opacity.inputs["To Min"].default_value = BODY_ALPHA
    opacity.inputs["To Max"].default_value = 1.0

    # Opaque to shadow rays. A shadow ray that has to walk every translucent layer
    # between the surface and each of three suns is where the cost of this material
    # actually goes — with it left transparent a single gyroid frame took minutes.
    # Terminating them at the first hit also gives the shape a firmer core, which
    # it needs, having no opaque interior of its own.
    shadow = nt.nodes.new("ShaderNodeLightPath")
    shadow.location = (-800, 320)
    solid = nt.nodes.new("ShaderNodeMath")
    solid.operation, solid.location = "MAXIMUM", (-220, 150)

    nt.links.new(edge, bias.inputs[0])
    nt.links.new(bias.outputs["Value"], strength.inputs[0])
    nt.links.new(strength.outputs["Value"], bsdf.inputs["Emission Strength"])
    nt.links.new(edge, opacity.inputs["Value"])
    nt.links.new(opacity.outputs["Result"], solid.inputs[0])
    nt.links.new(shadow.outputs["Is Shadow Ray"], solid.inputs[1])
    nt.links.new(solid.outputs["Value"], bsdf.inputs["Alpha"])
    # Cycles ignores this; it is what makes the transparency show in an EEVEE
    # viewport, so the scene looks the same when opened by hand.
    if hasattr(mat, "blend_method"):
        mat.blend_method = "HASHED"
    return mat


def build_bloom(scene):
    """Bleed light out of the bright parts, in scene-linear light.

    A clipped line is a bright line, not a glowing one; what makes it glow is light
    landing on the pixels around it. Cycles' Glare node does that here, before the
    view transform, so it works on the emission as emitted — the creases are several
    times over white in linear terms and there is real energy to spread — instead of
    on values already squeezed into a display range.

    Blender 5 keeps the scene's compositor in a node group, and there is no longer a
    Composite node: the tree ends at the group's output. It begins at a Render
    Layers node, *not* at the group's input — a group input renders black, since
    nothing in the scene feeds it. Every parameter of the Glare node is a socket in
    this version rather than a property, and the two menu sockets take their labels
    as strings.
    """
    group = bpy.data.node_groups.new("bloom", "CompositorNodeTree")
    group.interface.new_socket(name="Image", in_out="OUTPUT",
                              socket_type="NodeSocketColor")
    layers = group.nodes.new("CompositorNodeRLayers")
    glare = group.nodes.new("CompositorNodeGlare")
    glare.location = (300, 0)
    for socket, value in BLOOM.items():
        glare.inputs[socket].default_value = value
    dst = group.nodes.new("NodeGroupOutput")
    dst.location = (600, 0)
    group.links.new(layers.outputs["Image"], glare.inputs["Image"])
    group.links.new(glare.outputs["Image"], dst.inputs["Image"])
    scene.use_nodes = True
    scene.compositing_node_group = group


def build_scene(res, samples, bloom=True):
    """A fresh scene with the sky, camera and material, but no geometry."""
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene

    # World: the sky lights the scene, a flat grey stands behind it.
    #
    # Both, and which is which decided by a Light Path node — the sky is what every
    # ray except a camera ray sees, so it does all the lighting; the camera sees only
    # `BACKDROP`. Without that split the sky's own gradient and sun disc would end up
    # in frame, and a picture meant to explain a shape does not want a landscape
    # behind it.
    #
    # There are deliberately no lamps. The sky is the entire light.
    world = bpy.data.worlds.new("void")
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    mix = nt.nodes.new("ShaderNodeMixShader")
    path = nt.nodes.new("ShaderNodeLightPath")
    backdrop = nt.nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = linear_rgba(BACKDROP_LINEAR)
    sky = nt.nodes.new("ShaderNodeTexSky")
    for prop, value in SKY.items():
        if hasattr(sky, prop):
            setattr(sky, prop, value)
    # slot 1 is taken when the factor is 0, slot 2 when it is 1
    nt.links.new(sky.outputs["Color"], mix.inputs[1])
    nt.links.new(backdrop.outputs["Background"], mix.inputs[2])
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    scene.world = world

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = CAM_LENS
    cam = bpy.data.objects.new("cam", cam_data)
    cam.rotation_euler = (-Vector(CAM_DIR)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam

    mat = build_material()

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            if any(d.type != "CPU" for d in prefs.get_devices_for_type(backend)):
                break
        for dev in prefs.get_devices_for_type(prefs.compute_device_type):
            dev.use = dev.type != "CPU"  # the GPU alone is faster than a mixed queue
    except Exception as e:
        print("GPU backend not configured, falling back to CPU:", e)
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    # A translucent shape stacks alpha hits along one ray — the gyroid is several
    # sheets deep — so the default 8 would clip the far ones to black. It is not
    # free either: every extra bounce is another shader evaluation on every ray, so
    # this buys enough depth to see through the shape and stops.
    scene.cycles.transparent_max_bounces = 12
    scene.render.resolution_x = res
    scene.render.resolution_y = res

    # 10-bit AVIF, since it carries both the extra range and, later, the animation.
    image = scene.render.image_settings
    image.file_format = "AVIF"
    image.color_mode = "RGB"
    image.color_depth = "10"
    image.quality = 90

    # Assigned, then read back. These are dynamic enumerations out of the OCIO
    # config, and a silent fallback to the default view transform would still
    # render — just not in HDR, and not obviously wrong until much later.
    scene.display_settings.display_device = DISPLAY_DEVICE
    scene.view_settings.view_transform = VIEW_TRANSFORM
    scene.view_settings.look = LOOK
    for what, got, want in (
        ("display device", scene.display_settings.display_device, DISPLAY_DEVICE),
        ("view transform", scene.view_settings.view_transform, VIEW_TRANSFORM),
        ("look", scene.view_settings.look, LOOK),
    ):
        if got != want:
            raise SystemExit(f"colour management: {what} is {got!r}, wanted {want!r}")

    if bloom:
        build_bloom(scene)
    return scene, mat


def frame_camera(scene, radius):
    """Back the camera off far enough that the shape's bounding sphere fits from
    any angle, with a small margin — which is what makes a turntable hold still
    and a morph sequence keep one scale."""
    sensor = scene.camera.data.sensor_width  # 36 mm, and the render is square
    half_fov = math.atan(0.5 * sensor / CAM_LENS)
    scene.camera.location = (
        Vector(CAM_DIR).normalized() * (1.05 * radius / math.sin(half_fov))
    )


def import_mesh(path, mat):
    before = set(bpy.data.objects)
    # forward=Y, up=Z is the identity: the OBJ is already in Blender's axes
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    fresh = [o for o in bpy.data.objects if o not in before]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in fresh:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    # smooth shading, which keeps the OBJ's per-vertex normals as custom split
    # normals — that is where the flat facets and smooth curves come from
    bpy.ops.object.shade_smooth()
    return fresh


def drop(objs):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)


def main():
    args = parse_args()
    manifest = (args.mesh_dir / "manifest.txt").read_text().split("\n")
    wanted = set(args.only.split(",")) if args.only else None
    scene, mat = build_scene(args.res, args.samples, bloom=not args.no_bloom)
    saved_blend = False

    for line in manifest:
        if not line.strip():
            continue
        name, kind, frames, radius = line.split()
        frames, radius = int(frames), float(radius)
        if args.max_frames:
            frames = min(frames, args.max_frames)
        if wanted and name not in wanted:
            continue
        out_dir = args.out / name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Clear what was there. The output format has changed before now, and a
        # directory holding two generations of frames is a trap for `encode.py`,
        # which globs by index and would happily mux a mixture.
        for stale in out_dir.iterdir():
            if stale.is_file():
                stale.unlink()
        frame_camera(scene, radius)

        objs = import_mesh(args.mesh_dir / f"{name}.obj", mat) if kind == "solid" else []
        for i in range(frames):
            if kind != "solid":
                drop(objs)
                objs = import_mesh(args.mesh_dir / f"{name}_{i:03d}.obj", mat)
            # A full turn either way. For a solid that *is* the animation; for a
            # cross-section it rides along with the sweep, because one fixed
            # viewpoint on a convex polyhedron barely reads as three-dimensional —
            # and a whole turn keeps the loop closed just as the cosine sweep does.
            for obj in objs:
                obj.rotation_euler = (0.0, 0.0, math.tau * i / frames)
            if args.blend and not saved_blend:
                bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
                print("wrote", args.blend)
                saved_blend = True
            scene.render.filepath = str(out_dir / f"{i:03d}")
            bpy.ops.render.render(write_still=True)
            print(f"[render] {name} {i + 1}/{frames}", flush=True)
        drop(objs)

    print("\nframes ->", args.out)


if __name__ == "__main__":
    main()
