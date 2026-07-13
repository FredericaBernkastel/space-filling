"""
Imports a space-filling volumetric signed-distance grid (`.vol`, written by
`08_embedded_3d`) into Blender and builds a complete,
ready-to-render scene: an OpenVDB `distance` grid, a volumetric material whose
color, opacity and emission vary near the zero distance (glowing rim, milky
violet interior, transparent free space, turbulent haze), a camera, a soft key
light, a black world, and Cycles volume settings.

Usage:
  blender --python 08_embedded_3d_vdb.py -- out.vol
  blender -b --python 08_embedded_3d_vdb.py -- out.vol --render out.png

Requires an official Blender build (>= 3.0) — those bundle `pyopenvdb`.
A `.vdb` and a `.blend` are saved next to the input `.vol`.
"""
import bpy, struct, sys, os
from mathutils import Vector

def die(msg):
    raise SystemExit(f"[embedded_3d_vdb] {msg}")

try:
    import pyopenvdb as vdb
except ImportError:
    try:
        import openvdb as vdb
    except ImportError:
        die("this Blender build does not bundle pyopenvdb — use an official blender.org build (>= 3.0)")

import numpy as np

# ---------------- arguments ----------------
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
vol_path = argv[0] if argv and not argv[0].startswith('--') else 'out.vol'
render_path = None
if '--render' in argv:
    i = argv.index('--render')
    render_path = argv[i + 1] if i + 1 < len(argv) else '//render.png'

# ---------------- load the grid ----------------
with open(vol_path, 'rb') as f:
    if f.read(4) != b'SFVD':
        die(f"{vol_path}: not an SFVD volume")
    nx, ny, nz = struct.unpack('<III', f.read(12))
    data = np.frombuffer(f.read(nx * ny * nz * 4), dtype='<f4')
if data.size != nx * ny * nz:
    die(f"{vol_path}: truncated ({data.size} of {nx * ny * nz} voxels)")
data = data.reshape(nz, ny, nx)                        # file order: x fastest
data = np.ascontiguousarray(data.transpose(2, 1, 0))   # pyopenvdb wants [i][j][k] = (x, y, z)

# The background must equal the exporter's far-field clamp (`FAR` in the Rust
# example): at the grid boundary Cycles interpolates voxels toward the
# background, and a background of 0 would sweep through the zero isosurface —
# the densest, brightest band of the shader — painting every face of the cube.
FAR = 0.12
grid = vdb.FloatGrid(FAR)
grid.copyFromArray(data.astype(np.float32))
grid.name = 'distance'
grid.transform = vdb.createLinearTransform(voxelSize=1.0 / nx)

vdb_path = os.path.splitext(os.path.abspath(vol_path))[0] + '.vdb'
vdb.write(vdb_path, grids=[grid])
print('wrote', vdb_path)

# ---------------- fresh scene ----------------
bpy.ops.wm.read_homefile(use_empty=True)
scene = bpy.context.scene

bpy.ops.object.volume_import(filepath=vdb_path, align='WORLD', location=(-0.5, -0.5, -0.5))
volume = bpy.context.object

# ---------------- material ----------------
# The signed distance is remapped to t ∈ [0, 1]: deep interior → 0, the zero
# isosurface → 0.75, outside → 1. Three ramps over t drive color, density and
# emission; a noise texture modulates density into a turbulent haze.
mat = bpy.data.materials.new('sdf_cloud')
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()

out = nt.nodes.new('ShaderNodeOutputMaterial');   out.location = (950, 0)
pv  = nt.nodes.new('ShaderNodeVolumePrincipled'); pv.location = (650, 0)

attr = nt.nodes.new('ShaderNodeAttribute'); attr.location = (-950, 0)
attr.attribute_name = 'distance'

mapr = nt.nodes.new('ShaderNodeMapRange'); mapr.location = (-750, 0)
# scaled to the embedded generation's sphere sizes (r ≈ 0.005..0.04)
mapr.inputs['From Min'].default_value = -0.02
mapr.inputs['From Max'].default_value = 0.008
mapr.clamp = True

ramp_col = nt.nodes.new('ShaderNodeValToRGB'); ramp_col.location = (-500, 300)
els = ramp_col.color_ramp.elements
els[0].position = 0.403; els[0].color = (0.011764705882352941, 0.00392156862745098, 0.7098039215686275, 1.0)   # deep interior: night blue
els[1].position = 0.590816; els[1].color = (0.5843137254901961, 0.5647058823529412, 1.00, 1.0)   # body: blue-violet
e = ramp_col.color_ramp.elements.new(0.72449); e.color = (1.0, 0.788235294117647, 0.6, 1.0)  # rim: hot pink
e = ramp_col.color_ramp.elements.new(0.83301); e.color = (1.0, 0.5254901960784314, 0.011764705882352941, 1.0)  # halo: amber
e = ramp_col.color_ramp.elements.new(0.845204); e.color = (1.0, 0.0, 0.0, 1.0)  # exterior: deep red

ramp_dens = nt.nodes.new('ShaderNodeValToRGB'); ramp_dens.location = (-500, 0)
els = ramp_dens.color_ramp.elements
els[0].position = 0; els[0].color = (0.12, 0.12, 0.12, 1.0)  # interior: translucent haze
els[1].position = 0.75; els[1].color = (0.12, 0.25, 0.25, 1.0)
e = ramp_dens.color_ramp.elements.new(0.77); e.color = (0.55, 0.55, 0.55, 1.0)  # thin rim
e = ramp_dens.color_ramp.elements.new(0.79); e.color = (0.12, 0.12, 0.12, 1.0)    # free space: transparent
e = ramp_dens.color_ramp.elements.new(0.80); e.color = (0.0, 0.0, 0.0, 1.0)    # free space: transparent

ramp_emis = nt.nodes.new('ShaderNodeValToRGB'); ramp_emis.location = (-500, -300)
els = ramp_emis.color_ramp.elements
els[0].position = 0.75; els[0].color = (0.0, 0.0, 0.0, 1.0)
els[1].position = 0.77; els[1].color = (1.0, 1.0, 1.0, 1.0)      # emission peaks on the rim
e = ramp_emis.color_ramp.elements.new(0.78); e.color = (0.3, 0.3, 0.3, 1.0)
e = ramp_emis.color_ramp.elements.new(1); e.color = (0.0, 0.0, 0.0, 1.0)

noise = nt.nodes.new('ShaderNodeTexNoise'); noise.location = (-500, -600)
noise.inputs['Scale'].default_value = 9.0
noise.inputs['Detail'].default_value = 4.0
noise.inputs['Roughness'].default_value = 0.55

# density = ramp_dens · (0.55 + 0.45·noise) · 45
n_map = nt.nodes.new('ShaderNodeMath'); n_map.location = (-250, -450)
n_map.operation = 'MULTIPLY_ADD'
n_map.inputs[1].default_value = 0.45
n_map.inputs[2].default_value = 0.55
dens = nt.nodes.new('ShaderNodeMath'); dens.location = (-50, -150)
dens.operation = 'MULTIPLY'
dens_scaled = nt.nodes.new('ShaderNodeMath'); dens_scaled.location = (150, -150)
dens_scaled.operation = 'MULTIPLY'
dens_scaled.inputs[1].default_value = 50.0
emis_scaled = nt.nodes.new('ShaderNodeMath'); emis_scaled.location = (150, -400)
emis_scaled.operation = 'MULTIPLY'
emis_scaled.inputs[1].default_value = 25.0

ln = nt.links.new
ln(attr.outputs['Fac'],         mapr.inputs['Value'])
ln(mapr.outputs['Result'],      ramp_col.inputs['Fac'])
ln(mapr.outputs['Result'],      ramp_dens.inputs['Fac'])
ln(mapr.outputs['Result'],      ramp_emis.inputs['Fac'])
ln(noise.outputs['Fac'],        n_map.inputs[0])
ln(ramp_dens.outputs['Color'],  dens.inputs[0])
ln(n_map.outputs['Value'],      dens.inputs[1])
ln(dens.outputs['Value'],       dens_scaled.inputs[0])
ln(ramp_emis.outputs['Color'],  emis_scaled.inputs[0])
ln(ramp_col.outputs['Color'],   pv.inputs['Color'])
ln(ramp_col.outputs['Color'],   pv.inputs['Emission Color'])
ln(dens_scaled.outputs['Value'], pv.inputs['Density'])
ln(emis_scaled.outputs['Value'], pv.inputs['Emission Strength'])
pv.inputs['Anisotropy'].default_value = 0.3
ln(pv.outputs['Volume'], out.inputs['Volume'])

volume.data.materials.append(mat)

# ---------------- camera, light, world ----------------
# frame the cluster itself: bounding box of the near-surface voxels, in the
# volume object's world placement (grid [0,1]³ shifted by -0.5)
idx = np.argwhere(data < 0.02)
lo = idx.min(axis=0) / nx - 0.5
hi = (idx.max(axis=0) + 1) / nx - 0.5
target = Vector(((lo + hi) / 2).tolist())
extent = float((hi - lo).max())

def aim(obj, at):
    obj.rotation_euler = (Vector(at) - Vector(obj.location)).to_track_quat('-Z', 'Y').to_euler()

cam_data = bpy.data.cameras.new('cam')
cam_data.lens = 60
cam = bpy.data.objects.new('cam', cam_data)
scene.collection.objects.link(cam)
cam.location = target + Vector((1.0, -1.0, 0.55)).normalized() * (2.4 * extent)
aim(cam, target)
scene.camera = cam

key_data = bpy.data.lights.new('key', 'AREA')
key_data.energy = 160.0
key_data.size = 4.0
key_data.color = (0.85, 0.9, 1.0)
key = bpy.data.objects.new('key', key_data)
scene.collection.objects.link(key)
key.location = target + Vector((1.4, 0.9, 1.9))
aim(key, target)

world = bpy.data.worlds.new('void')
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
bg.inputs['Strength'].default_value = 0.0
scene.world = world

# ---------------- render settings ----------------
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
# The .blend stores the scene-level "GPU Compute" choice above; the backend
# type lives in user preferences, so also enable whatever this machine offers
# for the `--render` path (non-persistent). Without a GPU, Cycles falls back
# to CPU on its own.
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'):
        try:
            prefs.compute_device_type = backend
            break
        except TypeError:
            continue
    for dev in prefs.get_devices_for_type(prefs.compute_device_type):
        dev.use = True
except Exception as e:
    print('GPU backend not configured:', e)
scene.cycles.samples = 1024
scene.cycles.use_denoising = True
scene.cycles.volume_step_rate = 0.5
scene.cycles.volume_max_steps = 1024
scene.cycles.volume_bounces = 2
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
for view_transform, look in [('AgX', 'AgX - Punchy'), ('Filmic', 'High Contrast')]:
    try:
        scene.view_settings.view_transform = view_transform
        scene.view_settings.look = look
        break
    except TypeError:
        continue

blend_path = os.path.splitext(os.path.abspath(vol_path))[0] + '.blend'
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print('wrote', blend_path)

if render_path:
    scene.render.filepath = os.path.abspath(render_path)
    bpy.ops.render.render(write_still=True)
    print('rendered', scene.render.filepath)
elif bpy.app.background:
    print('scene saved; render with:')
    print(f'  blender -b {blend_path} -f 1')
