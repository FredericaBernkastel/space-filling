//! Stage one of the doc-image pipeline: evaluate every shape and combinator's
//! real signed distance field, and dump what the later stages need.
//!
//! The point of doing this in Rust is that the numbers come from the crates
//! themselves — the pictures in the API docs are the real fields, and they go
//! stale the moment a field changes rather than quietly lying.
//!
//! A shape that lives in the plane is dumped as a raster of field values, which
//! `encode.py` colours into a contour plot. A shape that does not is *meshed*
//! ([`mesh`]) and rendered in Blender by `render.py`, because a stack of 2D
//! slices only ever hints at a 3D form. Two flavours of that:
//!
//!   - a 3D field is meshed once and turntabled — 2π of rotation, so the loop
//!     closes;
//!   - a 4D field is sliced by a moving hyperplane and meshed once per frame. The
//!     changing section is the animation, and it is how a 4-polytope is legibly
//!     shown at all; the sweep is a cosine cycle and the turntable rides along
//!     with it, so that loop closes too.
//!
//! Outputs, all under `doc/shapes/`:
//!
//! ```text
//! _raw/<name>.bin      u32 frames, u32 w, u32 h, then f32 field values
//! _raw/manifest.txt    `name frames interval` per planar entry
//! _mesh/<name>.obj     solids
//! _mesh/<name>_NNN.obj one per frame, for the sliced 4D entries
//! _mesh/manifest.txt   `name kind frames radius` per meshed entry
//! ```

mod mesh;

use {
  anyhow::Result,
  nalgebra::{Rotation2, Rotation3},
  rayon::prelude::*,
  space_filling::{
    geometry::*,
    sdf::{self, SDF},
  },
  std::{fs, io::Write, path::Path, time::Instant},
};

/// Local to this binary; `geometry` exports `P2`/`V2` but not the 3D pair.
type P3<T> = Point<T, 3>;

/// Raster resolution of the planar contour plots.
const RES: usize = 512;
// Meshing resolution: `res³` field evaluations, so this is what dominates
// stage-one runtime — and what limits how sharp an edge can be. A crease is
// resolved to about one cell, and the leftover wobble along it is the one visible
// defect that no amount of care in placing the vertex can remove: where a wedge is
// thinner than a cell, the corner signs simply do not describe the surface.
//
// So the resolution is chosen per shape, from what that shape can actually gain by
// it. A torus or a knot is smooth: its normals come from the field analytically and
// are exact at any spacing, its silhouette is smooth however coarse the grid, and a
// finer one buys nothing — which is fortunate, because the knot is by far the most
// expensive field here. A polytope is all creases, and every one of them wobbles.
//
// Deciding needs two probes, not one, because a single measurement cannot tell a
// crease from a tight curve: at a coarse spacing both turn the normal sharply from
// cell to cell. Refining is what separates them — see
// [`crease_scale`](mesh::Mesh::crease_scale). Both probes together cost about a
// seventh of the coarse pass, and are thrown away.
const PROBE_RES: usize = 80;
const COARSE_RES: usize = 320;
const FINE_RES: usize = 512;
/// How much of its sharpness a shape must keep when the grid is refined to earn
/// [`FINE_RES`]. A perfectly smooth surface scores 0.5, a perfect crease 1.
const CREASE_PERSISTENCE: f64 = 0.72;
/// For the 4D slices, which are meshed once per frame — [`MORPH_FRAMES`] times
/// over, where a solid is meshed once. Their sections are polyhedra with a great
/// many edges, every one a crease the renderer draws a line along, so too coarse and
/// the lines come out dotted; but at [`FINE_RES`] a single one of these would take
/// the better part of two hours, so they get their own middle ground.
///
/// This is also what the intermediate directory costs: a section at this resolution
/// runs to some 30 MB of OBJ, so the five sliced shapes come to about 20 GB between
/// them. Lower this before anything else if disk is short.
const MORPH_RES: usize = 288;
/// Turntable frames for a solid, and sweep frames for a 4D slice.
///
/// A full turn in 128 steps is 2.8° a frame, which is smooth. The frame *rate* lives
/// in `encode.py`, so raising this alone makes each loop longer rather than smoother
/// at the same speed.
const TURN_FRAMES: usize = 128;
const MORPH_FRAMES: usize = 128;

/// How a field becomes an image.
enum Kind {
  /// Sampled in the plane and coloured into a contour plot.
  Flat(Box<dyn Fn(P2<f64>) -> f64 + Send + Sync>),
  /// A 3D field, meshed once and turntabled.
  Solid {
    /// half-width of the meshing cube, `[-half, half]³`
    half: f64,
    f: Box<dyn Fn(P3<f64>) -> f64 + Send + Sync>,
  },
  /// A 4D field cut by a hyperplane that moves with `t ∈ 0..1`; the closure
  /// decides which axis is swept and how far. Meshed once per frame.
  Morph {
    half: f64,
    f: Box<dyn Fn(f64, P3<f64>) -> f64 + Send + Sync>,
  },
}

struct Entry {
  name: &'static str,
  /// half-width of the square view — [`Kind::Flat`] only
  extent: f64,
  /// [`Kind::Flat`] only
  centre: P2<f64>,
  /// distance between contour lines; scale it down for fields whose values are
  /// compressed (a normalized level set) so the contours stay legible.
  /// [`Kind::Flat`] only
  interval: f64,
  kind: Kind,
}

fn flat(
  name: &'static str,
  extent: f64,
  f: impl Fn(P2<f64>) -> f64 + Send + Sync + 'static,
) -> Entry {
  Entry {
    name,
    extent,
    centre: P2::new(0.0, 0.0),
    interval: 0.12,
    kind: Kind::Flat(Box::new(f)),
  }
}

/// A 3D field, meshed inside `[-half, half]³`.
fn solid(
  name: &'static str,
  half: f64,
  f: impl Fn(P3<f64>) -> f64 + Send + Sync + 'static,
) -> Entry {
  Entry {
    name,
    extent: 0.0,
    centre: P2::new(0.0, 0.0),
    interval: 0.0,
    kind: Kind::Solid { half, f: Box::new(f) },
  }
}

/// A 4D field, sliced and meshed once per frame.
fn morph(
  name: &'static str,
  half: f64,
  f: impl Fn(f64, P3<f64>) -> f64 + Send + Sync + 'static,
) -> Entry {
  Entry {
    name,
    extent: 0.0,
    centre: P2::new(0.0, 0.0),
    interval: 0.0,
    kind: Kind::Morph { half, f: Box::new(f) },
  }
}

/// `t ∈ 0..1` mapped onto `[-a, a]` and back — one full cosine cycle, so the
/// frame sequence loops seamlessly without storing it twice, and the sweep eases
/// in and out at the extremes.
fn span(t: f64, a: f64) -> f64 {
  a * (std::f64::consts::TAU * t).cos()
}

fn entries() -> Vec<Entry> {
  let mut v: Vec<Entry> = vec![];

  // ---- dn: the dimension-generic shapes, shown in the plane ---------------
  v.push(flat("hypersphere", 1.5, |p| Hypersphere.sdf(p)));
  v.push(flat("hyperrect", 1.5, |p| Hyperrect { size: V2::new(1.5, 0.9) }.sdf(p)));
  v.push(flat("hypersquare", 1.6, |p| Hypersquare.sdf(p)));
  v.push(flat("line", 1.5, |p| Line {
    a: P2::new(-0.7, -0.45), b: P2::new(0.65, 0.5), thickness: 0.35,
  }.sdf(p)));
  v.push(flat("ring", 1.5, |p| Ring { inner_r: 0.55 }.sdf(p)));
  v.push(flat("moon", 1.5, |p| Moon { phase: 0.5 }.sdf(p)));
  v.push(flat("kakera", 1.5, |p| Kakera { width: 0.55 }.sdf(p)));
  v.push(flat("cross", 1.6, |p| Cross { thickness: 0.3 }.sdf(p)));
  v.push(flat("orthoplex", 1.5, |p| Orthoplex.sdf(p)));
  v.push(flat("lp_ball", 1.5, |p| LpBall { p: 4.0 }.sdf(p)));
  v.push(flat("polyline", 1.5, |p| Polyline {
    vertices: [
      P2::new(-0.75, -0.3), P2::new(-0.2, 0.55), P2::new(0.3, -0.5),
      P2::new(0.8, 0.35),
    ],
    thickness: 0.22,
  }.sdf(p)));
  v.push(flat("simplex", 1.5, |p| simplex::<f64, 2>().sdf(p)));
  v.push(flat("permutohedron", 1.5, |p| permutohedron::<f64, 2>().sdf(p)));
  v.push(flat("polytope", 1.5, |p| {
    // five half-spaces, at unit circumradius: a pentagon
    let hs: Vec<HalfSpace<f64, 2>> = (0..5)
      .map(|i| {
        let a = std::f64::consts::TAU * i as f64 / 5.0 + std::f64::consts::FRAC_PI_2;
        HalfSpace::new(V2::new(a.cos(), a.sin()), (std::f64::consts::PI / 5.0).cos())
      })
      .collect();
    Polytope { half_spaces: hs }.sdf(p)
  }));
  // The 2D gyroid degenerates — `sin(x)cos(y) + sin(y)cos(x) = sin(x + y)` is a
  // plain diagonal wave — so it is only ever shown in 3D. Thickened into a sheet
  // and clipped to a ball, since the surface itself is unbounded and, being a
  // surface, has no interior to shade.
  //
  // `shell` needs a half-width in the *field's* units, and this field is divided
  // by its gradient bound `2k√D` — at k = 4, D = 3 the whole range is only
  // ±0.108, so a naive 0.05 would thicken the sheet into a solid ball. Near the
  // zero set the value runs at roughly 0.35 of true distance, which puts 0.02
  // here at a sheet some 0.12 thick.
  v.push(solid("gyroid", 1.05, |p| {
    Gyroid { frequency: 4.0 }
      .shell(0.02)
      .intersection(Hypersphere::<3>.scale(0.95))
      .sdf(p)
  }));
  // the unit-cube walls, positive inside — drawn over its own domain
  v.push(Entry {
    centre: P2::new(0.5, 0.5),
    interval: 0.06,
    ..flat("boundary_rect", 0.75, |p| sdf::boundary_rect(p))
  });

  // ---- d2: the plane-only families ---------------------------------------
  v.push(flat("ngon_c", 1.5, |p| NGonC::<6>.sdf(p)));
  v.push(flat("ngon_r", 1.5, |p| NGonR { n: 5 }.sdf(p)));
  v.push(flat("star", 1.5, |p| Star { n: 5, m: 10.0 / 3.0 }.sdf(p)));
  v.push(flat("pentagram", 1.5, |p| Pentagram.sdf(p)));
  v.push(flat("hexagram", 1.5, |p| Hexagram.sdf(p)));
  v.push(flat("polygon", 1.5, |p| Polygon {
    vertices: [
      P2::new(-0.9, -0.5), P2::new(0.8, -0.7),
      P2::new(0.5, 0.9), P2::new(-0.3, 0.4),
    ],
  }.sdf(p)));
  v.push(flat("holy_cross", 1.5, |p| HolyCross.sdf(p)));

  // ---- the combinators ---------------------------------------------------
  v.push(flat("translation", 1.5, |p| {
    Hypersquare.scale(0.45).translate(V2::new(0.45, 0.3)).sdf(p)
  }));
  v.push(flat("rotation", 1.5, |p| {
    Hyperrect { size: V2::new(1.5, 0.6) }
      .rotate(Rotation2::new(30f64.to_radians())).sdf(p)
  }));
  v.push(flat("scale", 1.5, |p| Pentagram.scale(0.55).sdf(p)));
  let pair_a = || Hypersphere.scale(0.6).translate(V2::new(-0.28, -0.1));
  let pair_b = || Hypersquare.scale(0.5).translate(V2::new(0.3, 0.15));
  v.push(flat("union", 1.5, move |p| pair_a().union(pair_b()).sdf(p)));
  v.push(flat("subtraction", 1.5, move |p| pair_a().subtraction(pair_b()).sdf(p)));
  v.push(flat("intersection", 1.5, move |p| pair_a().intersection(pair_b()).sdf(p)));
  v.push(flat("smooth_min", 1.5, move |p| pair_a().smooth_min(pair_b(), 8.0).sdf(p)));
  v.push(flat("shell", 1.5, |p| Hypersquare.scale(0.6).shell(0.09).sdf(p)));
  v.push(flat("offset", 1.6, |p| Cross { thickness: 0.22 }.offset(0.14).sdf(p)));

  // `extrude` and `revolve` change dimension, so the plane cannot hold them
  v.push(solid("extrude", 1.05, |p| Pentagram.extrude(0.36).sdf(p)));
  // The axis of revolution is axis 0. Deliberately left lying along x rather
  // than stood upright: which axis it is is part of what the reader needs.
  v.push(solid("revolve", 1.15, |p| {
    // a fat profile relative to the ring, so the star is the thing you see and
    // not a ripple on a torus; 0.62 - 0.38 > 0 keeps the hole open
    Pentagram.scale(0.38).revolve(0.62).sdf(p)
  }));

  // ---- dn shapes whose interesting instance needs a third dimension ------
  v.push(solid("torus", 1.05, |p| Torus { major: 0.62, minor: 0.3 }.sdf(p)));
  // a cylinder: a disc over axes 0-1, an interval over axis 2
  v.push(solid("product_ball", 0.9, |p| {
    ProductBall { spec: [(2, 0.75), (1, 0.55)] }.sdf(p)
  }));

  // ---- d3: no analogue in the plane, so meshed and turntabled -------------
  for (name, hull) in [
    ("dodecahedron", dodecahedron::<f64>()),
    ("icosahedron", icosahedron::<f64>()),
    ("truncated_octahedron", truncated_octahedron::<f64>()),
    ("rhombic_dodecahedron", rhombic_dodecahedron::<f64>()),
  ] {
    v.push(solid(name, 1.08, move |p| hull.sdf(p)));
  }
  v.push(solid("torus_knot", 1.15, |p| {
    torus_knot::<f64>(2, 3, 220, 0.1).sdf(p)
  }));

  // ---- d4: the 3D cross-section, swept along the fourth axis --------------
  // A 4-polytope has no 3D form to render, so what is rendered is its solid
  // cross-section — the honest 3D object obtained by cutting with the hyperplane
  // `x3 = w` — and `w` is what the animation moves. Cutting a *regular*
  // 4-polytope this way is how its cell structure becomes visible at all: the
  // section grows from a point through a succession of polyhedra and back.
  // How far `w` may travel is *not* the circumradius: it is how far the polytope
  // itself reaches along that one axis, which differs per shape — the 24-cell's
  // vertices are the permutations of `(±1, ±1, 0, 0)/√2`, so it stops at 0.707,
  // while the 120- and 600-cells both have a vertex on the axis and reach 1. So
  // take it from each shape's own vertices, and stay just inside, or the extreme
  // sections come out empty.
  for (name, hull, verts) in [
    ("cell_24", cell_24::<f64>(), cell_24_vertices::<f64>()),
    ("cell_120", cell_120::<f64>(), cell_120_vertices::<f64>()),
    ("cell_600", cell_600::<f64>(), cell_600_vertices::<f64>()),
  ] {
    let reach = 0.82 * verts.iter().map(|v| v[3].abs()).fold(0.0, f64::max);
    v.push(morph(name, 1.05, move |t, p| {
      hull.sdf(Point::from([p.x, p.y, p.z, span(t, reach)]))
    }));
  }
  // The duocylinder is the product of two discs, so every section is a cylinder:
  // full radius over the first factor, a height set by how far `w` has eaten into
  // the second. That the height collapses while the radius does not is exactly
  // what distinguishes it from a 4-ball.
  v.push(morph("duocylinder", 0.9, |t, p| {
    duocylinder(0.75, 0.55).sdf(Point::from([p.x, p.y, p.z, span(t, 0.44)]))
  }));
  // Sections of the Clifford torus are tori, thickening and thinning as `w`
  // moves through the second factor circle.
  v.push(morph("clifford_torus", 1.0, |t, p| {
    CliffordTorus { r1: 0.62, r2: 0.62, thickness: 0.2 }
      .sdf(Point::from([p.x, p.y, p.z, span(t, 0.5)]))
  }));

  // a 3D rotation, since `rotate` is dimension-generic
  v.push(solid("rotation_3d", 1.15, |p| {
    Hyperrect { size: Vector::from([1.5, 0.8, 0.6]) }
      .rotate(Rotation3::from_euler_angles(0.4, 0.7, -0.3))
      .sdf(p)
  }));

  v
}

/// Sample one frame; row 0 is `+y`, matching `doc/video/fields.py::sample`.
fn raster(e: &Entry, f: &(dyn Fn(P2<f64>) -> f64 + Send + Sync)) -> Vec<f32> {
  let n = RES;
  let mut out = vec![0f32; n * n];
  out.par_chunks_mut(n).enumerate().for_each(|(row, line)| {
    let y = e.centre.y + e.extent - 2.0 * e.extent * row as f64 / (n - 1) as f64;
    for (col, px) in line.iter_mut().enumerate() {
      let x = e.centre.x - e.extent + 2.0 * e.extent * col as f64 / (n - 1) as f64;
      *px = f(P2::new(x, y)) as f32;
    }
  });
  out
}

/// Report what a freshly meshed field looks like, and complain about the two
/// ways it can be wrong: a surface that runs into the meshing box is clipped,
/// and quads wound the wrong way would light incorrectly.
fn report(
  name: &str,
  m: &mesh::Mesh,
  d: &mesh::Diagnostics,
  elapsed: std::time::Duration,
) {
  let creased = m.edge.iter().filter(|&&e| e > 0.5).count();
  println!(
    "{name:<22} {:>7} verts {:>7} tris  r={:.3}  crease {:>4.1}%  clamp {:>4.2}%  flip {}->{} ({} fixed)  {:.1?}",
    m.verts.len(),
    2 * m.quads.len(),
    m.radius(),
    100.0 * creased as f64 / m.verts.len().max(1) as f64,
    100.0 * d.clamped as f64 / m.verts.len().max(1) as f64,
    d.inverted_before,
    d.inverted,
    d.repaired,
    elapsed
  );
  if m.is_empty() {
    eprintln!("  !! {name}: empty mesh — the field is nowhere negative in the box");
  }
  if !d.contained {
    eprintln!("  !! {name}: the surface reaches the meshing box and is being clipped");
  }
  let w = m.winding_agreement();
  if !m.is_empty() && w < 0.9 {
    eprintln!("  !! {name}: quad winding disagrees with the analytic normals ({w:.3})");
  }
  if d.inverted > 0 {
    // Not everything is repairable. Where the surface turns a right angle, two of
    // the four cells around a grid edge hold one face and two hold the other, and
    // the quad between them is twisted badly enough that both of its triangulations
    // fold — the rim of the duocylinder's cross-section does this along its whole
    // length. Undoing it needs several vertices per cell, which is a different
    // algorithm; these faces are slivers at a crease, and render invisibly.
    eprintln!(
      "  !! {name}: {} face(s) twisted past repair — sliver(s) at a right-angle crease",
      d.inverted
    );
  }
  // A sharp corner legitimately clamps, so this is a proportion rather than a
  // presence test: a large share means the fits are degenerate.
  let share = d.clamped as f64 / m.verts.len().max(1) as f64;
  if share > 0.02 {
    eprintln!(
      "  !! {name}: {:.1}% of vertices solved outside their cell — under-resolved",
      100.0 * share
    );
  }
}

fn write_raw(path: &Path, frames: usize, data: &[f32]) -> Result<()> {
  let mut out = Vec::with_capacity(12 + data.len() * 4);
  out.extend_from_slice(&(frames as u32).to_le_bytes());
  out.extend_from_slice(&(RES as u32).to_le_bytes());
  out.extend_from_slice(&(RES as u32).to_le_bytes());
  for x in data {
    out.extend_from_slice(&x.to_le_bytes());
  }
  fs::File::create(path)?.write_all(&out)?;
  Ok(())
}

/// The manifests are keyed by name and must stay complete even when only some
/// entries were regenerated, so a run merges into what is already on disk rather
/// than truncating it. Lines come back out in `entries()` order.
fn merge_manifest(path: &Path, fresh: &[(&str, String)], order: &[&str]) -> Result<()> {
  let mut lines: Vec<(String, String)> = fs::read_to_string(path)
    .unwrap_or_default()
    .lines()
    .filter_map(|l| l.split_whitespace().next().map(|n| (n.to_string(), l.to_string())))
    .collect();
  for (name, line) in fresh {
    match lines.iter_mut().find(|(n, _)| n == name) {
      Some(slot) => slot.1 = line.clone(),
      None => lines.push((name.to_string(), line.clone())),
    }
  }
  let mut out = String::new();
  for name in order {
    if let Some((_, line)) = lines.iter().find(|(n, _)| n == name) {
      out.push_str(line);
      out.push('\n');
    }
  }
  fs::write(path, out)?;
  Ok(())
}

fn main() -> Result<()> {
  let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../shapes");
  let raw = root.join("_raw");
  let mesh_dir = root.join("_mesh");
  fs::create_dir_all(&raw)?;
  fs::create_dir_all(&mesh_dir)?;

  // Names on the command line restrict the run to those entries, so one shape can
  // be re-meshed without rewriting the rest — which matters because a full pass
  // takes minutes and would clobber files a running `render.py` is reading.
  let only: Vec<String> = std::env::args().skip(1).collect();
  let all = entries();
  if let Some(bad) = only.iter().find(|n| !all.iter().any(|e| e.name == n.as_str())) {
    anyhow::bail!("no such entry: {bad}");
  }

  let (mut flat_manifest, mut mesh_manifest) = (vec![], vec![]);
  for e in all.iter().filter(|e| only.is_empty() || only.iter().any(|n| n == e.name)) {
    match &e.kind {
      Kind::Flat(f) => {
        write_raw(&raw.join(format!("{}.bin", e.name)), 1, &raster(e, f.as_ref()))?;
        flat_manifest.push((e.name, format!("{} 1 {}", e.name, e.interval)));
        println!("{:<22} planar raster", e.name);
      }
      Kind::Solid { half, f } => {
        let t0 = Instant::now();
        let (coarse, _) = mesh::dual_contour(f.as_ref(), *half, PROBE_RES);
        let (finer, _) = mesh::dual_contour(f.as_ref(), *half, 2 * PROBE_RES);
        // halving the cell halves a smooth surface's deviation and leaves a
        // crease's alone, so this ratio is near 0.5 for one and near 1 for the other
        let persistence = finer.crease_scale() / coarse.crease_scale().max(1e-9);
        let res = if persistence > CREASE_PERSISTENCE { FINE_RES } else { COARSE_RES };
        println!(
          "{:<22} probe: sharpness {:.3} -> {:.3} on refinement ({persistence:.2}) -> res {res}",
          e.name,
          coarse.crease_scale(),
          finer.crease_scale(),
        );
        let (m, d) = mesh::dual_contour(f.as_ref(), *half, res);
        report(e.name, &m, &d, t0.elapsed());
        mesh::write_obj(&mesh_dir.join(format!("{}.obj", e.name)), e.name, &m)?;
        mesh_manifest
          .push((e.name, format!("{} solid {} {:.5}", e.name, TURN_FRAMES, m.radius())));
      }
      Kind::Morph { half, f } => {
        // one mesh per frame; the camera is framed on the largest of them, so
        // the section is seen to grow and shrink instead of being re-zoomed
        let t0 = Instant::now();
        let mut radius = 0.0f64;
        // Reported once for the whole sweep rather than per frame: the sections
        // of the same shape fail in the same way or not at all.
        let mut worst = 0usize;
        // Sweep files are numbered, so a shorter sweep than last time would leave
        // the tail of the old one lying around — and at 30 MB a section, that is not
        // a small mistake.
        let prefix = format!("{}_", e.name);
        for stale in fs::read_dir(&mesh_dir)? {
          let path = stale?.path();
          let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_owned();
          if name.starts_with(&prefix) && name.ends_with(".obj") {
            fs::remove_file(path)?;
          }
        }
        for i in 0..MORPH_FRAMES {
          let t = i as f64 / MORPH_FRAMES as f64;
          let (m, d) = mesh::dual_contour(&|p: P3<f64>| f(t, p), *half, MORPH_RES);
          if i == 0 {
            report(&format!("{}[{i}]", e.name), &m, &d, t0.elapsed());
          } else if m.is_empty() || !d.contained {
            report(&format!("{}[{i}]", e.name), &m, &d, t0.elapsed());
          }
          worst = worst.max(d.inverted);
          mesh::write_obj(
            &mesh_dir.join(format!("{}_{:03}.obj", e.name, i)),
            e.name,
            &m,
          )?;
          radius = radius.max(m.radius());
        }
        mesh_manifest
          .push((e.name, format!("{} morph {} {:.5}", e.name, MORPH_FRAMES, radius)));
        println!(
          "{:<22} {MORPH_FRAMES} sections  r={radius:.3}  worst flip {worst}  {:.1?}",
          e.name,
          t0.elapsed()
        );
      }
    }
  }
  let order: Vec<&str> = all.iter().map(|e| e.name).collect();
  merge_manifest(&raw.join("manifest.txt"), &flat_manifest, &order)?;
  merge_manifest(&mesh_dir.join("manifest.txt"), &mesh_manifest, &order)?;
  println!(
    "\n{} of {} entries -> {}",
    flat_manifest.len() + mesh_manifest.len(),
    all.len(),
    root.display()
  );
  Ok(())
}
