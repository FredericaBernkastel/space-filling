//! Samples every shape and combinator onto a grid and dumps the raw field
//! values for `encode.py` to colour and encode.
//!
//! The point of doing this in Rust is that the numbers come from the crates
//! themselves — the pictures in the API docs are the real fields, and they go
//! stale the moment a field changes rather than quietly lying.
//!
//! Output: `doc/shapes/_raw/<name>.bin`, each `u32 frames, u32 w, u32 h` then
//! `frames·w·h` little-endian `f32`, plus a `manifest.txt` of
//! `name frames interval` lines. A shape that cannot be seen in the plane is
//! emitted as several frames — 2D slices through the higher-dimensional field,
//! swept along one axis — which `encode.py` turns into an animated WebP.

use {
  anyhow::Result,
  nalgebra::{Rotation2, Rotation3},
  rayon::prelude::*,
  space_filling::{
    geometry::*,
    sdf::{self, SDF},
  },
  std::{fs, io::Write, path::Path},
};

const RES: usize = 512;

/// How a field becomes an image: either one frame, or a swept stack of them.
enum Kind {
  Flat(Box<dyn Fn(P2<f64>) -> f64 + Send + Sync>),
  /// `f(t, p)` with `t` walking `0..1` across `frames`; the closure decides how
  /// the plane and `t` embed into the field's own dimension.
  Swept {
    frames: usize,
    f: Box<dyn Fn(f64, P2<f64>) -> f64 + Send + Sync>,
  },
}

struct Entry {
  name: &'static str,
  /// half-width of the square view
  extent: f64,
  centre: P2<f64>,
  /// distance between contour lines; scale it down for fields whose values are
  /// compressed (a normalized level set) so the contours stay legible
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

fn swept(
  name: &'static str,
  extent: f64,
  frames: usize,
  f: impl Fn(f64, P2<f64>) -> f64 + Send + Sync + 'static,
) -> Entry {
  Entry {
    name,
    extent,
    centre: P2::new(0.0, 0.0),
    interval: 0.12,
    kind: Kind::Swept { frames, f: Box::new(f) },
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
  // A level set, not a distance field: its values are divided by the gradient
  // bound, so the contours need a proportionally finer interval. Swept in 3D
  // because the 2D case degenerates — `sin(x)cos(y) + sin(y)cos(x) = sin(x + y)`
  // is a plain diagonal wave, and the labyrinth needs a third axis to exist.
  v.push(Entry {
    interval: 0.025,
    ..swept("gyroid", 1.6, 32, |t, p| {
      Gyroid { frequency: 5.0 }.sdf(Point::from([p.x, p.y, span(t, 0.62)]))
    })
  });
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

  // `extrude` and `revolve` change dimension, so they are swept
  v.push(swept("extrude", 1.5, 32, |t, p| {
    // a pentagram prism, sliced perpendicular to the extrusion axis: a constant
    // cross-section that stops at the end cap
    let z = span(t, 0.55);
    Pentagram.extrude(0.36).sdf(Point::from([p.x, p.y, z]))
  }));
  v.push(swept("revolve", 1.5, 32, |t, p| {
    // A star profile swept into a ring. Sliced by a plane *containing* the axis
    // of revolution (vertical, through the middle), which is the only family of
    // slices that shows the profile itself: two mirrored copies of it, at
    // ±offset. The sweep walks the plane off the axis, and they shrink — the
    // signature of a solid of revolution. Slicing perpendicular instead would
    // only ever give concentric annuli, hiding the profile entirely.
    Pentagram.scale(0.25).revolve(0.72).sdf(Point::from([p.y, p.x, span(t, 0.3)]))
  }));

  // ---- dn shapes whose interesting instance needs a third dimension ------
  v.push(swept("torus", 1.5, 32, |t, p| {
    // sliced across the axis of revolution: concentric annuli
    let x = span(t, 0.29);
    Torus { major: 0.62, minor: 0.3 }.sdf(Point::from([x, p.x, p.y]))
  }));
  v.push(swept("product_ball", 1.5, 32, |t, p| {
    // a cylinder: a disc over axes 0-1, an interval over axis 2
    let x = span(t, 0.85);
    ProductBall { spec: [(2, 0.75), (1, 0.55)] }.sdf(Point::from([x, p.x, p.y]))
  }));

  // ---- d3: no analogue in the plane, so swept along z --------------------
  for (name, hull, reach) in [
    ("dodecahedron", dodecahedron::<f64>(), 0.95),
    ("icosahedron", icosahedron::<f64>(), 0.95),
    ("truncated_octahedron", truncated_octahedron::<f64>(), 0.95),
    ("rhombic_dodecahedron", rhombic_dodecahedron::<f64>(), 0.95),
  ] {
    v.push(swept(name, 1.5, 32, move |t, p| {
      hull.sdf(Point::from([p.x, p.y, span(t, reach)]))
    }));
  }
  v.push(swept("torus_knot", 1.5, 36, |t, p| {
    let knot = torus_knot::<f64>(2, 3, 220, 0.1);
    knot.sdf(Point::from([p.x, p.y, span(t, 0.36)]))
  }));

  // ---- d4: swept along the fourth axis ----------------------------------
  for (name, hull) in [
    ("cell_24", cell_24::<f64>()),
    ("cell_120", cell_120::<f64>()),
    ("cell_600", cell_600::<f64>()),
  ] {
    v.push(swept(name, 1.5, 32, move |t, p| {
      hull.sdf(Point::from([p.x, p.y, 0.0, span(t, 0.95)]))
    }));
  }
  // Both of these are products of two circles, and a slice *parallel* to one of
  // the factor planes only ever shows that factor — a disc, indistinguishable
  // from a ball. So instead of translating the slice, rotate it: keep axis 0 and
  // tilt the second basis vector from axis 1 (inside the first factor) round to
  // axis 2 (inside the second). The section morphs between the two factors, which
  // is the whole content of the shape. `e_b` stays a unit vector, so the sampled
  // field is still the true 4D distance restricted to the plane.
  let tilt = |t: f64| std::f64::consts::FRAC_PI_4 * (1.0 - span(t, 1.0));
  v.push(swept("duocylinder", 1.5, 32, move |t, p| {
    // a disc of radius r1 at θ = 0, the rectangle 2·r1 × 2·r2 at θ = π/2
    let (s, c) = tilt(t).sin_cos();
    duocylinder(0.75, 0.55).sdf(Point::from([p.x, p.y * c, p.y * s, 0.0]))
  }));
  v.push(swept("clifford_torus", 1.5, 32, move |t, p| {
    // an annulus at θ = 0, four blobs at (±r1, ±r2) at θ = π/2
    let (s, c) = tilt(t).sin_cos();
    CliffordTorus { r1: 0.62, r2: 0.62, thickness: 0.2 }
      .sdf(Point::from([p.x, p.y * c, p.y * s, 0.0]))
  }));

  // a 3D rotation, since `rotate` is dimension-generic
  v.push(swept("rotation_3d", 1.5, 32, |t, p| {
    let r = Rotation3::from_euler_angles(0.4, 0.7, -0.3);
    Hyperrect { size: Vector::from([1.5, 0.8, 0.6]) }
      .rotate(r).sdf(Point::from([p.x, p.y, span(t, 0.7)]))
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

fn main() -> Result<()> {
  let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../shapes");
  let raw = root.join("_raw");
  fs::create_dir_all(&raw)?;

  let all = entries();
  let mut manifest = String::new();
  for e in &all {
    let (frames, data) = match &e.kind {
      Kind::Flat(f) => (1usize, raster(e, f.as_ref())),
      Kind::Swept { frames, f } => {
        let mut data = Vec::with_capacity(frames * RES * RES);
        for i in 0..*frames {
          let t = i as f64 / *frames as f64;
          let g = |p: P2<f64>| f(t, p);
          data.extend_from_slice(&raster(e, &g));
        }
        (*frames, data)
      }
    };

    let mut out = Vec::with_capacity(12 + data.len() * 4);
    out.extend_from_slice(&(frames as u32).to_le_bytes());
    out.extend_from_slice(&(RES as u32).to_le_bytes());
    out.extend_from_slice(&(RES as u32).to_le_bytes());
    for x in &data {
      out.extend_from_slice(&x.to_le_bytes());
    }
    let mut fh = fs::File::create(raw.join(format!("{}.bin", e.name)))?;
    fh.write_all(&out)?;

    manifest.push_str(&format!("{} {} {}\n", e.name, frames, e.interval));
    println!("{:<22} {:>3} frame(s)", e.name, frames);
  }
  fs::write(raw.join("manifest.txt"), manifest)?;
  println!("\n{} entries -> {}", all.len(), raw.display());
  Ok(())
}
