//! Bakes an expensive fractal distance estimator into a discrete signed
//! distance field: rasterize a membership mask once, run an exact Euclidean
//! distance transform (EDT) over it, then sample the grid bilinearly.
//!
//! Compared to `06_custom_primitive` (same scene, same insertion loop):
//!  - one field evaluation: ~256-iteration complex loop → one bilinear fetch;
//!  - Lipschitz bound: empirical `L = 4` → certified `L ≤ 1 + √2`
//!    (an exact EDT is 1-Lipschitz, bilinear interpolation of it is √2-Lipschitz,
//!    and the outside-the-window extension adds at most 1), so primitive
//!    pruning is sound rather than heuristic;
//!  - the bake is paid once and shared — by the static obstacle and by every
//!    one of the 20'000 inserted primitives.
//!
//! The price: field accuracy is capped at the grid pitch, and fractal detail
//! below one cell is lost.

use {
  space_filling::{
    sdf::{self, SDF, Lipschitz},
    solver::{ADF, LineSearch, Primitive},
    drawing::Draw,
    geometry::{WorldSpace, BoundingBox, Shape, Scale, Translation, P2},
    util
  },
  euclid::{Point2D, Vector2D as V2, Box2D},
  image::{RgbaImage, Luma, Pixel},
  anyhow::Result,
  num_traits::Float,
  num_complex::Complex,
  rayon::prelude::*,
  std::{sync::Arc, time::Instant}
};

#[derive(Debug, Copy, Clone)]
/// Same analytic estimator as in `06_custom_primitive`; here it is only
/// evaluated during the bake.
struct MandlelDE;

impl <T: Float> SDF<T> for MandlelDE {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let c = Complex::new(pixel.x, pixel.y);
    let mut z = Complex::new(T::zero(), T::zero());
    let mut dz = Complex::new(T::one(), T::zero());
    let mut is_inside = true;
    for _ in 0..256 {
      let z_new = z.powi(2) + c;
      let dz_new = z.scale(T::from(2.0).unwrap()) * dz + T::one();
      z = z_new;
      dz = dz_new;
      if z.norm_sqr() > T::from(1e9).unwrap() {
        is_inside = false;
        break;
      }
    }
    if is_inside {
      return T::zero();
    }
    z.norm() * z.norm().ln() / dz.norm()
  }
}

impl<T: Float> BoundingBox<T> for MandlelDE {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::new(T::from(-2.5).unwrap(), T::from(-1.25).unwrap()),
      Point2D::new(T::from(0.5).unwrap(), T::from(1.25).unwrap())
    )}}

/// scaled to a box [-1, 1]
fn mandel_de_norm<T: Float>() -> Scale<Translation<MandlelDE, T>, T> {
  MandlelDE
    .translate(V2::new(T::from(1.0).unwrap(), T::zero()))
    .scale(T::one() / T::from(1.5).unwrap())
}

/// A signed distance field on a regular grid, bilinearly interpolated.
struct BakedSdf {
  data: Vec<f32>,
  res: usize,
  /// Square window the grid covers.
  window: Box2D<f64, WorldSpace>,
  cell: f64,
  /// Bounding box reported to the combinators (their rotation/scale pivot) —
  /// kept equal to the analytic shape's, so the scene matches `06` exactly.
  bbox: Box2D<f64, WorldSpace>,
}

impl BakedSdf {
  /// Rasterize `inside` over `window` at `res`² and take the exact signed
  /// Euclidean distance transform of the mask.
  fn bake(
    inside: impl Fn(P2<f64>) -> bool + Sync,
    window: Box2D<f64, WorldSpace>,
    bbox: Box2D<f64, WorldSpace>,
    res: usize,
  ) -> Self {
    let cell = window.width() / (res - 1) as f64;
    let mask: Vec<bool> = (0..res * res).into_par_iter()
      .map(|i| inside(Point2D::new(
        window.min.x + (i % res) as f64 * cell,
        window.min.y + (i / res) as f64 * cell,
      )))
      .collect();

    let d_out = edt_squared(&mask, res, false); // squared distance to the set
    let d_in = edt_squared(&mask, res, true);   // squared distance to the complement
    let data = d_out.iter().zip(&d_in)
      .map(|(&o, &i)| ((o.sqrt() - i.sqrt()) * cell) as f32)
      .collect();
    Self { data, res, window, cell, bbox }
  }

  /// Clamped bilinear sample at a world-space point inside the window.
  fn sample(&self, p: P2<f64>) -> f64 {
    let gx = ((p.x - self.window.min.x) / self.cell).clamp(0.0, (self.res - 1) as f64);
    let gy = ((p.y - self.window.min.y) / self.cell).clamp(0.0, (self.res - 1) as f64);
    let (x0, y0) = ((gx as usize).min(self.res - 2), (gy as usize).min(self.res - 2));
    let (fx, fy) = (gx - x0 as f64, gy - y0 as f64);
    let at = |x: usize, y: usize| self.data[y * self.res + x] as f64;
    (at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx) * (1.0 - fy)
      + (at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx) * fy
  }
}

/// Cheaply clonable handle. Implements `SDF` + `BoundingBox`, hence gets every
/// `Shape` combinator (translate / rotate / scale) for free.
#[derive(Clone)]
struct Baked(Arc<BakedSdf>);

impl SDF<f64> for Baked {
  fn sdf(&self, p: P2<f64>) -> f64 {
    let b = &self.0;
    // Outside the baked window, extend with the distance to the window: the
    // field keeps growing (a small primitive can never wrongly become the
    // global minimum far away), and the total stays (1 + √2)-Lipschitz.
    let q = Point2D::new(
      p.x.clamp(b.window.min.x, b.window.max.x),
      p.y.clamp(b.window.min.y, b.window.max.y),
    );
    b.sample(q) + (p - q).length()
  }
}
impl BoundingBox<f64> for Baked {
  fn bounding_box(&self) -> Box2D<f64, WorldSpace> { self.0.bbox }
}

/// Certified: an exact EDT is 1-Lipschitz, bilinear interpolation of it √2,
/// and the outside-the-window extension adds at most 1. Declared once here,
/// the bound propagates through every combinator chain automatically.
const L_BAKED: f64 = 2.415;

impl Lipschitz<f64> for Baked {
  fn lipschitz(&self) -> f64 { L_BAKED }
}

/// Exact squared Euclidean distance transform (Felzenszwalb & Huttenlocher):
/// per-pixel squared distance to the nearest `mask == !invert` pixel, in pixels.
fn edt_squared(mask: &[bool], res: usize, invert: bool) -> Vec<f64> {
  const INF: f64 = 1e20;
  let mut field: Vec<f64> = mask.iter()
    .map(|&m| if m != invert { 0.0 } else { INF })
    .collect();

  let mut d = vec![0.0; res];
  let mut v = vec![0usize; res];
  let mut z = vec![0.0; res + 1];
  for row in field.chunks_mut(res) {
    dt1d(row, &mut d, &mut v, &mut z);
    row.copy_from_slice(&d);
  }
  let mut col = vec![0.0; res];
  for x in 0..res {
    for y in 0..res { col[y] = field[y * res + x]; }
    dt1d(&col, &mut d, &mut v, &mut z);
    for y in 0..res { field[y * res + x] = d[y]; }
  }
  field
}

/// 1D squared distance transform via the lower envelope of parabolas.
fn dt1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
  let n = f.len();
  let sq = |x: usize| (x * x) as f64;
  let mut k = 0usize;
  v[0] = 0;
  z[0] = -1e30;
  z[1] = 1e30;
  for q in 1..n {
    let mut s = ((f[q] + sq(q)) - (f[v[k]] + sq(v[k]))) / (2 * (q - v[k])) as f64;
    while s <= z[k] {
      k -= 1;
      s = ((f[q] + sq(q)) - (f[v[k]] + sq(v[k]))) / (2 * (q - v[k])) as f64;
    }
    k += 1;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e30;
  }
  k = 0;
  for q in 0..n {
    while z[k + 1] < q as f64 { k += 1; }
    d[q] = (q as f64 - v[k] as f64).powi(2) + f[v[k]];
  }
}

// profile, 06 analytic estimator (Lipschitz B&B pruning): 19.2s
fn main() -> Result<()> {
  let start_time = Instant::now();
  let path = "out.png";

  // --- bake once ---
  // (The field is exact for the *rasterized* set; detail below one grid cell
  // is gone.)
  const RES: usize = 2048;

  let analytic = mandel_de_norm::<f64>();
  let bbox = analytic.bounding_box();
  let window = {
    let c = bbox.center();
    let half = bbox.width().max(bbox.height()) / 2.0 + 0.25;
    Box2D::new(
      Point2D::new(c.x - half, c.y - half),
      Point2D::new(c.x + half, c.y + half),
    )
  };
  let t_bake = Instant::now();
  let baked = Baked(Arc::new(BakedSdf::bake(
    |p| analytic.sdf(p) <= 0.0,
    window, bbox, RES,
  )));
  println!("bake ({RES}x{RES} mask + signed EDT): {:?}", t_bake.elapsed());

  // --- same scene as 06, driven by the baked field ---
  let main_de = baked.clone()
    .translate(V2::new(0.4, 0.5))
    .scale(0.5);
  let mut image = RgbaImage::new(2048, 2048);
  let representation = ADF::new(7, vec![
    Primitive::new(sdf::boundary_rect),
    // the Lipschitz bound is picked up from the `Lipschitz` impl automatically
    Primitive::from_shape(main_de.clone()),
  ]);

  util::local_maxima_iter(
    Box::new(|p| representation.sdf(p)),
    32,
    0,
    LineSearch { Δ: 1.0 / 1024.0, ..Default::default() }
  ).filter_map(|local_max| {
    // sample gradient of the (now cheap) distance field at local_max
    let gradient = LineSearch::default().grad(|p| main_de.sdf(p), local_max.point);
    let angle = gradient.angle_from_x_axis();

    let primitive = baked.clone()
      .rotate(angle)
      .translate(local_max.point.to_vector())
      .scale(local_max.distance / 4.0);

    // the shape is scaled to d/4; its bounding-box half-diagonal is ~1.31,
    // so it reaches at most ~0.33·d from the maximum
    unsafe { representation.as_mut() }.insert_within(
      local_max.point,
      local_max.distance * 0.33,
      Primitive::from_shape(primitive)
    ).then_some(())
  }).enumerate()
    .take(20000)
    .for_each(|(i, _)| if i % 1000 == 0 { println!("#{i}"); });

  println!("{representation:#?}");
  representation
    .texture(Luma([255]).to_rgba())
    .draw(&mut image);

  let elapsed = start_time.elapsed();
  println!("Task completed in: {:?}", elapsed);

  image.save(path)?;
  open::that(path)?;
  Ok(())
}
