use std::time::Instant;
/// An example of user-defined shape.
/// Unsafe lock-free ADF access is used for additional 50% speedup.

use {
  space_filling::{
    sdf::{self, SDF, Lipschitz},
    solver::{ADF, LineSearch, Primitive},
    drawing::Draw,
    geometry::{WorldSpace, BoundingBox, Shape, Scale, Translation},
    util
  },
  euclid::{Point2D, Vector2D as V2, Box2D},
  image::{RgbaImage, Luma, Pixel},
  anyhow::Result,
  num_traits::Float,
  num_complex::Complex,
};

#[derive(Debug, Copy, Clone)]
/// Based on Hubbard-Douady equations, but partial derivatives in the interior behave much better.
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
      // The exterior estimate tends to 0 approaching the boundary, so 0 in the
      // interior keeps the field continuous (discontinuities would
      // make the Lipschitz-based redundancy test unusable).
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

// MandlelDE is a distance *estimator*, not a true SDF: its gradient is not
// bounded by 1. Declaring a larger Lipschitz constant keeps the redundancy test
// conservative for it (nothing contributing is ever dropped or skipped), while
// still letting it be pruned from buckets far away from the set. Declared once
// here, the bound propagates through every combinator chain below —
// rotation, translation and scaling all preserve the Lipschitz constant.
const L_MANDEL: f64 = 4.0;

impl Lipschitz<f64> for MandlelDE {
  fn lipschitz(&self) -> f64 { L_MANDEL }
}

/// scaled to a box [-1, 1]
fn mandel_de_norm<T: Float>() -> Scale<Translation<MandlelDE, T>, T> {
  MandlelDE
    .translate(V2::new(T::from(1.0).unwrap(), T::zero()))
    .scale(T::one() / T::from(1.5).unwrap())
}

// profile (GD pruning), safe, 20k primitives, adf_subdiv = 7, gd_lattice = 1: 51.8s,
// profile (GD pruning), unsafe, 20k primitives, adf_subdiv = 7, gd_lattice = 1: 34.3s
// profile (GD pruning), unsafe, 20k primitives, adf_subdiv = 7, gd_lattice = 3: 165.1s
// profile (Lipschitz B&B pruning, per-primitive bounds), unsafe: 19.2s
// profile (Lipschitz B&B pruning, per-primitive bounds, sound insertion domains via D*-pruned walk), unsafe: 22.6s
fn main() -> Result<()> {
  let start_time = Instant::now();

  let path = "out.png";
  let main_de = mandel_de_norm()
    .translate(V2::new(0.4, 0.5))
    .scale(0.5);
  let mut image = RgbaImage::new(2048, 2048);
  let representation = ADF::new(7, vec![
    Primitive::new(sdf::boundary_rect),
    // the Lipschitz bound is picked up from the `Lipschitz` impl automatically
    Primitive::from_shape(main_de),
  ]);

  util::local_maxima_iter(
    Box::new(|p| representation.sdf(p)),
    32,
    0,
    LineSearch { Δ: 1.0 / 1024.0, ..Default::default() }
  ).filter_map(|local_max| {
    // sample gradient of the distance field at local_max
    let gradient = LineSearch::default().grad(|p| main_de.sdf(p), local_max.point);
    // gradient direction
    let angle = gradient.angle_from_x_axis();

    let primitive = mandel_de_norm()
      .rotate(angle)
      .translate(local_max.point.to_vector())
      .scale(local_max.distance / 4.0);

    // alternately use safe RwLock<ADF> or imperative style
    // the shape is scaled to d/4; its bounding-box half-diagonal is ~1.31,
    // so it reaches at most ~0.33·d from the maximum
    unsafe { representation.as_mut() }.insert_within(
      local_max.point,
      local_max.distance * 0.33,
      Primitive::from_shape(primitive)
    ).then(|| primitive)
  }).enumerate()
    .take(20000)
    .for_each(|(i, _)| if i % 1000 == 0 { println!("#{i}"); });

  println!("{representation:#?}");
  // ADF implements SDF - combining all primitives into one complex distance function.
  // Therefore, Draw is implemented automatically as well, making it possible to display the field
  // with a single call. Slightly faster than drawing each shape separately.
  representation
    .texture(Luma([255]).to_rgba())
    .draw(&mut image);

  let elapsed = start_time.elapsed();
  println!("Task completed in: {:?}", elapsed);

  image.save(path)?;
  open::that(path)?;
  Ok(())
}