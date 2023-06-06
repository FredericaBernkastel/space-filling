/// An example of user-defined shape.
/// Unsafe lock-free ADF access is used for additional 50% speedup.

use {
  space_filling::{
    sdf::{self, SDF},
    solver::{adf::ADF, line_search::LineSearch},
    drawing::Draw,
    geometry::{WorldSpace, BoundingBox, Shape, Scale, Translation},
    util
  },
  euclid::{Point2D, Vector2D as V2, Box2D},
  image::{RgbaImage, Luma, Pixel},
  anyhow::Result,
  num_traits::Float,
  num_complex::Complex,
  std::sync::Arc
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
    let mut result = z.norm() * z.norm().ln() / dz.norm();
    if is_inside {
      result = -result - T::from(128.0).unwrap();
    }
    result
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

// profile, safe: 51.8s, 20k primitives, adf_subdiv = 7, gd_lattice = 1
// unsafe: 34.3s
// unsafe, gd_lattice = 3: 165.1s
fn main() -> Result<()> {
  let path = "out.png";
  let main_de = mandel_de_norm()
    .translate(V2::new(0.4, 0.5))
    .scale(0.5);
  let mut image = RgbaImage::new(2048, 2048);
  let representation = ADF::new(7, vec![
    Arc::new(sdf::boundary_rect),
    Arc::new(move |p| main_de.sdf(p))
  ]).with_gd_lattice_density(1);

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
    unsafe { representation.as_mut() }.insert_sdf_domain(
      util::domain_empirical(local_max),
      Arc::new(move |p| primitive.sdf(p))
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

  image.save(path)?;
  open::that(path)?;
  Ok(())
}