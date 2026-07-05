//! Frame-streaming Mandelbrot-estimator render for the explainer video (Scene 8).
//!
//! The crate's `06_custom_primitive` example — 20k instances of a Mandelbrot
//! distance estimator (declared 4-Lipschitz) fitted by GD-ADF — with each
//! accepted primitive drawn incrementally. A PNG frame goes to stdout on a
//! geometric insertion schedule (~`FRAMES` frames from the first mandelbrot to
//! all 20k), so the fill accelerates smoothly when piped into ffmpeg at a fixed
//! framerate. Progress goes to stderr. Driven by `build_assets.py::build_stream_mp4`:
//!
//!   cargo build --release --bin custom_primitive
//!   target/release/custom_primitive | ffmpeg -f image2pipe -framerate 30 -i - out.mp4

use {
  space_filling::{
    sdf::{self, SDF, Lipschitz},
    solver::{ADF, LineSearch, Primitive},
    drawing::Draw,
    geometry::{WorldSpace, BoundingBox, Shape, Scale, Translation},
    util,
  },
  euclid::{Point2D, Vector2D as V2, Box2D},
  image::{codecs::png::PngEncoder, ColorType, ImageEncoder, Luma, Pixel, RgbaImage},
  anyhow::Result,
  num_traits::Float,
  num_complex::Complex,
  std::{io::Write, sync::RwLock, time::Instant},
};

const COUNT: usize = 20_000;
const FRAMES: usize = 300;
const FIRST: f64 = 1.0;
const SIZE: u32 = 2048;

/// Verbatim copy of the estimator in `examples/gd_adf/06_custom_primitive.rs`
/// (the example defines it privately; it is not part of the library API).
#[derive(Debug, Copy, Clone)]
struct MandlelDE;

impl<T: Float> SDF<T> for MandlelDE {
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

fn emit(out: &mut impl Write, image: &RgbaImage) -> Result<()> {
  let mut buf: Vec<u8> = Vec::new();
  PngEncoder::new(&mut buf)
    .write_image(image.as_raw(), image.width(), image.height(), ColorType::Rgba8)?;
  out.write_all(&buf)?;
  Ok(())
}

fn main() -> Result<()> {
  let t0 = Instant::now();
  let main_de = mandel_de_norm()
    .translate(V2::new(0.4, 0.5))
    .scale(0.5);
  let mut image = RgbaImage::new(SIZE, SIZE);
  let representation = RwLock::new(ADF::new(7, vec![
    Primitive::new(sdf::boundary_rect),
    Primitive::from_shape(main_de),
  ]));

  let stdout = std::io::stdout();
  let mut out = stdout.lock();

  // the seed shape is part of the field from the start — draw and show it first
  main_de.texture(Luma([255u8]).to_rgba()).draw(&mut image);
  emit(&mut out, &image)?;

  let ratio = (COUNT as f64 / FIRST).powf(1.0 / FRAMES as f64);
  let mut next = FIRST;
  let mut drawn = 0usize;

  let inserted = util::local_maxima_iter(
    Box::new(|p| representation.read().unwrap().sdf(p)),
    32, 0, LineSearch::default()
  ).filter_map(|local_max| {
    let gradient = LineSearch::default().grad(|p| main_de.sdf(p), local_max.point);
    let angle = gradient.angle_from_x_axis();

    let primitive = mandel_de_norm()
      .rotate(angle)
      .translate(local_max.point.to_vector())
      .scale(local_max.distance / 4.0);

    representation.write().unwrap().insert_within(
      local_max.point,
      local_max.distance * 0.33,
      Primitive::from_shape(primitive)
    ).then(|| primitive)
  });

  for shape in inserted.take(COUNT) {
    shape.texture(Luma([255u8]).to_rgba()).draw(&mut image);
    drawn += 1;
    if drawn as f64 >= next {
      emit(&mut out, &image)?;
      eprintln!("#{drawn} ({:.0?})", t0.elapsed());
      while next <= drawn as f64 {
        next *= ratio;
      }
    }
  }
  emit(&mut out, &image)?; // final frame: all 20k
  out.flush()?;
  eprintln!("done: {drawn} mandelbrots in {:?}", t0.elapsed());
  Ok(())
}
