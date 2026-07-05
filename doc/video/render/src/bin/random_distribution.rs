//! Frame-streaming random distribution render for the explainer video (Scene 7).
//!
//! The crate's `02_random_distribution` example driven to one million circles
//! (ADF max depth 10). A PNG frame goes to stdout on a geometric insertion
//! schedule — ~`FRAMES` frames from the first circles to the full million — so
//! the fill accelerates smoothly when piped into ffmpeg at a fixed framerate.
//! Progress goes to stderr. Driven by `build_assets.py::build_stream_mp4`:
//!
//!   cargo build --release --bin random_distribution
//!   target/release/random_distribution | ffmpeg -f image2pipe -framerate 30 -i - out.mp4

use {
  space_filling::{
    geometry::{Shape, Circle, Translation, Scale, P2},
    sdf::{self, SDF},
    solver::{LineSearch, ADF, Primitive},
    drawing::Draw,
    util,
  },
  image::{codecs::png::PngEncoder, ColorType, ImageEncoder, Luma, Pixel, RgbaImage},
  anyhow::Result,
  rand::prelude::*,
  std::{io::Write, sync::RwLock, time::Instant},
};

type AffineT<T> = Scale<Translation<T, f64>, f64>;

const COUNT: usize = 1_000_000;
const FRAMES: usize = 330;
const FIRST: f64 = 4.0;
const SIZE: u32 = 2048;

/// Verbatim distribution of `examples/gd_adf/02_random_distribution.rs`:
/// given a maximum `(xy, d)`, a random circle is placed within the free ball.
fn random_distribution(representation: &RwLock<ADF<f64>>) -> impl Iterator<Item = AffineT<Circle>> + '_ {
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  util::local_maxima_iter(
    Box::new(|p| representation.read().unwrap().sdf(p)),
    32, 0, LineSearch::default()
  ).filter_map(move |local_max| {
    let circle = {
      use std::f64::consts::PI;

      let angle = rng.gen_range(-PI..=PI);
      let r = (rng.gen_range(0f64..1.0).powf(1.0) * local_max.distance)
        .min(1.0 / 6.0);
      let delta = local_max.distance - r;
      let offset = P2::from([angle.cos(), angle.sin()]) * delta;

      Circle.translate(local_max.point - offset)
        .scale(r)
    };
    representation.write().unwrap().insert_at_maximum(
      local_max,
      Primitive::from_shape(circle)
    ).then(|| circle)
  })
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
  let representation = RwLock::new(
    ADF::new(10, vec![Primitive::new(sdf::boundary_rect)])
      .with_prune_subdiv(8));
  let mut image = RgbaImage::new(SIZE, SIZE);

  let stdout = std::io::stdout();
  let mut out = stdout.lock();

  let ratio = (COUNT as f64 / FIRST).powf(1.0 / FRAMES as f64);
  let mut next = FIRST;
  let mut drawn = 0usize;

  for shape in random_distribution(&representation).take(COUNT) {
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
  emit(&mut out, &image)?; // final frame: the full million
  out.flush()?;
  eprintln!("done: {drawn} circles in {:?}", t0.elapsed());
  Ok(())
}
