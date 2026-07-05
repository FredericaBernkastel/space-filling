//! Frame-streaming fractal distribution render for the explainer video (Scene 3).
//!
//! Same distribution as the crate's `01_fractal_distribution` example, but writes
//! a PNG to stdout every `STRIDE` insertions so the sequence can be piped into
//! ffmpeg. Progress goes to stderr to keep stdout a clean PNG stream. Driven by
//! `doc/video/build_assets.py::build_fractal_mp4`:
//!
//!   cargo build --release --bin fractal_distribution
//!   target/release/fractal_distribution | ffmpeg -f image2pipe -framerate 30 -i - out.mp4

use {
  space_filling::{
    geometry::{Shape, Circle},
    sdf::{self, SDF},
    solver::Argmax2D,
    drawing::Draw,
    util,
  },
  image::{codecs::png::PngEncoder, ColorType, ImageEncoder, Luma, Pixel, RgbaImage},
  anyhow::Result,
  std::io::Write,
};

const COUNT: usize = 1000;
const STRIDE: usize = 6;

fn emit(out: &mut impl Write, image: &RgbaImage) -> Result<()> {
  let mut buf: Vec<u8> = Vec::new();
  PngEncoder::new(&mut buf)
    .write_image(image.as_raw(), image.width(), image.height(), ColorType::Rgba8)?;
  out.write_all(&buf)?;
  Ok(())
}

fn main() -> Result<()> {
  let mut representation = Argmax2D::new(1024, 16)?;
  let mut image = RgbaImage::new(1024, 1024);
  representation.insert_sdf(sdf::boundary_rect);

  let stdout = std::io::stdout();
  let mut out = stdout.lock();

  for i in 0..COUNT {
    let global_max = representation.find_max();
    let circle = Circle
      .translate(global_max.point.to_vector())
      .scale(global_max.distance / 4.0);
    representation.insert_sdf_domain(
      util::domain_global_max(global_max),
      |v| circle.sdf(v),
    );
    circle.texture(Luma([255u8]).to_rgba()).draw(&mut image);

    if i % STRIDE == 0 {
      emit(&mut out, &image)?;
      if i % 100 == 0 {
        eprintln!("#{i}");
      }
    }
  }
  emit(&mut out, &image)?; // final frame
  out.flush()?;
  Ok(())
}
