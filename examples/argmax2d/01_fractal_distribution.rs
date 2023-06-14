/// Generate a fractal-like pattern using discrete distance field representation.
/// Simple impeative code style is shown.

use {
  space_filling::{
    geometry::{Shape, Circle},
    sdf::{self, SDF},
    solver::Argmax2D,
    drawing::Draw,
    util
  },
  anyhow::Result,
  image::{Luma, Pixel, RgbaImage}
};

fn fractal_distribution(representation: &mut Argmax2D, image: &mut RgbaImage) {
  representation.insert_sdf(sdf::boundary_rect);

  for _ in 0..1000 {
    let global_max = representation.find_max();
    let circle = Circle
      .translate(global_max.point.to_vector())
      .scale(global_max.distance / 4.0);
    representation.insert_sdf_domain(
      util::domain_empirical(global_max),
      |v| circle.sdf(v)
    );
    circle
      .texture(Luma([255u8]).to_rgba())
      .draw(image);
  }
}

// profile: 158ms, 1000 circles, Δ = 2^-10
fn main() -> Result<()> {
  let path = "out.png";
  let mut representation = Argmax2D::new(1024, 16)?;
  let mut image = RgbaImage::new(2048, 2048);

  fractal_distribution(&mut representation, &mut image);

  image.save(path)?;
  open::that(path)?;
  Ok(())
}
