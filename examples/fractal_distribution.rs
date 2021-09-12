use {
  space_filling::{
    geometry::{Circle, WorldSpace},
    error::Result,
    sdf::{self, SDF},
    argmax2d::Argmax2D,
    drawing::{Draw, Shape}
  },
  image::{Luma, Pixel}
};

// 158ms, 1000 circles, Δ = 2^-10, chunk = 2^4
fn fractal_distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = Circle<f32, WorldSpace>> + '_ {
  argmax.insert_sdf(sdf::boundary_rect);

  argmax.iter().build()
    .take(1000)
    .map(|(argmax_ret, argmax)| {
      let circle = Circle {
        xy: argmax_ret.point,
        r: argmax_ret.distance / 4.0
      };
      argmax.insert_sdf_domain(
        Argmax2D::domain_empirical(argmax_ret.point, argmax_ret.distance),
        |pixel| circle.sdf(pixel)
      );

      circle
    })
}

fn main() -> Result<()> {
  let path = "out.png";
  let mut argmax = Argmax2D::new(1024, 16)?;
  let mut image = image::RgbaImage::new(2048, 2048);
  fractal_distribution(&mut argmax)
    .for_each(|shape| shape
      .texture(Luma([255]).to_rgba())
      .draw(&mut image));
  image.save(path)?;
  open::that(path)?;
  Ok(())
}