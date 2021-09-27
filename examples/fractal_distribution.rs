use {
  space_filling::{
    geometry::{Shape, Circle, Translation, Scale},
    error::Result,
    sdf::{self, SDF},
    solver::argmax2d::Argmax2D,
    drawing::Draw
  },
  image::{Luma, Pixel}
};

type AffineT<T> = Scale<Translation<T, f32>, f32>;

// 158ms, 1000 circles, Δ = 2^-10, chunk = 2^4
fn fractal_distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = AffineT<Circle>> + '_ {
  argmax.insert_sdf(sdf::boundary_rect);

  argmax.iter().build()
    .map(|(argmax_ret, argmax)| {
      let circle = Circle
        .translate(argmax_ret.point.to_vector())
        .scale(argmax_ret.distance / 4.0);
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
    .take(1000)
    .for_each(|shape| shape
      .texture(Luma([255u8]).to_rgba())
      .draw(&mut image));
  image.save(path)?;
  open::that(path)?;
  Ok(())
}
