use {
  space_filling::{
    geometry::{Shape, Circle, Translation, Scale},
    error::Result,
    sdf::{self, SDF},
    solver::argmax2d::Argmax2D,
    drawing::Draw
  },
  image::{Luma, Pixel},
  euclid::Point2D
};

type AffineT<T> = Scale<Translation<T, f32>, f32>;

// 104ms, 1000 circrles, Δ = 2^-10, chunk = 2^4
fn random_distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = AffineT<Circle>> + '_ {
  use rand::prelude::*;
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  argmax.insert_sdf(sdf::boundary_rect);

  let min_dist = 0.5 * std::f32::consts::SQRT_2 / argmax.resolution() as f32;
  argmax.iter()
    .min_dist(min_dist)
    .build()
    .map(move |(argmax_ret, argmax)| {
      let circle = {
        use std::f32::consts::PI;

        let angle = rng.gen_range::<f32, _>(-PI..=PI);
        let r = (rng.gen_range::<f32, _>(min_dist..1.0).powf(1.0) * argmax_ret.distance)
          .min(1.0 / 6.0);
        let delta = argmax_ret.distance - r;
        // polar to cartesian
        let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

        Circle.translate(argmax_ret.point - offset)
          .scale(r)
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
  random_distribution(&mut argmax)
    .take(1000)
    .for_each(|shape| shape
      .texture(Luma([255u8]).to_rgba())
      .draw(&mut image));
  image.save(path)?;
  open::that(path)?;
  Ok(())
}