use space_filling::{
  geometry::{Circle, WorldSpace},
  error::Result,
  sdf::{self, SDF},
  argmax2d::Argmax2D,
  drawing
};
use euclid::Point2D;

// 104ms, 1000 circrles, Δ = 2^-10, chunk = 2^4
pub fn random_distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = Circle<f32, WorldSpace>> + '_ {
  use rand::prelude::*;
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  argmax.insert_sdf(sdf::boundary_rect);

  let min_dist = 0.5 * std::f32::consts::SQRT_2 / argmax.resolution as f32;
  argmax.iter()
    .min_dist(min_dist)
    .build()
    .take(1000)
    .map(move |(argmax_ret, argmax)| {
      let circle = {
        use std::f32::consts::PI;

        let angle = rng.gen_range::<f32, _>(-PI..=PI);
        let r = (rng.gen_range::<f32, _>(min_dist..1.0).powf(1.0) * argmax_ret.distance)
          .min(1.0 / 6.0);
        let delta = argmax_ret.distance - r;
        // polar to cartesian
        let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

        Circle {
          xy: (argmax_ret.point - offset).to_point(), r
        }
      };
      argmax.insert_sdf_domain(
        Argmax2D::domain_empirical(circle.xy, argmax_ret.distance),
        |pixel| circle.sdf(pixel)
      );

      circle
    })
}

fn main() -> Result<()> {
  let path = "out.png";
  let mut argmax = Argmax2D::new(1024, 16)?;
  let circles = random_distribution(&mut argmax);
  drawing::draw_circles(
    circles,
    (2048, 2048).into()
  ).save(path)?;
  open::that(path)?;
  Ok(())
}