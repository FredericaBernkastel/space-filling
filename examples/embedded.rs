use space_filling::{
  geometry::{Circle, Point},
  error::Result,
  sdf::{self, SDF},
  argmax2d::{Argmax2D, ArgmaxResult},
  drawing
};

pub fn report_progress<'a>(iter: impl Iterator<Item = (ArgmaxResult<f32>, &'a mut Argmax2D)>, nth: usize)
  -> impl Iterator<Item = (ArgmaxResult<f32>, &'a mut Argmax2D)> {
  iter.enumerate()
    .map(move |(i, (argmax_ret, argmax))| {
      if i % nth == 0 {
        println!("#{} argmax ∇ {}", i, argmax_ret.distance * argmax.resolution as f32);
      };
      (argmax_ret, argmax)
    })
}

/// A regular distribution embedded in a random one
/// 88.4s, 100'000 circrles, Δ = 2^-14, chunk = 2^6
pub fn embedded(argmax: &mut Argmax2D) -> impl Iterator<Item = Circle> + '_ {
  use rand::prelude::*;
  let mut rng = rand_pcg::Pcg64::seed_from_u64(1);

  argmax.insert_sdf(sdf::boundary_rect);

  let min_dist = 3.0 * std::f32::consts::SQRT_2 / argmax.resolution as f32;
  report_progress(
    argmax.iter()
      .min_dist(min_dist)
      .build()
      .take(100000),
    1000
  ).for_each(|(argmax_ret, argmax)| {
    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(min_dist..1.0).powf(1.0) * argmax_ret.distance)
        .min(1.0 / 4.0);
      let delta = argmax_ret.distance - r;
      let offset = Point { x: delta * angle.cos(), y: delta * angle.sin() };

      Circle {
        xy: argmax_ret.point.translate(offset), r
      }
    };

    argmax.insert_sdf_domain(
      Argmax2D::domain_empirical(circle.xy, argmax_ret.distance).into(),
      |pixel| circle.sdf(pixel)
    );
  });

  argmax.invert();

  report_progress(
    argmax.iter()
      .min_dist_px(1.0 * std::f32::consts::SQRT_2)
      .build()
      .take(100000),
    1000
  ).map(|(argmax_ret, argmax)| {
    let circle = Circle {
      xy: argmax_ret.point, r: argmax_ret.distance / 3.0
    };

    argmax.insert_sdf_domain(
      Argmax2D::domain_empirical(circle.xy, argmax_ret.distance).into(),
      |pixel| circle.sdf(pixel)
    );

    circle
  })
}

fn main() -> Result<()> {
  let path = "out.png";
  let mut argmax = Argmax2D::new(16384, 64)?;
  let circles = embedded(&mut argmax);
  drawing::draw_circles(
    circles,
    (16384, 16384).into()
  ).save(path)?;
  open::that(path)?;
  Ok(())
}