use {
  space_filling::{
    geometry::{Shape, Circle, WorldSpace, Scale, Translation},
    error::Result,
    sdf::{self, SDF},
    argmax2d::{Argmax2D, ArgmaxResult},
    drawing::Draw
  },
  euclid::{Point2D, Vector2D as V2},
  image::{Luma, Pixel}
};

type AffineT<T> = Scale<Translation<T, f32>, f32>;

pub fn report_progress<'a>(iter: impl Iterator<Item = (ArgmaxResult<f32, WorldSpace>, &'a mut Argmax2D)>, nth: usize)
  -> impl Iterator<Item = (ArgmaxResult<f32, WorldSpace>, &'a mut Argmax2D)> {
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
pub fn embedded(argmax: &mut Argmax2D) -> impl Iterator<Item = AffineT<Circle>> + '_ {
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
      let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

      Circle.translate(argmax_ret.point - offset)
        .scale(V2::splat(r))
    };

    argmax.insert_sdf_domain(
      Argmax2D::domain_empirical(argmax_ret.point, argmax_ret.distance),
      |pixel| circle.sdf(pixel)
    );
  });

  argmax.invert();

  report_progress(
    argmax.iter()
      .min_dist_px(1.0 * std::f32::consts::SQRT_2)
      .build(),
    1000
  ).map(|(argmax_ret, argmax)| {
    let circle = Circle
      .translate(argmax_ret.point.to_vector())
      .scale(V2::splat(argmax_ret.distance / 3.0));

    argmax.insert_sdf_domain(
      Argmax2D::domain_empirical(argmax_ret.point, argmax_ret.distance),
      |pixel| circle.sdf(pixel)
    );

    circle
  })
}

fn main() -> Result<()> {
  let path = "out.png";
  let mut argmax = Argmax2D::new(16384, 64)?;
  let mut image = image::RgbaImage::new(16384, 16384);
  embedded(&mut argmax)
    .take(100000)
    .for_each(|c| c.texture(Luma([255]).to_rgba()).draw(&mut image));
  image.save(path)?;
  open::that(path)?;
  Ok(())
}