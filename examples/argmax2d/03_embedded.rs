use std::time::Instant;
/// Unlike ADF, Argmax2D supports cheap sign inversion, thus it is easy to
/// embed one distribution inside another.

use {
  space_filling::{
    geometry::{Shape, Circle, Scale, Translation, V2},
    sdf::{self, SDF},
    solver::Argmax2D,
    drawing::Draw,
    util
  },
  anyhow::Result,
  image::{Luma, Pixel, RgbaImage}
};

type AffineT<T, P> = Scale<Translation<T, P, 2>, P>;

pub fn report_progress<'a, I>(iter: impl Iterator<Item = I>) -> impl Iterator<Item = I> {
  iter.enumerate()
    .map(move |(i, item)| {
      if i % 1000 == 0 {
        println!("#{i}");
      };
      item
    })
}

pub fn embedded(representation: &mut Argmax2D) -> impl Iterator<Item = AffineT<Circle, f32>> + '_ {
  use rand::prelude::*;
  let mut rng = rand_pcg::Pcg64::seed_from_u64(1);

  representation.insert_sdf(sdf::boundary_rect);

  report_progress(0..100000)
    .for_each(|_| {
      let global_max = representation.find_max();
      let circle = {
        use std::f32::consts::PI;

        let angle = rng.random_range(-PI..=PI);
        let r = (rng.random_range(0f32..1.0).powf(1.0) * global_max.distance)
          .min(1.0 / 4.0);
        let delta = global_max.distance - r;
        let offset = V2::new(angle.cos(), angle.sin()) * delta;

        Circle.translate(global_max.point.coords - offset)
          .scale(r)
      };
      representation.insert_sdf_domain(
        util::domain_global_max(global_max),
        |v| circle.sdf(v)
      );
    });


  representation.invert();


  report_progress(0..).map(|_| {
    let global_max = representation.find_max();
    let circle = Circle
      .translate(global_max.point.coords)
      .scale(global_max.distance / 3.0);

    representation.insert_sdf_domain(
      util::domain_global_max(global_max),
      |v| circle.sdf(v)
    );

    circle
  })
}

// profile, Δ = 2^-14: 119.2s, 
// profile (rayon disjoint parallel writes), Δ = 2^-14: 68.8s
fn main() -> Result<()> {
  let path = "out.png";
  let mut image = RgbaImage::new(16384, 16384);
  embedded(&mut Argmax2D::new(16384, 64)?)
    .take(100000)
    .for_each(|c| c.texture(Luma([255]).to_rgba()).draw(&mut image));
  image.save(path)?;
  open::that(path)?;
  Ok(())
}