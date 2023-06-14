/// Generate a random distribution using ADF representation.
/// Iterator code style is shown, which is lazy evaluated; fully parallel.

use {
  space_filling::{
    geometry::{Shape, Circle, Translation, Scale, P2},
    sdf::{self, SDF},
    solver::{LineSearch, ADF},
    drawing::Draw,
    util
  },
  image::{Luma, Pixel},
  anyhow::Result,
  rand::prelude::*,
  std::sync::{Arc, RwLock}
};

type AffineT<T> = Scale<Translation<T, f64>, f64>;

// profile: 62ms, 1000 circrles, adf_subdiv = 5, gd_lattice = 1
fn random_distribution(representation: &RwLock<ADF<f64>>) -> impl Iterator<Item = AffineT<Circle>> + '_  {
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
      // polar to cartesian
      let offset = P2::from([angle.cos(), angle.sin()]) * delta;

      Circle.translate(local_max.point - offset)
        .scale(r)
    };
    representation.write().unwrap().insert_sdf_domain(
      util::domain_empirical(local_max),
      Arc::new(move |p| circle.sdf(p))
    ).then(|| circle)
  })
}

fn main() -> Result<()> {
  let path = "out.png";
  let representation = RwLock::new(
    ADF::new(5, vec![Arc::new(sdf::boundary_rect)])
      .with_gd_lattice_density(3)); // set ADF to a high precision
  let mut image = image::RgbaImage::new(2048, 2048);

  random_distribution(&representation)
    .take(1000)
    .for_each(|shape| shape
      .texture(Luma([255u8]).to_rgba())
      .draw(&mut image));

  image.save(path)?;
  open::that(path)?;
  Ok(())
}