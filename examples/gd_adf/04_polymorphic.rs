use {
  std::sync::{Arc, RwLock},
  space_filling::{
    geometry::{Shape, Ring, Square},
    sdf::{self, SDF},
    solver::{adf::ADF, line_search::LineSearch},
    drawing::{self, Draw},
    util
  },
  image::{RgbaImage, Rgba, Luma, Pixel, DynamicImage},
  anyhow::Result,
  rand::prelude::*,
  euclid::{Point2D, Angle}
};

fn polymorphic(representation: &RwLock<ADF<f64>>, texture: Arc<DynamicImage>)
  -> impl Iterator<Item = Arc<dyn Draw<f64, RgbaImage> + Send + Sync>> + '_
{
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  util::local_maxima_iter(
    Box::new(|p| representation.read().unwrap().sdf(p)),
    32, 0, LineSearch::default()
  ) .enumerate()
    .filter_map(move |(i, local_max)| {
      let shape: Arc<dyn Draw<_, _> + Send + Sync> = match i % 2 {

        0 => Arc::new({
          use std::f64::consts::PI;

          let angle = rng.gen_range(-PI..=PI);
          let r = (rng.gen_range(0.0..1.0) * local_max.distance)
            .min(1.0 / 6.0);
          let delta = local_max.distance - r;
          let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

          Ring { inner_r: 0.5 }
            .translate(local_max.point - offset)
            .scale(r)
            .texture(texture.clone())
        }),

        1 | _ => Arc::new(Square
          .translate(local_max.point.to_vector())
          .scale(local_max.distance / 2.0)
          .rotate(Angle::degrees(rng.gen_range(0.0..45.0)))
          .texture(Rgba([
            ((local_max.distance * 2.0).sqrt() * 255.0) as u8,
            32,
            rng.gen_range(64..128),
            255]
          )))

      };
      representation.write().unwrap().insert_sdf_domain(
        util::domain_empirical(local_max),
        Arc::new({
          let shape = shape.clone();
          move |v| shape.sdf(v)
        })
      ).then(|| shape)
  })
}

fn main() -> Result<()> {
  use rayon::prelude::*;

  let path = "out.png";
  let mut representation = RwLock::new(
    ADF::new(5, vec![Arc::new(sdf::boundary_rect)])
      .with_gd_lattice_density(2)
  );
  let texture = Arc::new(image::open("doc/fractal_distribution.png")?);
  let shapes = polymorphic(&mut representation, texture)
    .take(1000).par_bridge();
  drawing::draw_parallel(&mut RgbaImage::from_pixel(1024, 1024, Luma([255]).to_rgba()), shapes)
    .save(path)?;
  open::that(path)?;
  Ok(())
}