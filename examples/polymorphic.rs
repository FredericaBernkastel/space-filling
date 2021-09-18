use {
  std::sync::Arc,
  space_filling::{
    geometry::{Shape, Circle, Square},
    error::Result,
    sdf,
    argmax2d::Argmax2D,
    drawing::{self, DrawSync}
  },
  image::{RgbaImage, Rgba, DynamicImage},
  euclid::{Vector2D as V2, Point2D, Angle}
};

// 174ms, 1000 circles, Δ = 2^-10, chunk = 2^4
fn polymorphic(argmax: &mut Argmax2D, texture: Arc<DynamicImage>) -> impl Iterator<Item = Box<dyn DrawSync<RgbaImage>>> + '_
{
  use rand::prelude::*;
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  argmax.insert_sdf(sdf::boundary_rect);

  let min_dist = 0.5 * std::f32::consts::SQRT_2 / argmax.resolution as f32;
  argmax.iter()
    .min_dist(min_dist)
    .build()
    .enumerate()
    .map(move |(i, (argmax_ret, argmax))| {
      let shape: Box<dyn DrawSync<_>> = match i % 2 {

        0 => Box::new({
            use std::f32::consts::PI;

            let angle = rng.gen_range::<f32, _>(-PI..=PI);
            let r = (rng.gen_range::<f32, _>(min_dist..1.0).powf(1.0) * argmax_ret.distance)
              .min(1.0 / 6.0);
            let delta = argmax_ret.distance - r;
            let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

          Circle
            .translate(argmax_ret.point - offset)
            .scale(V2::splat(r))
            .texture(texture.clone())
          }),

        1 | _ => Box::new(Square
          .translate(argmax_ret.point.to_vector())
          .scale(V2::splat(argmax_ret.distance / 2.0))
          .rotate(Angle::degrees(rng.gen_range::<f32, _>(0.0..45.0)))
          .texture(Rgba([(argmax_ret.distance.sqrt() * 255.0) as u8, 32, 128, 255])))

      };
      argmax.insert_sdf_domain(
        Argmax2D::domain_empirical(argmax_ret.point, argmax_ret.distance),
        |pixel| shape.sdf(pixel)
      );
      shape
    })
}

fn main() -> Result<()> {
  use rayon::prelude::*;

  let path = "out.png";
  let mut argmax = Argmax2D::new(1024, 16)?;
  let texture = Arc::new(image::open("doc/embedded.jpg")?);
  let shapes = polymorphic(&mut argmax, texture)
    .take(1000).par_bridge();
  drawing::draw_parallel_unsafe(&mut RgbaImage::new(1024, 1024), shapes)
    .save(path)?;
  open::that(path)?;
  Ok(())
}