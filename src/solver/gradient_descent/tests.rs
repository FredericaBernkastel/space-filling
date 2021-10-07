use {
  super::*,
  crate::{
    geometry::{self, Circle, Shape},
    drawing::Draw,
    sdf::{self, SDF},
    solver::{Argmax2D, z_order_storage::ZOrderStorage},
    error::Result
  },
  image::{Luma, Pixel, Rgba}
};

#[test] #[ignore] fn gradient() -> Result<()> {
  let mut grad = GradientDescent::<Vec<Box<dyn Fn(_) -> _ + Send + Sync>>, _>
    ::new(LineSearchConfig::default());
  grad.insert_sdf(Box::new(sdf::boundary_rect));
  grad.insert_sdf(Box::new(|p| Circle
    .translate(V2::splat(0.25))
    .scale(0.25)
    .sdf(p)));
  grad
    .display_sdf(3.0, Some(18))
    .save("test/test_grad.png")?;
  Ok(())
}

#[test] #[ignore] fn trajectory() -> Result<()> {
  let mut grad = GradientDescent::<Vec<Box<dyn Fn(_) -> _ + Send + Sync>>, _>
    ::new(LineSearchConfig::default());
  grad.insert_sdf(Box::new(sdf::boundary_rect));
  grad.insert_sdf(Box::new(|p| Circle
    .translate(V2::splat(0.25))
    .scale(0.25)
    .sdf(p)));
  grad.trajectory_animation(
    3.0,
    |i, img| {
      img.save(format!("test/test_grad/{}.png", i)).ok();
    }
  );
  Ok(())
}

// profile: 69ms
#[test] #[ignore] fn z_order_backend() -> Result<()> {
  use rand::prelude::*;

  let resolution = 1024;
  let config = LineSearchConfig {
    Δ: 1.0 / resolution as f32,
    ..Default::default()
  };
  let mut grad = GradientDescent::<ZOrderStorage<_>, f32>
    ::new(config, resolution, 16)?;
  let mut image = image::RgbaImage::new(1024, 1024);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  grad.insert_sdf(sdf::boundary_rect);

  grad.iter().build()
    .take(1000)
    .for_each(|(local_max, grad)| {
      let circle = {
        use std::f32::consts::PI;

        let angle = rng.gen_range::<f32, _>(-PI..=PI);
        let r = (rng.gen_range::<f32, _>(config.Δ..1.0).powf(1.0) * local_max.distance)
          .min(1.0 / 6.0);
        let delta = local_max.distance - r;
        let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

        Circle.translate(local_max.point - offset)
          .scale(r)
      };
      grad.insert_sdf_domain(
        Argmax2D::domain_empirical(local_max.point, local_max.distance),
        |p| circle.sdf(p)
      );

      circle
        .texture(|_| Luma([255u8]).to_rgba())
        .draw(&mut image);
    });

  image.save("test/test_grad.png")?;
  Ok(())
}

// profile: 22ms
#[test] #[ignore] fn diffusion_limited_aggregation() -> Result<()> {
  use rand::prelude::*;

  let config = LineSearchConfig {
    step_limit: Some(0),
    ..Default::default()
  };
  let size_bound = 1.0 / 128.0;

  let mut grad = GradientDescent::<Vec<Box<dyn Fn(_) -> _ + Send + Sync>>, _>
    ::new(config);
  let mut image = image::RgbaImage::new(1024, 1024);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  // seed object
  let cross = geometry::HolyCross
    .translate(V2::splat(0.5))
    .scale(0.25);
  cross
    .texture(Luma([0]).to_rgba())
    .draw(&mut image);
  grad.insert_sdf(Box::new(move |p| cross.sdf(p)));

  grad.iter().build()
    .take(1000)
    .for_each(|(local_max, grad)| {
      let shape = {
        let Δ = grad.Δf(local_max.point).normalize();
        let r = rng.gen_range(grad.line_config.Δ..1.0).powf(1.0) * local_max.distance.min(size_bound);
        let offset = local_max.point - Δ * (local_max.distance - r);

        Circle
          .translate(offset.to_vector())
          .scale(r)
      };
      grad.insert_sdf(Box::new(move |p| shape.sdf(p)));
      shape.texture(Rgba([255, 0, 0, 255]))
        .draw(&mut image);
    });

  image.save("test/test_grad.png")?;
  Ok(())
}
