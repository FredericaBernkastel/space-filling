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
  let mut grad = GradientDescent::<ZOrderStorage<_>, _>
    ::new(LineSearchConfig::default(), 1024, 16)?;
  grad.insert_sdf(sdf::boundary_rect);
  grad.insert_sdf(|p| Circle
    .translate(V2::splat(0.25))
    .scale(0.25)
    .sdf(p));
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

  let mut grad = GradientDescent::<ZOrderStorage<_>, f32>
    ::new(LineSearchConfig::default(), 1024, 16)?;
  let mut image = image::RgbaImage::new(512, 512);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles = vec![];
  grad.insert_sdf(sdf::boundary_rect);

  let min_dist = 0.5 * std::f32::consts::SQRT_2 / image.width() as f32;
  let t0 = std::time::Instant::now();
  for _ in 0..1000 {
    let local_max = match grad.find_local_max(&mut rng) {
      Some(x) => x,
      None => {
        println!("local_max = None");
        break
      }
    };
    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(min_dist..1.0).powf(1.0) * local_max.distance)
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
    circles.push(circle);
  };
  println!("{}ms", t0.elapsed().as_millis());

  circles.into_iter().for_each(|c| c
    .texture(Luma([255]).to_rgba())
    .draw(&mut image));
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
  let mut grad = GradientDescent::<Vec<Box<dyn Fn(_) -> _ + Send + Sync>>, _>
    ::new(config);
  let mut image = image::RgbaImage::new(1024, 1024);
  // seed object
  let cross = geometry::HolyCross
    .translate(V2::splat(0.5))
    .scale(0.25);
  grad.insert_sdf(Box::new(move |p| cross.sdf(p)));
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles = vec![];
  let size_bound = 1.0 / 128.0;
  let t0 = std::time::Instant::now();
  let mut i = 0;
  while i < 1000 {
    let local_max = match grad.find_local_max(&mut rng) {
      Some(x) => x,
      None => break
    };
    let circle = {
      let Δ = grad.Δf(local_max.point).normalize();
      let r = rng.gen_range(grad.line_config.Δ..1.0).powf(1.0) * local_max.distance.min(size_bound);
      let offset = local_max.point - Δ * (local_max.distance - r);

      Circle
        .translate(offset.to_vector())
        .scale(r)
    };
    grad.insert_sdf(Box::new(move |p| circle.sdf(p)));
    circles.push(circle);
    i += 1;
  }
  println!("{}ms", t0.elapsed().as_millis());
  cross
    .texture(Luma([0]).to_rgba())
    .draw(&mut image);
  circles.into_iter().for_each(|c| {
    c.texture(Rgba([255, 0, 0, 255]))
      .draw(&mut image);
  });
  image.save("test/test_grad.png")?;
  Ok(())
}

// profile: 142ms
#[test] #[ignore] fn iter_interface() -> Result<()> {
  let mut grad = GradientDescent::<ZOrderStorage<_>, f64>
    ::new(LineSearchConfig::default(), 1024, 16)?;
  let mut image = image::RgbaImage::new(512, 512);
  grad.insert_sdf(sdf::boundary_rect);
  grad.iter().build()
    .take(1000)
    .for_each(|(local_max, grad)| {
      let circle = Circle
        .translate(local_max.point.to_vector())
        .scale(local_max.distance / 4.0);
      grad.insert_sdf_domain(
        Argmax2D::domain_empirical(local_max.point, local_max.distance),
        move |p| circle.sdf(p)
      );
      circle.texture(Luma([255]).to_rgba())
        .draw(&mut image);
    });
  image.save("test/test_grad.png")?;
  Ok(())
}