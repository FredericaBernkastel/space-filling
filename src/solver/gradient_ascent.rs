#![allow(dead_code)]
#![allow(non_snake_case)]
use crate::{
  sdf::{self, SDF}
};
use euclid::{Point2D, Vector2D as V2};
use crate::geometry::WorldSpace;
use crate::solver::DistPoint;
use rand_pcg::Lcg128Xsl64;
use rand::Rng;

pub struct GradientAscent<T> {
  dist_field: T,
  Δ: f64,
  rng_state: Lcg128Xsl64
}

impl <T> GradientAscent<Vec<T>> {
  pub fn new(Δ: f64) -> Self {
    use rand::prelude::*;
    Self {
      dist_field: vec![],
      Δ: Δ / 2.0,
      rng_state: Lcg128Xsl64::seed_from_u64(0)
    }
  }
}

impl <T> GradientAscent<Vec<T>>
  where T: SDF<f64>
{
  pub fn insert_sdf(&mut self, sdf: T) {
    self.dist_field.push(sdf);
  }

  pub fn sample_sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    let sdf = self.dist_field.iter()
      .map(move |s| s.sdf(pixel))
      .reduce(|a, b| a.min(b))
      .unwrap_or(f64::MAX / 2.0);
    sdf.min(sdf::boundary_rect(pixel))
  }

  pub fn Δf(&self, p: Point2D<f64, WorldSpace>) -> V2<f64, WorldSpace> {
    let Δp = self.Δ;
    V2::new(
      self.sample_sdf(p + V2::new(Δp, 0.0)) - self.sample_sdf(p - V2::new(Δp, 0.0)),
      self.sample_sdf(p + V2::new(0.0, Δp)) - self.sample_sdf(p - V2::new(0.0, Δp)),
    ) / (Δp * 2.0)
  }

  fn ascend(&self, mut p: Point2D<f64, WorldSpace>) -> Point2D<f64, WorldSpace> {
    let mut step_size = 1.0;
    let decay_factor = 0.85f64;
    let step_limit = Some(40);
    let gradient_tolerance = self.Δ;
    for _ in 0..step_limit.unwrap_or(usize::MAX) {
      let grad = self.Δf(p) * step_size;
      if grad.length() < gradient_tolerance { break; }
      step_size *= decay_factor;
      p += grad;
    }
    p
  }

  fn find_local_max(&mut self) -> Option<DistPoint<f64, f64, WorldSpace>> {
    let num_trials = 100;

    (0..num_trials).find_map(|_| {
      let p0 = Point2D::new(
        self.rng_state.gen_range(0.0..1.0),
        self.rng_state.gen_range(0.0..1.0)
      );
      let p1 = self.ascend(p0);
      let p1 = DistPoint {
        point: p1,
        distance: self.sample_sdf(p1)
      };
      (p1.distance > self.Δ).then(|| p1)
    })
  }

  // for debugging only
  pub fn trajectory(&self, mut p: Point2D<f64, WorldSpace>) -> Vec<Point2D<f64, WorldSpace>> {
    let mut trajectory = vec![p];
    let mut step_size = 0.5;
    let decay_factor = 0.95f64;
    // decay -> limit:
    // fast: 0.5 -> 20
    // slow: 0.85 -> 40
    // veryslow: 0.95 -> 128
    let step_limit = Some(160);
    let gradient_tolerance = self.Δ;
    for _ in 0..step_limit.unwrap_or(usize::MAX) {
      let grad = self.Δf(p) * step_size;
      if grad.length() < gradient_tolerance {
        break;
      }
      step_size *= decay_factor;
      p += grad;
      trajectory.push(p)
    }
    trajectory
  }
}

#[cfg(test)] mod tests {
  use super::*;
  use crate::geometry::{Circle, Scale, Translation, Shape};
  use crate::error::Result;
  use crate::drawing::Draw;
  use image::{Luma, Pixel};

  type AffineT<T> = Scale<Translation<T, f64>, f64>;

  #[test] #[ignore] fn test() -> Result<()> {
    use rand::prelude::*;

    let Δp = 1.0 / 1024.0;
    let mut grad = GradientAscent::<Vec<AffineT<Circle>>>::new(Δp);
    let mut image = image::RgbaImage::new(1024, 1024);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let t0 = std::time::Instant::now();
    for _ in 0..1000 {
      let local_max = match grad.find_local_max() {
        Some(x) => x,
        None => break
      };
      let circle = {
        use std::f64::consts::PI;

        let angle = rng.gen_range(-PI..=PI);
        let r = (rng.gen_range(Δp..1.0).powf(1.0) * local_max.distance)
          .min(1.0 / 6.0);
        let delta = local_max.distance - r;
        // polar to cartesian
        let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

        Circle.translate(local_max.point - offset)
          .scale(r)
      };
      grad.insert_sdf(circle);
    }
    println!("{}ms", t0.elapsed().as_millis());
    grad.dist_field.iter().for_each(|c| c
      .texture(Luma([255]).to_rgba())
      .draw(&mut image));
    image.save("test/test_grad.png")?;
    Ok(())
  }
}