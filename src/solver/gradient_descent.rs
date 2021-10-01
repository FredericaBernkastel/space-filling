#![allow(dead_code)]
#![allow(non_snake_case)]
use crate::{
  error::Result
};
use euclid::{Point2D, Vector2D as V2, Rect, Size2D, Box2D};
use crate::geometry::WorldSpace;
use crate::solver::DistPoint;
use rand_pcg::Lcg128Xsl64;
use rand::Rng;
use num_traits::{Float, Signed};
use crate::solver::z_order_storage::ZOrderStorage;
use crate::sdf;

#[derive(Copy, Clone)]
pub struct LineSearchConfig<P> {
  Δ: P,
  initial_step_size: P,
  decay_factor: P,
  step_limit: Option<u64>,
  max_attempts: u64,
  control_factor: P
}

impl <P: Float> Default for LineSearchConfig<P> {
  fn default() -> Self {
    LineSearchConfig {
      Δ: P::from(1.0 / 2048.0).unwrap(),
      initial_step_size: P::one(),
      decay_factor: P::from(0.85).unwrap(),
      step_limit: Some(40),
      max_attempts: 50,
      control_factor: P::one()
    }}}

pub trait LineSearch<P: Float> {
  fn config(&self) -> LineSearchConfig<P>;
  fn sample_sdf(&self, pixel: Point2D<P, WorldSpace>) -> P;

  fn Δf(&self, p: Point2D<P, WorldSpace>) -> V2<P, WorldSpace> {
    let Δp = self.config().Δ;
    V2::new(
      self.sample_sdf(p + V2::new(Δp, P::zero())) - self.sample_sdf(p - V2::new(Δp, P::zero())),
      self.sample_sdf(p + V2::new(P::zero(), Δp)) - self.sample_sdf(p - V2::new(P::zero(), Δp)),
    ) / (Δp * (P::one() + P::one()))
  }
  fn ascend(&self, mut p: Point2D<P, WorldSpace>) -> Point2D<P, WorldSpace> {
    let config = self.config();
    let mut step_size = config.initial_step_size;
    for _ in 0..config.step_limit.unwrap_or(u64::MAX) {
      let grad = self.Δf(p) * step_size;
      if grad.length() < config.Δ { break; }
      step_size = step_size * config.decay_factor;
      p = p + grad * config.control_factor;
    }
    p
  }
  fn find_local_max(&mut self, rng: &mut Lcg128Xsl64) -> Option<DistPoint<P, P, WorldSpace>> {
    let config = self.config();

    (0..config.max_attempts).find_map(|_| {
      let p0 = Point2D::new(
        P::from(rng.gen_range::<f64, _>(0.0..1.0)).unwrap(),
        P::from(rng.gen_range::<f64, _>(0.0..1.0)).unwrap()
      );
      let p1 = self.ascend(p0);
      let p1 = DistPoint {
        point: p1,
        distance: self.sample_sdf(p1)
      };
      (p1.distance > config.Δ).then(|| p1)
    })
  }
  // for debugging only
  fn trajectory(&self, mut p: Point2D<P, WorldSpace>) -> Vec<Point2D<P, WorldSpace>> {
    let mut trajectory = vec![p];
    let config = self.config();
    let mut step_size = config.initial_step_size;
    // decay -> limit:
    // fast: 0.5 -> 20
    // slow: 0.85 -> 40
    // veryslow: 0.95 -> 128
    for _ in 0..config.step_limit.unwrap_or(u64::MAX) {
      let grad = self.Δf(p) * step_size;
      if grad.length() < config.Δ { break; }
      step_size = step_size * config.decay_factor;
      p = p + grad * config.control_factor;
      trajectory.push(p)
    }
    trajectory
  }
}

pub struct GradientDescent<T, P> {
  dist_field: T,
  rng_state: Lcg128Xsl64,
  line_config: LineSearchConfig<P>
}

impl <T, P> GradientDescent<Vec<T>, P> {
  pub fn new(line_config: LineSearchConfig<P>) -> Self {
    use rand::prelude::*;
    Self {
      dist_field: vec![],
      rng_state: Lcg128Xsl64::seed_from_u64(0),
      line_config,
    }}

  pub fn insert_sdf(&mut self, sdf: T) {
    self.dist_field.push(sdf);
  }
}

impl <P> LineSearch<P> for GradientDescent<Vec<Box<dyn Fn(Point2D<P, WorldSpace>) -> P + Send + Sync>>, P>
  where P: Float + Signed + Send + Sync {
  fn config(&self) -> LineSearchConfig<P> { self.line_config }
  fn sample_sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    use rayon::prelude::*;

    self.dist_field.par_iter()
      .map(move |s| s(pixel))
      .reduce(
        || P::max_value() / (P::one() + P::one()),
        |a, b| a.min(b)
      )
  }}

impl <P> GradientDescent<ZOrderStorage<Vec<P>>, P>
  where P: Float + Send + Sync {
  pub fn new(line_config: LineSearchConfig<P>, resolution: u64, chunk_size: u64) -> Result<Self> {
    use rand::prelude::*;
    let storage = ZOrderStorage::new(
      resolution, chunk_size, P::max_value() / (P::one() + P::one())
    )?;
    Ok(Self {
      dist_field: storage,
      rng_state: Lcg128Xsl64::seed_from_u64(0),
      line_config,
    })
  }

  pub fn insert_sdf(&mut self, sdf: impl Fn(Point2D<P, WorldSpace>) -> P + Sync + Send) {
    self.insert_sdf_domain(
      Rect::new(
        Point2D::splat(P::zero()),
        Size2D::splat(P::one()),
      ),
      sdf
    );
  }

  pub fn insert_sdf_domain(&mut self, domain: Rect<P, WorldSpace>, sdf: impl Fn(Point2D<P, WorldSpace>) -> P + Sync + Send) {
    use rayon::prelude::*;

    self.dist_field.chunks_domain_par_iter(domain)
      .for_each(move |chunk_xy| {
        let chunk = self.dist_field.get_chunk_xy(chunk_xy);
        chunk.pixels_mut().for_each(|(xy_normalized, value)| {
          *value = (*value).min(sdf(xy_normalized));
        })
      });
  }
}

impl <P> LineSearch<P> for GradientDescent<ZOrderStorage<Vec<P>>, P>
  where P: Float + Signed
{
  fn config(&self) -> LineSearchConfig<P> { self.line_config }

  fn sample_sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    // check whether pixel is out of bounds
    match Box2D::from_size(Size2D::new(P::one(), P::one()))
      .contains(pixel) {
      true => {
        let pixel = (pixel.to_f64() * self.dist_field.resolution as f64)
          .cast_unit()
          .cast::<u64>();
        self.dist_field.pixel(pixel)
      },
      false => sdf::boundary_rect(pixel)
    }
  }
}

#[cfg(test)] mod tests {
  use super::*;
  use crate::geometry::{self, Circle, Shape};
  use crate::drawing::Draw;
  use crate::sdf::{self, SDF};
  use image::{Luma, Pixel, Rgba};
  use crate::solver::Argmax2D;

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

  #[test] #[ignore] fn z_order_backend() -> Result<()> {
    use rand::prelude::*;

    let mut grad = GradientDescent::<ZOrderStorage<_>, f32>
      ::new(LineSearchConfig::default(), 4096, 32)?;
    let mut image = image::RgbaImage::new(4096, 4096);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let mut circles = vec![];
    grad.insert_sdf(sdf::boundary_rect);

    let min_dist = 0.5 * std::f32::consts::SQRT_2 / image.width() as f32;
    let t0 = std::time::Instant::now();
    for _ in 0..16000 {
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
    let size_bound = 1.0 / 256.0;
    let t0 = std::time::Instant::now();
    let mut i = 0;
    while i < 4000 {
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
}