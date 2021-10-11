#![allow(dead_code)]
#![allow(non_snake_case)]
use {
  crate::{
    geometry::WorldSpace,
    solver::DistPoint
  },
  euclid::{Point2D, Vector2D as V2},
  rand_pcg::Lcg128Xsl64,
  num_traits::Float
};
use std::fmt::Debug;

mod impl_gradientdescent_vec_fn;
mod impl_gradientdescent_zorderstorage;
mod impl_gradientdescent_adf;
#[cfg(test)] mod tests;

pub struct GradientDescent<T, P> {
  dist_field: T,
  line_config: LineSearchConfig<P>
}

#[derive(Copy, Clone)]
pub struct LineSearchConfig<P> {
  pub Δ: P,
  pub initial_step_size: P,
  pub decay_factor: P,
  pub step_limit: Option<u64>,
  pub max_attempts: u64,
  pub control_factor: P
}

impl <P: Float> Default for LineSearchConfig<P> {
  fn default() -> Self {
    LineSearchConfig {
      Δ: P::from(1.0 / 1024.0).unwrap(),
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
      self.sample_sdf(p + V2::new(Δp, P::zero())) - self.sample_sdf(p),
      self.sample_sdf(p + V2::new(P::zero(), Δp)) - self.sample_sdf(p),
    ) / Δp
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
    use rand::Rng;
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

pub struct GradientDescentIter<'a, T, P> {
  grad: &'a mut GradientDescent<T, P>,
  rng: Lcg128Xsl64
}

impl <T, P> GradientDescent<T, P>
  where P: Float,
        Self: LineSearch<P> {
  pub fn iter(&mut self) -> GradientDescentIter<T, P> {
    use rand::prelude::*;
    GradientDescentIter {
      grad: self,
      rng: Lcg128Xsl64::seed_from_u64(0)
    }
  }
}

impl <'a, T, P> GradientDescentIter<'a, T, P>
  where P: Float + Send + Debug,
        GradientDescent<T, P>: LineSearch<P> {
  pub fn build(self) -> impl Iterator<Item = (DistPoint<P, P, WorldSpace>, &'a mut GradientDescent<T, P>)> {
    use rayon::prelude::*;
    use rand::prelude::*;

    let grad = self.grad as *const _ as usize;
    (0..)
      .map(move |i| {
        let points: Vec<_> = (0..32).into_par_iter()
          .filter_map(|j| {
            let grad = unsafe { &mut *(grad as *mut GradientDescent<_, _>)};
            grad.find_local_max(&mut Lcg128Xsl64::seed_from_u64(i * 32 + j))
          })
          .collect();
        let mut points1 = vec![];
        points.into_iter()
          .for_each(|pn| {
            points1.iter()
              .all(|p: &DistPoint<_, _, _>| p.point.distance_to(pn.point) / P::from(2.0).unwrap() > pn.distance)
              .then(|| points1.push(pn));
          });

        points1
      })
      .take_while(|ps| ps.len() > 0)
      .flat_map(move |ps| {
        ps.into_iter()
          .map(move |p| (p, unsafe { &mut *(grad as *mut GradientDescent<_, _>) }))
      })
  }
}