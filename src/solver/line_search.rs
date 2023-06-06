#![allow(dead_code)]
#![allow(non_snake_case)]

use {
  crate::{
    geometry::{P2, WorldSpace},
  },
  euclid::{Vector2D as V2},
  num_traits::Float,
};

#[derive(Copy, Clone)]
pub struct LineSearch<P> {
  pub Δ: P,
  pub initial_step_size: P,
  pub decay_factor: P,
  pub step_limit: Option<u64>,
}

impl <P: Float> Default for LineSearch<P> {
  fn default() -> Self {
    Self {
      Δ: P::from(1e-6).unwrap(),
      initial_step_size: P::one(),
      decay_factor: P::from(0.85).unwrap(),
      step_limit: None,
    }}}

impl<P: Float> LineSearch<P> {
  pub fn grad(&self, f: impl Fn(P2<P>) -> P, p: P2<P>) -> V2<P, WorldSpace> {
    let fp = f(p);
    V2::new(
      f(p + V2::new(self.Δ, P::zero())) - fp,
      f(p + V2::new(P::zero(), self.Δ)) - fp,
    ) / self.Δ
  }

  pub fn optimize(&self, f: impl Fn(P2<P>) -> P, mut p: P2<P>) -> P2<P> {
    let mut step_size = self.initial_step_size;
    for _ in 0..self.step_limit.unwrap_or(u64::MAX) {
      let grad = self.grad(&f, p) * step_size;
      if grad.length() < self.Δ { break; }
      step_size = step_size * self.decay_factor;
      p += grad
    }
    p
  }

  pub fn optimize_normal(&self, f: impl Fn(P2<P>) -> P, mut p: P2<P>) -> bool {
    let mut step_size = self.initial_step_size;
    loop {
      if step_size < self.Δ { break; }

      let fp = f(p);
      if fp > P::zero() { return true }

      let grad = V2::new(
        f(p + V2::new(self.Δ, P::zero())) - fp,
        f(p + V2::new(P::zero(), self.Δ)) - fp,
      ).normalize() * step_size;

      step_size = step_size * self.decay_factor;
      p += grad;
    }
    false
  }

  // for debugging only
  fn trajectory(&self, grad: impl Fn(P2<P>) -> V2<P, WorldSpace>, mut p: P2<P>) -> Vec<P2<P>> {
    let mut trajectory = vec![p];
    let mut step_size = self.initial_step_size;
    // decay -> limit:
    // fast: 0.5 -> 20
    // slow: 0.85 -> 40
    // veryslow: 0.95 -> 128
    for _ in 0..self.step_limit.unwrap_or(u64::MAX) {
      let grad = grad(p) * step_size;
      if grad.length() < self.Δ { break; }
      step_size = step_size * self.decay_factor;
      p += grad;
      trajectory.push(p);
    }
    trajectory
  }
}