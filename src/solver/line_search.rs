//! Implements an adaptive gradient-ascent optimizer.

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
  /// Probe distance for finite-difference gradients, and the convergence
  /// tolerance: iteration stops once the step length falls below `Δ`.
  pub Δ: P,
  /// Initial — and maximum — step length.
  pub initial_step_size: P,
  /// Step shrink factor on a rejected (non-improving) move.
  pub decay_factor: P,
  /// Step growth factor on an accepted move.
  pub growth_factor: P,
  pub step_limit: Option<u64>,
}

impl <P: Float> Default for LineSearch<P> {
  fn default() -> Self {
    Self {
      Δ: P::from(1e-9).unwrap(),
      initial_step_size: P::one(),
      decay_factor: P::from(0.35).unwrap(),
      growth_factor: P::from(1.25).unwrap(),
      step_limit: None,
    }}}

impl<P: Float> LineSearch<P> {
  /// Sample gradient of `f` at `p`.
  pub fn grad(&self, f: impl Fn(P2<P>) -> P, p: P2<P>) -> V2<P, WorldSpace> {
    let fp = f(p);
    V2::new(
      f(p + V2::new(self.Δ, P::zero())) - fp,
      f(p + V2::new(P::zero(), self.Δ)) - fp,
    ) / self.Δ
  }

  /// Find a local maximum of `f`, using `p` as the initial location.
  ///
  /// Adaptive ascent: a candidate step of length `h` along the sampled
  /// gradient direction is taken only if it improves `f` — the iterate is
  /// monotone and never ends below the best point seen. `h` grows by
  /// `growth_factor` on acceptance (capped at `initial_step_size`) and shrinks
  /// by `decay_factor` on rejection, refining the kink maxima of distance
  /// fields (non-smooth on the medial axis) bisection-style. Terminates once
  /// `h < Δ`, at a vanishing gradient (flat region — e.g. an estimator
  /// clamping its interior to a constant), or after `step_limit` iterations —
  /// unlike a fixed decay schedule, early when converged.
  pub fn optimize(&self, f: impl Fn(P2<P>) -> P, mut p: P2<P>) -> P2<P> {
    let mut h = self.initial_step_size;
    let mut fp = f(p);
    // direction of the last accepted move; blending it into the next candidate
    // direction cancels the across-ridge zigzag at kink maxima (the two
    // witnesses' gradients alternate), leaving travel along the ridge
    let mut momentum = V2::<P, WorldSpace>::zero();
    for _ in 0..self.step_limit.unwrap_or(u64::MAX) {
      if h < self.Δ { break; }
      let g = V2::<P, WorldSpace>::new(
        f(p + V2::new(self.Δ, P::zero())) - fp,
        f(p + V2::new(P::zero(), self.Δ)) - fp,
      );
      let len = g.length();
      if !(len > P::zero()) { break; } // flat (or non-finite) — nothing to climb
      let dir = (g / len + momentum).robust_normalize();
      let candidate = p + dir * h;
      let fc = f(candidate);
      if fc > fp {
        p = candidate;
        fp = fc;
        momentum = dir;
        h = (h * self.growth_factor).min(self.initial_step_size);
      } else {
        momentum = V2::zero();
        h = h * self.decay_factor;
      }
    }
    p
  }

  // for debugging only
  #[allow(dead_code)]
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

#[cfg(test)]
mod tests {
  use {super::*, std::cell::Cell};

  // Two point obstacles at (0.2, 0.5) and (0.8, 0.5) inside the unit square:
  // on the bisector x = 0.5 the field min(|p−a|, |p−b|, boundary) peaks where
  // the point distance meets the top-edge distance — the exact (kink) maximum
  // is (0.5, 0.66) with value 0.34, witnessed by both points and the top edge.
  #[test] fn optimize_precision() {
    let evals = Cell::new(0u64);
    let f = |p: P2<f64>| {
      evals.set(evals.get() + 1);
      let a = P2::new(0.2, 0.5);
      let b = P2::new(0.8, 0.5);
      let bnd = p.x.min(p.y).min(1.0 - p.x).min(1.0 - p.y);
      (p - a).length().min((p - b).length()).min(bnd)
    };
    let apex = P2::new(0.5, 0.66);

    let p = LineSearch::default().optimize(&f, P2::new(0.41, 0.57));
    let err = (p - apex).length();
    println!("optimize_precision: err = {err:.3e}, f = {:.9}, evals = {}", f(p), evals.get());
    // previous fixed-schedule optimizer: err = 6.6e-7 at 259 evaluations
    assert!(err < 1e-8, "kink maximum located to {err:.3e} only");
    assert!(evals.get() < 400, "{} field evaluations", evals.get());
  }
}