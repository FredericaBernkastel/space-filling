//! Implements an adaptive gradient-ascent optimizer.
//!
//! How the ascent *samples* a direction is a compile-time choice, because the
//! two options have different cost models rather than different constants:
//!
//! | probe | field evaluations per step | usable at |
//! |---|---|---|
//! | [`Axial`] | `D` forward differences | any `D`, exact |
//! | [`Gaussian`] | `m` probes, `m` chosen | `m ≪ D` |
//!
//! [`Axial`] is the historical estimator and stays the default: spell it
//! [`LineSearch`], which is an alias. [`Gaussian`] is the randomized estimate of
//! §5 of `doc/publications/infinite_dimensions`,
//!
//! ```text
//! ∇̂g(p) = (1/m) Σᵢ [g(p + σξᵢ) − g(p)]/σ · ξᵢ,    ξᵢ ~ N(0, C)
//! ```
//!
//! whose cost does not mention `D` at all — which is the whole point, since `D`
//! forward differences are meaningless when `D` is infinite and merely expensive
//! when it is a hundred.
//!
//! Honesty requires the caveat the publication attaches: **derivative-free rates
//! are not dimension-free.** Gaussian smoothing pays an explicit factor of the
//! ambient dimension, so `m ≪ D` is only viable when the field genuinely varies
//! in few directions. On a weighted domain it does, and `C` is where the weights
//! go — see [`Gaussian::weighted`]. Without weights this degrades toward random
//! search.

#![allow(non_snake_case)]

use {
  crate::geometry::{Point, Vector, Real, VectorExt},
  num_traits::Float,
};

/// How the ascent samples a direction to climb.
pub trait Probe<P: Real> {
  /// Short name, for diagnostics.
  const NAME: &'static str;
  /// Field evaluations one direction costs in `D` dimensions, excluding the
  /// value at the centre, which the caller already holds.
  fn cost(&self, dims: usize) -> usize;
  /// An unnormalised ascent direction at `p`, given `fp = f(p)`. A zero vector
  /// means flat — the caller stops.
  ///
  /// `step` is the iteration index, so that a randomized probe can vary its
  /// directions without holding mutable state: the ascent stays `Sync` and a run
  /// stays reproducible.
  fn direction<F, const D: usize>(
    &self,
    f: F,
    p: Point<P, D>,
    fp: P,
    delta: P,
    step: u64,
  ) -> Vector<P, D>
  where
    F: Fn(Point<P, D>) -> P;
}

/// One forward difference per axis: `D` evaluations, exact to `O(Δ)`.
#[derive(Copy, Clone, Debug, Default)]
pub struct Axial;

impl<P: Real> Probe<P> for Axial {
  const NAME: &'static str = "axial";

  fn cost(&self, dims: usize) -> usize {
    dims
  }

  #[inline]
  fn direction<F, const D: usize>(
    &self,
    f: F,
    p: Point<P, D>,
    fp: P,
    delta: P,
    _step: u64,
  ) -> Vector<P, D>
  where
    F: Fn(Point<P, D>) -> P,
  {
    // Undivided by `delta` — the caller normalises, and dividing would only
    // scale every component alike.
    Vector::from(std::array::from_fn(|a| {
      let mut q = p;
      q[a] = q[a] + delta;
      f(q) - fp
    }))
  }
}

/// `m` Gaussian probes: `m` evaluations whatever `D` is.
///
/// `scale` is the square root of the covariance `C`, per axis — the coordinate
/// weights of §2.3, so that probes explore the axes that matter in proportion to
/// how much they matter. Isotropic by default, which is the honest fallback and
/// also the setting in which this estimator degrades to random search.
#[derive(Copy, Clone, Debug)]
pub struct Gaussian<P, const D: usize> {
  /// Directions per step. The cost model, and the one number worth tuning.
  pub probes: usize,
  /// Smoothing radius `σ`. The estimate is of `∇g_σ`, not `∇g`, so this trades
  /// bias against the difference quotient's conditioning.
  pub sigma: P,
  /// Per-axis standard deviation — `√C`.
  pub scale: Vector<P, D>,
  /// Fixed, so a run reproduces exactly.
  pub seed: u64,
}

impl<P: Real, const D: usize> Default for Gaussian<P, D> {
  fn default() -> Self {
    Self::isotropic(8, P::from(1e-4).unwrap())
  }
}

impl<P: Real, const D: usize> Gaussian<P, D> {
  /// `C = I`: every direction equally likely.
  pub fn isotropic(probes: usize, sigma: P) -> Self {
    Self { probes, sigma, scale: Vector::repeat(P::one()), seed: 0 }
  }

  /// `√C = scale`, per axis. Pass a
  /// [`Manifold`](crate::adf::Manifold)'s weights to make the probes follow the
  /// manifold: this is the difference between tracking the effective dimension
  /// and searching randomly in the ambient one.
  pub fn weighted(probes: usize, sigma: P, scale: Vector<P, D>) -> Self {
    Self { probes, sigma, scale, seed: 0 }
  }

  /// A different, still reproducible, stream of directions.
  pub fn with_seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }
}

impl<P: Real, const D: usize> Probe<P> for Gaussian<P, D> {
  const NAME: &'static str = "gaussian";

  fn cost(&self, _dims: usize) -> usize {
    self.probes
  }

  fn direction<F, const E: usize>(
    &self,
    f: F,
    p: Point<P, E>,
    fp: P,
    _delta: P,
    step: u64,
  ) -> Vector<P, E>
  where
    F: Fn(Point<P, E>) -> P,
  {
    let mut rng = SplitMix64(self.seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let m: P = P::from(self.probes.max(1) as f64).unwrap();
    let mut acc = Vector::<P, E>::zeros();
    for _ in 0..self.probes.max(1) {
      // `scale` is indexed modulo `E` so that a probe built for one dimension
      // count cannot panic if used at another; in practice `E == D`.
      let xi = Vector::<P, E>::from_fn(|a, _| {
        rng.normal::<P>() * self.scale[a % D]
      });
      let fq = f(p + xi * self.sigma);
      acc += xi * ((fq - fp) / self.sigma);
    }
    acc / m
  }
}

/// SplitMix64 — a counter-based generator, so probe directions are a pure
/// function of `(seed, step)`.
///
/// Inlined rather than taken from `rand`: this is a library, and a hard
/// dependency on an RNG to draw a few normals is a poor trade for its users.
struct SplitMix64(u64);

impl SplitMix64 {
  fn next_u64(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }

  /// Uniform on `(0, 1]` — never zero, so the log below is finite.
  fn unit<P: Real>(&mut self) -> P {
    let bits = (self.next_u64() >> 11) + 1;
    P::from(bits as f64 * (1.0 / 9_007_199_254_740_992.0)).unwrap()
  }

  /// Standard normal, by Box–Muller.
  fn normal<P: Real>(&mut self) -> P {
    let two = P::one() + P::one();
    let tau = P::from(std::f64::consts::TAU).unwrap();
    let (u1, u2) = (self.unit::<P>(), self.unit::<P>());
    (-two * u1.ln()).sqrt() * (tau * u2).cos()
  }
}

/// Adaptive gradient ascent, over a choice of [`Probe`].
///
/// Spell it [`LineSearch`] for the historical axial estimator, or
/// [`RandomSearch`] for `m` Gaussian probes.
#[derive(Copy, Clone)]
pub struct Search<P, G> {
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
  /// How a direction is sampled — the compile-time choice.
  pub probe: G,
}

/// [`Search`] sampling by [`Axial`] forward differences: `D` evaluations per
/// step, and the historical behaviour, bit for bit.
pub type LineSearch<P> = Search<P, Axial>;

/// [`Search`] sampling by `m` [`Gaussian`] probes: cost independent of `D`.
pub type RandomSearch<P, const D: usize> = Search<P, Gaussian<P, D>>;

impl<P: Float, G: Default> Default for Search<P, G> {
  fn default() -> Self {
    Self {
      Δ: P::from(1e-9).unwrap(),
      initial_step_size: P::one(),
      decay_factor: P::from(0.35).unwrap(),
      growth_factor: P::from(1.25).unwrap(),
      step_limit: None,
      probe: G::default(),
    }}}

impl<P: Real, G: Probe<P>> Search<P, G> {
  /// Sample gradient of `f` at `p`, through the configured probe.
  pub fn grad<const D: usize>(
    &self,
    f: impl Fn(Point<P, D>) -> P,
    p: Point<P, D>,
  ) -> Vector<P, D> {
    let fp = f(p);
    self.probe.direction(&f, p, fp, self.Δ, 0) / self.Δ
  }

  /// Field evaluations one iteration costs, excluding the accept/reject probe.
  pub fn probe_cost(&self, dims: usize) -> usize {
    self.probe.cost(dims)
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
  pub fn optimize<const D: usize>(
    &self,
    f: impl Fn(Point<P, D>) -> P,
    mut p: Point<P, D>,
  ) -> Point<P, D> {
    let mut h = self.initial_step_size;
    let mut fp = f(p);
    // direction of the last accepted move; blending it into the next candidate
    // direction cancels the across-ridge zigzag at kink maxima (the two
    // witnesses' gradients alternate), leaving travel along the ridge
    let mut momentum = Vector::<P, D>::zeros();
    // The sampled direction, held while the iterate stands still. Shrinking `h`
    // after a rejection is a bisection *along a direction*, so the direction has
    // to survive the rejection: resampling one per iteration turns the search
    // into a random walk that never commits. It is also free accuracy for
    // [`Axial`], whose estimate at an unmoved `p` is identical anyway — it was
    // recomputing `D` forward differences after every rejected step for nothing.
    let mut sampled: Option<Vector<P, D>> = None;
    for step in 0..self.step_limit.unwrap_or(u64::MAX) {
      if h < self.Δ { break; }
      let g = match sampled {
        Some(g) => g,
        None => *sampled.insert(self.probe.direction(&f, p, fp, self.Δ, step)),
      };
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
        sampled = None;   // the iterate moved; the estimate is stale
      } else {
        momentum = Vector::zeros();
        h = h * self.decay_factor;
      }
    }
    p
  }

  // for debugging only
  #[allow(dead_code)]
  fn trajectory<const D: usize>(
    &self,
    grad: impl Fn(Point<P, D>) -> Vector<P, D>,
    mut p: Point<P, D>,
  ) -> Vec<Point<P, D>> {
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
  use {super::*, crate::geometry::{P2, V2}, std::cell::Cell};

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
    let apex = V2::new(0.5, 0.66);

    let p = LineSearch::default().optimize(&f, P2::new(0.41, 0.57));
    let err = (p.coords - apex).length();
    println!("optimize_precision: err = {err:.3e}, f = {:.9}, evals = {}", f(p), evals.get());
    // previous fixed-schedule optimizer: err = 6.6e-7 at 259 evaluations
    assert!(err < 1e-8, "kink maximum located to {err:.3e} only");
    assert!(evals.get() < 400, "{} field evaluations", evals.get());
  }

  /// The randomized probe climbs the same hill, and does it without ever
  /// looking along an axis.
  #[test] fn gaussian_probe_finds_the_same_maximum() {
    let f = |p: P2<f64>| {
      let a = P2::new(0.2, 0.5);
      let b = P2::new(0.8, 0.5);
      let bnd = p.x.min(p.y).min(1.0 - p.x).min(1.0 - p.y);
      (p - a).length().min((p - b).length()).min(bnd)
    };
    let search: RandomSearch<f64, 2> = Search {
      probe: Gaussian::isotropic(16, 1e-6),
      ..Default::default()
    };
    let p = search.optimize(&f, P2::new(0.41, 0.57));
    // a randomized estimate of a kink maximum is not an exact one: `∇̂` is of
    // the σ-smoothed field, and the ridge is where the smoothing bites hardest
    let err = (p.coords - V2::new(0.5, 0.66)).length();
    assert!(err < 1e-3, "gaussian ascent reached only {err:.3e}");
    assert!(f(p) > 0.339, "gaussian ascent settled below the apex: {}", f(p));
  }

  /// Same seed, same answer — the probe holds no mutable state, so an ascent is
  /// reproducible and the optimizer stays `Sync`.
  #[test] fn gaussian_probes_are_reproducible() {
    let f = |p: P2<f64>| -((p.coords - V2::new(0.3, 0.7)).length());
    let mk = |seed: u64| -> RandomSearch<f64, 2> {
      Search { probe: Gaussian::isotropic(6, 1e-5).with_seed(seed), ..Default::default() }
    };
    let a = mk(7).optimize(&f, P2::new(0.1, 0.1));
    let b = mk(7).optimize(&f, P2::new(0.1, 0.1));
    assert_eq!(a.coords.as_slice(), b.coords.as_slice(), "same seed, different path");

    let c = mk(8).optimize(&f, P2::new(0.1, 0.1));
    assert!(a.coords != c.coords || (a.coords - V2::new(0.3, 0.7)).length() < 1e-9,
      "different seeds gave an identical path without converging");
  }

  /// The cost model, which is the reason the probe exists: `D` against `m`.
  #[test] fn cost_is_dimension_free() {
    let axial = LineSearch::<f64>::default();
    assert_eq!(axial.probe_cost(3), 3);
    assert_eq!(axial.probe_cost(100), 100);

    let random: RandomSearch<f64, 100> = Search {
      probe: Gaussian::isotropic(8, 1e-4), ..Default::default()
    };
    assert_eq!(random.probe_cost(3), 8);
    assert_eq!(random.probe_cost(100), 8);
  }
}
