//! A certified minimum over time, by Lipschitz branch and bound.
//!
//! `voice::field` samples `t` on a grid and takes the smallest value it sees.
//! That is fine for measuring a slope and useless as a proof: a dissonance
//! narrower than the grid spacing is simply not seen, and the texture is declared
//! legal because nobody looked. Any fixed sampling density can be defeated by a
//! short enough collision.
//!
//! This is the same branch and bound the main crate's `sdf_geq_everywhere`
//! performs over boxes, in one dimension over time. On an interval `[a, b]` with
//! midpoint `m`, Lipschitz continuity of `R` gives
//!
//! ```text
//! min over [a,b] of (θ − R) ≥ (θ − R(m)) − L_t·(b−a)/2
//! ```
//!
//! so an interval whose bound already exceeds the best value seen anywhere cannot
//! contain the minimum and is discarded whole. The intervals that survive are the
//! ones near the worst instants, which is where the refinement goes. The returned
//! number is a **lower bound on the true minimum over the continuum** — never an
//! optimistic one, whatever the budget.

use {
  crate::roughness::Timbre,
  crate::voice::{self, Voice},
};

/// The largest `|dR/dt|` found on a fine sweep — an empirical lower bound on the
/// constant the branch and bound needs.
///
/// Measured rather than derived: it is dominated by the note envelopes, and a
/// derivation would have to bound the roughness curve's slope, the partial
/// spacing and the envelope's together. A measured value can be scaled up for
/// safety, which `SAFETY` does.
pub fn time_slope(
  voices: &[Voice],
  attack: f64,
  timbre: &Timbre,
  window: (f64, f64),
  samples: usize,
) -> f64 {
  let (t0, t1) = window;
  let dt = (t1 - t0) / samples as f64;
  let mut worst: f64 = 0.0;
  for k in 1..samples {
    let t = t0 + dt * k as f64;
    let a = voice::roughness_at(voices, t - dt * 0.5, attack, timbre);
    let b = voice::roughness_at(voices, t + dt * 0.5, attack, timbre);
    worst = worst.max((b - a).abs() / dt);
  }
  worst
}

/// A measured slope is a lower bound on the true constant, so the certificate
/// uses a multiple of it. Sound only to the extent this covers the gap — stated
/// plainly because it is the one assumption the proof rests on.
pub const SAFETY: f64 = 2.0;

/// A certified lower bound on `min over the window of (θ − R(t))`.
///
/// `max_depth` bounds the recursion; deeper is tighter, never less sound. The
/// second return is the smallest value actually *observed*, which brackets the
/// truth from above — so the true minimum lies in `[certified, observed]`.
pub fn certified_min(
  voices: &[Voice],
  theta: f64,
  attack: f64,
  timbre: &Timbre,
  window: (f64, f64),
  l_t: f64,
  max_depth: u32,
) -> (f64, f64, usize) {
  let l = l_t * SAFETY;
  let mut observed = f64::INFINITY;
  let mut certified = f64::INFINITY;
  let mut evaluated = 0usize;
  let mut stack = vec![(window.0, window.1, 0u32)];

  while let Some((a, b, depth)) = stack.pop() {
    let mid = 0.5 * (a + b);
    let v = theta - voice::roughness_at(voices, mid, attack, timbre);
    evaluated += 1;
    if v < observed {
      observed = v;
    }
    let lower = v - l * 0.5 * (b - a);

    // This interval cannot hold anything below what has already been seen, or
    // the budget is spent: either way it contributes its bound and stops.
    if lower >= observed || depth >= max_depth {
      if lower < certified {
        certified = lower;
      }
      continue;
    }
    stack.push((a, mid, depth + 1));
    stack.push((mid, b, depth + 1));
  }
  (certified, observed, evaluated)
}

/// A certified bound over a **box of placements crossed with time**.
///
/// Step 2 proved a single texture legal by bounding over `t`. A packing needs
/// more: whether *every* placement in a range of transpositions and entry
/// offsets is legal, which is a bound over the product space. Lipschitz
/// continuity gives it directly, one term per axis,
///
/// ```text
/// min over the box of (θ − R) ≥ (θ − R(centre)) − Σᵢ Lᵢ·halfᵢ
/// ```
///
/// The axes are wildly unlike each other — cents against seconds, with constants
/// differing by three orders of magnitude — so the cut goes to whichever axis
/// contributes most slack, `argmax Lᵢ·halfᵢ`. That is the main crate's `Widest`
/// policy in the metric the constants define, and for the same reason: halving
/// the axis that dominates the bound is the only cut that buys anything.
///
/// Returns the certified lower bound, the smallest value observed, and the
/// evaluations spent. `lower > 0` proves every placement in the box legal;
/// `observed < 0` exhibits one that is not.
pub fn certify_region(
  r: impl Fn(f64, f64, f64) -> f64,
  cents: (f64, f64),
  onset: (f64, f64),
  time: (f64, f64),
  theta: f64,
  l: [f64; 3],
  max_depth: u32,
) -> (f64, f64, usize) {
  let mut observed = f64::INFINITY;
  let mut certified = f64::INFINITY;
  let mut evaluated = 0usize;
  let mut stack = vec![([cents, onset, time], 0u32)];

  while let Some((b, depth)) = stack.pop() {
    let mid = [
      0.5 * (b[0].0 + b[0].1),
      0.5 * (b[1].0 + b[1].1),
      0.5 * (b[2].0 + b[2].1),
    ];
    let v = theta - r(mid[0], mid[1], mid[2]);
    evaluated += 1;
    if v < observed {
      observed = v;
    }

    let mut slack = 0.0;
    let mut worst_axis = 0;
    let mut worst_slack = -1.0;
    for i in 0..3 {
      let half = 0.5 * (b[i].1 - b[i].0);
      let s = l[i] * half;
      slack += s;
      if s > worst_slack {
        worst_slack = s;
        worst_axis = i;
      }
    }
    let lower = v - slack;

    if lower >= observed || depth >= max_depth {
      if lower < certified {
        certified = lower;
      }
      continue;
    }
    // halve the axis carrying the most slack — `Widest`, in the metric the
    // Lipschitz constants define
    let (a, c) = b[worst_axis];
    let m = 0.5 * (a + c);
    let (mut lo, mut hi) = (b, b);
    lo[worst_axis] = (a, m);
    hi[worst_axis] = (m, c);
    stack.push((lo, depth + 1));
    stack.push((hi, depth + 1));
  }
  (certified, observed, evaluated)
}
