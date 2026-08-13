//! Axial forward differences against `m` Gaussian probes.
//!
//! Roadmap step 4 of `doc/publications/infinite_dimensions`: replace `D` finite
//! differences with `m` Gaussian directions, at a cost that never mentions `D`.
//!
//! The publication attaches a caveat, and this benchmark exists to test it
//! rather than to repeat it: derivative-free rates are **not** dimension-free, so
//! the estimator is only viable when the field varies in few directions. The
//! field here has exactly that structure — a distance measured in a weighted
//! norm, active on `k` axes and nearly flat on the rest — and the three arms are
//! the axial estimator, isotropic probes that do not know where the action is,
//! and weighted probes that do.
//!
//! ```text
//! cargo test -p adaptive-distance-field --release --test probe -- --ignored --nocapture
//! ```

use {
  adaptive_distance_field::{
    geometry::{Point, Vector, VectorExt},
    line_search::{Axial, Gaussian, LineSearch, Search},
  },
  std::cell::Cell,
};

const ACTIVE: usize = 3;
const TAIL: f64 = 0.01;
/// Field evaluations every arm is allowed, so the comparison is efficiency and
/// not budget: a step costs `probe + 1`, so each arm gets its own step limit.
const BUDGET: u64 = 20_000;
/// The accuracy a space-filling step actually needs: enough to place a shape,
/// not enough to name the apex to nine digits.
const TOL: f64 = 1e-2;

/// Weights: `1` on the first [`ACTIVE`] axes, [`TAIL`] on the rest.
fn weights<const D: usize>() -> Vector<f64, D> {
  Vector::from_fn(|i, _| if i < ACTIVE { 1.0 } else { TAIL })
}

/// `−‖(p − a) ⊙ w‖`: a cone apex at `a`, steep on the active axes and nearly
/// flat elsewhere. A kink maximum, as a distance field's maxima are.
fn run<const D: usize, G>(mut search: Search<f64, G>) -> (u64, u64, f64)
where
  G: adaptive_distance_field::line_search::Probe<f64>,
{
  // one probe set plus one accept/reject evaluation per step
  search.step_limit = Some(BUDGET / (search.probe_cost(D) as u64 + 1));
  let w = weights::<D>();
  let a = Point::from(Vector::<f64, D>::repeat(0.5));
  let evals = Cell::new(0u64);
  let hit = Cell::new(u64::MAX);
  let f = |p: Point<f64, D>| {
    evals.set(evals.get() + 1);
    let d = (p - a).component_mul(&w).length();
    // evaluations until the search first *visits* a point inside `TOL` —
    // estimator-agnostic, and the only accuracy a placement needs
    if d < TOL && hit.get() == u64::MAX {
      hit.set(evals.get());
    }
    -d
  };
  // Offset only within the active subspace. Offsetting the tail too would put a
  // `√(D−k)·tail·offset` floor under the error — 2.9e-2 at D = 100 — which the
  // weighted probe cannot cross *by design*, since ignoring the tail is the
  // whole point of weighting it down. That floor measures the benchmark, not
  // the estimator.
  let start = Point::from(Vector::<f64, D>::from_fn(|i, _| {
    if i < ACTIVE { 0.9 } else { 0.5 }
  }));
  let p = search.optimize(&f, start);
  // error measured in the same weighted norm the field uses
  (hit.get(), evals.get(), (p - a).component_mul(&w).length())
}

fn show(hit: u64) -> String {
  if hit == u64::MAX { "—".into() } else { hit.to_string() }
}

macro_rules! band {
  ($D:literal) => {{
    const D: usize = $D;
    let axial: LineSearch<f64> = Search::default();
    let (ax_hit, ev, ax_err) = run::<D, Axial>(axial);
    println!("  {:>4} {:>12} {:>5} {:>9} {:>9} {:>11.2e}",
      D, "axial", D, show(ax_hit), ev, ax_err);

    let mut best = f64::MAX;
    for m in [4usize, 8, 16] {
      let s: Search<f64, Gaussian<f64, D>> = Search {
        probe: Gaussian::isotropic(m, 1e-6), ..Default::default() };
      let (hit, ev, err) = run::<D, Gaussian<f64, D>>(s);
      println!("  {:>4} {:>12} {:>5} {:>9} {:>9} {:>11.2e}",
        D, "isotropic", m, show(hit), ev, err);
      best = best.min(err);
    }

    let s: Search<f64, Gaussian<f64, D>> = Search {
      probe: Gaussian::weighted(8, 1e-6, weights::<D>()), ..Default::default() };
    let (w_hit, ev, w_err) = run::<D, Gaussian<f64, D>>(s);
    println!("  {:>4} {:>12} {:>5} {:>9} {:>9} {:>11.2e}",
      D, "weighted", 8, show(w_hit), ev, w_err);
    println!("  {}", "-".repeat(58));
    (ax_hit, w_hit)
  }};
}

#[test] #[ignore] fn probes_against_finite_differences() {
  println!("\n{ACTIVE} active axes of D, tail weight {TAIL}, {BUDGET} evaluations each");
  println!("the apex is a kink, as a distance field's maxima are\n");
  println!("  {:>4} {:>12} {:>5} {:>9} {:>9} {:>11}",
    "D", "probe", "m", "to 1e-2", "evals", "final err");
  println!("  {}", "-".repeat(58));
  let rows = [band!(2), band!(8), band!(24), band!(100)];

  // In the plane the axial estimator is exact and cheap, and should win. By
  // D = 100 it spends a hundred evaluations to learn what eight probes aimed at
  // the active subspace already knew, and should lose at the same budget.
  let (ax2, w2) = rows[0];
  assert!(ax2 <= w2, "axial should reach tolerance first at D = 2: {ax2} against {w2}");
  let (ax100, w100) = rows[3];
  assert!(w100 < ax100,
    "weighted probes should reach tolerance first at D = 100: {w100} against {ax100}");
}
