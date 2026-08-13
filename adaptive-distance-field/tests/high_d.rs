//! Anisotropic bodies against balls, on compact manifolds, up to `D = 100`.
//!
//! Roadmap step 3 of `doc/publications/infinite_dimensions`: per-axis radii,
//! because a ball is the wrong body at high `N`. A ball is limited by the
//! *thinnest* free direction and wastes every other one, and on a manifold whose
//! extents decay as `γᵢ = (i+1)^(−s)` that is nearly all of them.
//!
//! **Volume, in logs.** §7.3 warns that volume intuition fails above a handful of
//! dimensions, and it applies to the scoring here: counting bodies placed says
//! nothing about how much was claimed. So the metric is `Σ ln vol` against
//! `ln vol(domain)`, and it has to be logs — the unit ball's volume at `D = 100`
//! is about `10^(−40)` and a Sobolev domain's is `10^(−316)`. One more axis and
//! `f64` underflows to zero, taking the ratio with it.
//!
//! ```text
//! cargo test -p adaptive-distance-field --release --test high_d -- --ignored --nocapture
//! ```

use {
  adaptive_distance_field::{
    adf::{Manifold, Primitive, Reach, WeightedKd, ADF},
    geometry::{Point, Vector, VectorExt},
    sdf::SDF,
  },
  rand::prelude::*,
  std::time::Instant,
};

const BODIES: usize = 60;
const LEVELS: u16 = 24;
const PRUNE_LEVELS: u32 = 4;
const GROW_STEPS: u32 = 8;
const FREE_LEVELS: u32 = 12;

/// An axis-aligned box as a bare field: the exact SDF, hence 1-Lipschitz.
fn abox<const D: usize>(
  centre: Point<f64, D>,
  half: Vector<f64, D>,
) -> impl Fn(Point<f64, D>) -> f64 + Send + Sync + 'static {
  move |p| {
    let q = (p - centre).abs() - half;
    let outside = q.map(|x| x.max(0.0)).length();
    let inside = q.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)).min(0.0);
    outside + inside
  }
}

struct Run {
  placed: usize,
  /// `ln` of the total volume claimed, accumulated by log-sum-exp so that no
  /// single body's volume ever has to be representable.
  log_claimed: f64,
  nodes: usize,
  bytes: usize,
  millis: f64,
}

/// `ln(e^a + e^b)` without leaving log space.
fn log_add(a: f64, b: f64) -> f64 {
  if a == f64::NEG_INFINITY { return b }
  if b == f64::NEG_INFINITY { return a }
  let (hi, lo) = if a > b { (a, b) } else { (b, a) };
  hi + (lo - hi).exp().ln_1p()
}

/// `boxes = false` places the inscribed ball the solver hands over today;
/// `true` places the largest aspect-locked box the field will certify.
fn run<const D: usize>(m: &Manifold<f64, D>, boxes: bool) -> Run {
  run_inner(m, boxes, FREE_LEVELS)
}

/// The box arm at an explicit certification budget.
fn run_at<const D: usize>(m: &Manifold<f64, D>, levels: u32) -> Run {
  run_inner(m, true, levels)
}

fn run_inner<const D: usize>(m: &Manifold<f64, D>, boxes: bool, free_levels: u32) -> Run {
  let mut field: ADF<f64, D, WeightedKd> = m.field(1);
  field = field.with_levels(LEVELS).with_prune_levels(PRUNE_LEVELS);
  let aspect = *m.weights();
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0x1CE_B00);

  let (mut placed, mut log_claimed) = (0usize, f64::NEG_INFINITY);
  let t0 = Instant::now();
  for _ in 0..BODIES {
    // sampled in the manifold's own units, so a candidate is never dominated by
    // the axes it happens to be long in
    let c = Point::from(Vector::<f64, D>::from_fn(|i, _| {
      rng.random_range(-0.35..0.35) * m.weights()[i]
    }));
    let clearance = field.sdf(c);
    if !(clearance > 0.0) {
      continue;
    }

    let (half, reach) = if boxes {
      let bx = field.grow_box(c, aspect, GROW_STEPS, free_levels);
      (bx.size() / 2.0, Reach::Box(bx))
    } else {
      // today's rule: a ball of the clearance, scored as the cube it inscribes
      // so that the two are compared as *bodies* rather than as formulas
      let r = clearance / (D as f64).sqrt();
      (Vector::<f64, D>::repeat(r), Reach::Ball { centre: c, radius: clearance })
    };
    if !(half[0] > 0.0) {
      continue;
    }

    if field.insert_within_reach(reach, Primitive::new(abox(c, half))) {
      placed += 1;
      let log_vol: f64 = half.iter().map(|h| (2.0 * h).ln()).sum();
      log_claimed = log_add(log_claimed, log_vol);
    }
  }
  let millis = t0.elapsed().as_secs_f64() * 1e3;

  Run {
    placed, log_claimed,
    nodes: field.tree.node_count(),
    bytes: field.memory_bytes(),
    millis,
  }
}

fn human(bytes: usize) -> String {
  const UNIT: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
  let (mut v, mut u) = (bytes as f64, 0);
  while v >= 1024.0 && u + 1 < UNIT.len() { v /= 1024.0; u += 1 }
  format!("{v:.1} {}", UNIT[u])
}

macro_rules! band {
  ($D:literal, $s:expr) => {{
    const D: usize = $D;
    let m = Manifold::<f64, D>::sobolev($s);
    let (ball, boxed) = (run::<D>(&m, false), run::<D>(&m, true));
    let dom = m.log_volume();
    // ln of the ratio of claimed volumes — the κ^D gap, in the only units that
    // survive: e^{gap} itself overflows f64 by D = 48
    let gap = boxed.log_claimed - ball.log_claimed;
    println!("  {:>4} {:>4.1} {:>8.2} {:>8} {:>8} {:>11.1} {:>11.1} {:>9.1} {:>8} {:>9}",
      D, $s as f64, m.effective_dimension(),
      ball.placed, boxed.placed,
      ball.log_claimed - dom, boxed.log_claimed - dom,
      gap, boxed.nodes, human(boxed.bytes));
    (gap, ball.millis + boxed.millis)
  }};
}

#[test] #[ignore] fn anisotropic_bodies_at_high_d() {
  println!("\n{BODIES} candidates, {LEVELS} levels, prune {PRUNE_LEVELS} levels, \
            grow {GROW_STEPS}×{FREE_LEVELS}");
  println!("claimed volumes are ln(Σ vol) − ln vol(domain); nothing here is \
            representable outside logs\n");
  println!("  {:>4} {:>4} {:>8} {:>8} {:>8} {:>11} {:>11} {:>9} {:>8} {:>9}",
    "D", "s", "eff dim", "balls", "boxes", "ln ball÷dom", "ln box÷dom",
    "ln gap", "nodes", "memory");
  println!("  {}", "-".repeat(100));

  let mut total = 0.0;
  let mut gaps = vec![];
  for (gap, ms) in [
    band!(24, 1.0), band!(24, 2.0),
    band!(48, 1.0), band!(48, 2.0),
    band!(100, 1.0), band!(100, 2.0),
  ] {
    gaps.push(gap);
    total += ms;
  }
  println!("\n  {:.1} s of measurement", total / 1e3);

  println!("\n  what binds is the certificate, not the geometry — `ln gap` against");
  println!("  the levels `box_is_free` may refine, at D = 24, s = 2\n");
  println!("  {:>8} {:>11}", "levels", "ln gap");
  println!("  {}", "-".repeat(22));
  let m = Manifold::<f64, 24>::sobolev(2.0);
  let ball = run::<24>(&m, false);
  for lv in [0u32, 2, 4, 8, 12] {
    println!("  {:>8} {:>11.1}", lv, run_at::<24>(&m, lv).log_claimed - ball.log_claimed);
  }

  // What is true and worth pinning: every arm places bodies, and the field stays
  // finite and exact. The *sign* of the gap is deliberately not asserted — it is
  // negative wherever the anisotropy is strong, and the sweep above is the
  // evidence for why. See CHANGELOG.md, "A ball is the wrong body, and the
  // certificate does not care".
  assert!(gaps.iter().all(|g| g.is_finite()), "a run claimed nothing at all");
}
