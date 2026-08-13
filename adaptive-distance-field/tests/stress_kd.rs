//! How far a k-d backed field gets in ten seconds, from one dimension to twenty,
//! with and without the dimension-dependent split policy.
//!
//! The loop is the whole GD-ADF pipeline: climb to a local maximum of the free
//! space, place a ball there of *diameter* half the clearance — a radius of one
//! quarter — and repeat until the budget runs out. Nothing is capped but time and
//! memory, so each row reports what one dimension's worth of geometry costs.
//!
//! Each dimension runs twice, both arms getting the full budget and the same seed:
//!
//! - **divide** — `with_cut_must_prune(false)`: an overflowing leaf always divides.
//!   The library's default below [`CUT_MUST_PRUNE_MIN_DIMS`].
//! - **bite** — `with_cut_must_prune(true)`: it divides only if the division prunes
//!   something. The default at and above that dimension.
//!
//! The pair is the point. A single arm cannot show that the right answer is
//! dimensional, and the threshold is only defensible if each arm wins on its own
//! side of it.
//!
//! [`Orthant`](adaptive_distance_field::adf::Orthant) cannot be instantiated above
//! six dimensions at all, and would want a million nodes per subdivision at
//! twenty, so this is a k-d-only test by necessity rather than by choice.
//!
//! Two settings are not the library defaults, and both are forced by dimension:
//!
//! - `prune_subdiv = 1`, against a default of 8. The redundancy proof clears a box
//!   when the centre margin covers `(L_f + L_g)·h(R)`, and `h` is the *half
//!   diagonal* — `√D/2` for the unit cube, so `√D` times the half-side. Every cell
//!   is therefore `√D` times harder to certify than in one dimension, the early
//!   exit fires far less often, and the branch-and-bound descends much deeper. At
//!   the default budget a *single insertion* measured 6.1 s at `D = 8` and 4.4 s at
//!   `D = 10`; at 1 it is 3 ms at `D = 20`.
//! - A memory ceiling, because without the split policy the same weakness has a
//!   structural consequence: when no cut prunes anything, every leaf keeps every
//!   primitive and every leaf overflows again, so the tree *doubles on each
//!   insertion*. Left alone at `D = 20` that is `2^120` leaves.
//!
//! ```text
//! cargo test -p adaptive-distance-field --release --test stress_kd -- --ignored --nocapture
//! ```
//!
//! Roughly seven minutes, by construction.

use {
  adaptive_distance_field::{
    adf::{tree::Kd, Primitive, ADF, CUT_MUST_PRUNE_MIN_DIMS},
    geometry::{Point, Vector, VectorExt},
    line_search::LineSearch,
    sdf::{self, SDF},
  },
  rand::prelude::*,
  std::time::{Duration, Instant},
};

/// Wall-clock per dimension, per arm.
const BUDGET: Duration = Duration::from_secs(10);
/// Full subdivisions — halvings of every axis. A k-d tree stores `SPLITS · D`
/// levels for it.
const SPLITS: u8 = 6;
/// Proof refinement budget. See the module note: 8 is a two-dimensional default.
const PRUNE_SUBDIV: u32 = 1;
/// Stop an arm rather than let the arena run away.
const MEM_CEILING: usize = 256 << 20;
/// Checked every this many insertions, since it costs a traverse.
const MEM_EVERY: usize = 64;
/// Bounded so one pathological ascent cannot eat the budget.
const ASCENT_STEPS: u64 = 150;

struct Row {
  d: usize,
  bite: bool,
  inserted: usize,
  first: f64,
  last: f64,
  levels: u16,
  nodes: usize,
  leaves: usize,
  slots: usize,
  bytes: usize,
  elapsed: Duration,
  stopped: &'static str,
  debug: String,
}

impl Row {
  fn ins_per_s(&self) -> f64 {
    self.inserted as f64 / self.elapsed.as_secs_f64()
  }
  /// Mean primitives per leaf; internal nodes hold a scalar and contribute none.
  fn occupancy(&self) -> f64 {
    self.slots as f64 / self.leaves.max(1) as f64
  }
}

macro_rules! stress {
  ($D:literal) => {{
    const D: usize = $D;

    let run = |bite: bool| -> Row {
      let mut field = ADF::<f64, D, Kd>::new(SPLITS, vec![Primitive::new(sdf::boundary_rect)])
        .with_prune_subdiv(PRUNE_SUBDIV)
        .with_cut_must_prune(bite);
      let ls = LineSearch { step_limit: Some(ASCENT_STEPS), ..LineSearch::default() };
      // Identical across arms, so the two see the same sequence of ascents.
      let mut rng = rand_pcg::Pcg64::seed_from_u64(0x57_2E_55 + D as u64);

      let mut inserted = 0usize;
      let (mut first, mut last) = (f64::NAN, f64::NAN);
      let mut stopped = "time";
      let t0 = Instant::now();
      while t0.elapsed() < BUDGET {
        let start = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)));
        let peak = ls.optimize(|p| field.sdf(p), start);
        let clearance = field.sdf(peak);
        if !(clearance > 1e-12) {
          continue;
        }
        // diameter = clearance / 2, hence a radius of one quarter of the free ball
        // — well inside it, so `insert_within` may be given the ball's true reach
        let radius = clearance / 4.0;
        let placed = field.insert_within(
          peak, radius,
          Primitive::new(move |p: Point<f64, D>| (p - peak).length() - radius));
        if placed {
          inserted += 1;
          if first.is_nan() {
            first = clearance;
          }
          last = clearance;
          if inserted % MEM_EVERY == 0 && field.memory_bytes() > MEM_CEILING {
            stopped = "memory";
            break;
          }
        }
      }
      let elapsed = t0.elapsed();

      let (mut slots, mut levels) = (0usize, 0u16);
      field.tree.traverse(&mut |n| {
        slots += n.data.len();
        levels = levels.max(n.depth);
        Ok(())
      }).ok();

      let row = Row {
        d: D, bite, inserted, first, last, levels,
        nodes: field.tree.node_count(),
        leaves: field.tree.leaf_count(),
        slots,
        bytes: field.memory_bytes(),
        elapsed, stopped,
        debug: format!("{:?}", field),
      };
      eprintln!("  D = {:>2}  {:>6}  {:>8} circles  depth {:>3}  {}",
        row.d, if bite { "bite" } else { "divide" }, row.inserted, row.levels, row.debug);
      row
    };

    [run(false), run(true)]
  }};
}

fn human(bytes: usize) -> String {
  const UNIT: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
  let (mut v, mut u) = (bytes as f64, 0);
  while v >= 1024.0 && u + 1 < UNIT.len() {
    v /= 1024.0;
    u += 1;
  }
  format!("{v:.1} {}", UNIT[u])
}

#[test] #[ignore] fn stress_kd_1_to_20() {
  eprintln!("k-d stress: {BUDGET:?} per dimension per arm, radius = clearance / 4, \
             {SPLITS} full subdivisions, prune_subdiv {PRUNE_SUBDIV}");
  eprintln!("the library's own default is `divide` below D = {CUT_MUST_PRUNE_MIN_DIMS} \
             and `bite` at and above it\n");

  let pairs = vec![
    stress!(1),  stress!(2),  stress!(3),  stress!(4),  stress!(5),
    stress!(6),  stress!(7),  stress!(8),  stress!(9),  stress!(10),
    stress!(11), stress!(12), stress!(13), stress!(14), stress!(15),
    stress!(16), stress!(17), stress!(18), stress!(19), stress!(20),
  ];

  println!("\n  {:>2} {:>7} {:>8} {:>8} {:>6} {:>9} {:>9} {:>7} {:>10} {:>8} {:>8} {:>7}",
    "D", "policy", "circles", "ins/s", "depth", "nodes", "leaves", "occ",
    "memory", "d first", "d last", "stop");
  println!("  {}", "-".repeat(116));
  for [divide, bite] in &pairs {
    for r in [divide, bite] {
      println!("  {:>2} {:>7} {:>8} {:>8.1} {:>6} {:>9} {:>9} {:>7.1} {:>10} {:>8.4} {:>8.4} {:>7}",
        r.d, if r.bite { "bite" } else { "divide" }, r.inserted, r.ins_per_s(),
        r.levels, r.nodes, r.leaves, r.occupancy(), human(r.bytes),
        r.first, r.last, r.stopped);
    }
  }

  println!("\n  bite ÷ divide — above 1.0 the policy wins\n");
  println!("  {:>2} {:>9} {:>9} {:>9} {:>7}", "D", "circles", "nodes", "memory", "default");
  println!("  {}", "-".repeat(50));
  for [divide, bite] in &pairs {
    let ratio = |b: f64, d: f64| if d > 0.0 { format!("×{:.2}", b / d) } else { "—".into() };
    println!("  {:>2} {:>9} {:>9} {:>9} {:>7}",
      divide.d,
      ratio(bite.inserted as f64, divide.inserted as f64),
      // inverted: fewer nodes and bytes is the win
      ratio(divide.nodes as f64, bite.nodes as f64),
      ratio(divide.bytes as f64, bite.bytes as f64),
      if divide.d >= CUT_MUST_PRUNE_MIN_DIMS { "bite" } else { "divide" });
  }

  println!("\n  full ADF stats");
  for [divide, bite] in &pairs {
    for r in [divide, bite] {
      println!("    D = {:>2}  {:>6}  {}", r.d, if r.bite { "bite" } else { "divide" }, r.debug);
    }
  }

  // The field must stay sane in every dimension and either arm: the walls still
  // bound it, and the clearance never exceeds the unit cube's inradius.
  for r in pairs.iter().flatten() {
    assert!(r.inserted > 0, "D = {}, bite = {}: nothing was placed at all", r.d, r.bite);
    assert!(r.first <= 0.5 + 1e-9,
      "D = {}, bite = {}: clearance {} exceeds the inradius", r.d, r.bite, r.first);
  }
}
