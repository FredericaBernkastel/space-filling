//! Cost of the two subdivision layouts, at `D = 3` and `D = 6`.
//!
//! Both layouts are given the *same* insertion sequence and the same depth
//! budget in full subdivisions, so they end up representing the same field at the
//! same resolution — a fact this benchmark also checks, since a disagreement
//! would make the timings meaningless. What differs is the branching factor: an
//! [`Orthant`] node splits into `2^D` children and pays one redundancy proof per
//! child, a [`Kd`] node splits into 2 and pays two.
//!
//! Diagnostics, not assertions about speed — run on demand:
//!
//! ```text
//! cargo test -p adaptive-distance-field --release --test layout -- --ignored --nocapture
//! ```

use {
  adaptive_distance_field::{
    adf::{tree::{Kd, Layout, Orthant}, Primitive, ADF},
    geometry::{Point, Vector, VectorExt},
    sdf::{self, SDF},
  },
  rand::prelude::*,
  std::time::{Duration, Instant},
};

struct Report {
  layout: &'static str,
  children: usize,
  build: Duration,
  query: Duration,
  nodes: usize,
  leaves: usize,
  splits: usize,
  slots: usize,
  levels: u16,
  bytes: usize,
  checksum: f64,
}

fn row(r: &Report, queries: usize) {
  println!(
    "  {:<8} {:>4} {:>10.3?} {:>10.3?} {:>9} {:>9} {:>8} {:>7} {:>7} {:>11}",
    r.layout, r.children, r.build, r.query, r.nodes, r.leaves, r.splits,
    r.slots / r.leaves.max(1), r.levels, human(r.bytes),
  );
  let _ = queries;
}

fn human(bytes: usize) -> String {
  const UNIT: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
  let mut v = bytes as f64;
  let mut u = 0;
  while v >= 1024.0 && u + 1 < UNIT.len() {
    v /= 1024.0;
    u += 1;
  }
  format!("{v:.1} {}", UNIT[u])
}

/// One `(layout, dimension)` cell of the benchmark: build a field from `balls`
/// pseudo-random balls, then time `queries` point samples of it.
macro_rules! measure {
  ($D:literal, $L:ty, $splits:expr, $balls:expr, $queries:expr, $probes:expr) => {{
    const D: usize = $D;
    // The redundancy proof refines boxes in the same layout, so its budget is
    // charged identically to both cells; kept well below the default 8 because at
    // D = 6 an orthant proof fans out 64-way per level, and this benchmark is
    // about the tree rather than about the proof.
    let mut field = ADF::<f64, D, $L>::new($splits, vec![Primitive::new(sdf::boundary_rect)])
      .with_prune_subdiv(PRUNE_SUBDIV);

    // Same seed for every cell, so every layout sees the same insertion sequence.
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xC0FFEE);
    let t0 = Instant::now();
    for _ in 0..$balls {
      let centre = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.1..0.9)));
      let radius = rng.random_range(0.03..0.10);
      field.insert_within(centre, radius,
        Primitive::new(move |p: Point<f64, D>| (p - centre).length() - radius));
    }
    let build = t0.elapsed();

    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xBEEF);
    let pts: Vec<Point<f64, D>> = (0..$queries)
      .map(|_| Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0))))
      .collect();
    // Fastest of several passes rather than one timed pass: a single query sweep
    // is memory-bandwidth bound and swung by a factor of two between runs, while
    // the minimum is the least noisy estimator of the cost with nothing else
    // competing. The sum keeps the loop from being optimized away, and doubles as
    // a check that both layouts answered identically.
    let mut checksum = 0.0;
    let mut query = Duration::MAX;
    for _ in 0..QUERY_PASSES {
      let t1 = Instant::now();
      checksum = pts.iter().map(|&p| field.sdf(p)).sum();
      query = query.min(t1.elapsed());
    }

    let (mut slots, mut levels) = (0usize, 0u16);
    field.tree.traverse(&mut |n| { slots += n.data.len(); levels = levels.max(n.depth); Ok(()) }).ok();
    let children = <$L as Layout<D>>::CHILDREN;

    // `probes` are compared across layouts to prove they hold the same field
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xf1e1d);
    let probe: Vec<f64> = (0..$probes)
      .map(|_| field.sdf(Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)))))
      .collect();

    (Report {
      layout: <$L as Layout<D>>::NAME,
      children,
      build,
      query,
      nodes: field.tree.node_count(),
      leaves: field.tree.leaf_count(),
      splits: (field.tree.node_count() - 1) / children,
      slots,
      levels,
      bytes: field.memory_bytes(),
      checksum,
    }, probe)
  }};
}

fn header(title: &str) {
  println!("\n{title}");
  println!("  {:<8} {:>4} {:>10} {:>10} {:>9} {:>9} {:>8} {:>7} {:>7} {:>11}",
    "layout", "2^D", "build", "query", "nodes", "leaves", "splits", "leaf buk", "levels", "memory");
  println!("  {}", "-".repeat(94));
}

fn compare(a: &Report, b: &Report, pa: &[f64], pb: &[f64]) {
  let worst = pa.iter().zip(pb).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
  assert!(worst == 0.0, "layouts disagree on the field by {worst:e} — timings are not comparable");
  println!("  {}", "-".repeat(94));
  println!("  field agreement over {} probes: exact (max |Δ| = 0)", pa.len());
  println!("  k-d relative to orthant:  build ×{:.2}   query ×{:.2}   memory ×{:.2}   nodes ×{:.2}",
    b.build.as_secs_f64() / a.build.as_secs_f64(),
    b.query.as_secs_f64() / a.query.as_secs_f64(),
    b.bytes as f64 / a.bytes as f64,
    b.nodes as f64 / a.nodes as f64);
  // each subdivision runs one redundancy proof per child per bucket slot, so the
  // branching factor is the dominant term in build cost
  println!("  proofs charged to subdivision (splits × children): orthant {}, k-d {}",
    a.splits * a.children, b.splits * b.children);
}

/// Proof refinement budget, in full subdivisions — the ADF default is 8, which at
/// D = 6 makes the *proof* rather than the tree dominate the clock.
const PRUNE_SUBDIV: u32 = 3;
const QUERIES: usize = 300_000;
/// Timed query sweeps per cell; the fastest is reported.
const QUERY_PASSES: usize = 5;
const PROBES: usize = 1_000;

macro_rules! bench {
  ($name:ident, $D:literal, $splits:expr, $balls:expr) => {
    #[test] #[ignore] fn $name() {
      let (orthant, pa) = measure!($D, Orthant, $splits, $balls, QUERIES, PROBES);
      let (kd, pb) = measure!($D, Kd, $splits, $balls, QUERIES, PROBES);

      header(&format!("D = {} — {} balls, {} full subdivisions, {} queries, prune_subdiv {}",
        $D, $balls, $splits, QUERIES, PRUNE_SUBDIV));
      row(&orthant, QUERIES);
      row(&kd, QUERIES);
      compare(&orthant, &kd, &pa, &pb);
      assert_eq!(orthant.checksum.to_bits(), kd.checksum.to_bits());
    }
  };
}

// Ball count falls with dimension because the orthant cell — the thing being
// measured — grows as 2^D per split. The k-d column would happily take more.
// Sized so each benchmark spends a couple of seconds per layout: short enough to
// run on demand, long enough that timer noise and scheduling jitter are well
// below the differences being reported.
bench!(bench_d2, 2, 6u8, 6000usize);
bench!(bench_d3, 3, 5u8, 1500usize);
bench!(bench_d6, 6, 2u8, 150usize);
