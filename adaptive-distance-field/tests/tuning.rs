//! Where each layout's optimum sits, over bucket capacity and depth budget.
//!
//! Both are pure performance knobs — the field is bit-identical at every setting,
//! since neither pruning nor the redundancy proof consults them — so the two
//! layouts need not be compared at one shared setting. Each can be tuned on its
//! own terms and the winners compared, which is the only comparison that says
//! anything about the layouts rather than about a default.
//!
//! The interesting tension for a binary layout: a leaf count `N` costs `log2 N`
//! levels of descent against `log_{2^D} N` for an all-axes layout, so [`Kd`] pays
//! `D` times the pointer chases for the same resolution. Fatter buckets buy back
//! levels at the price of primitives per query.
//!
//! ```text
//! cargo test -p adaptive-distance-field --release --test tuning -- --ignored --nocapture
//! ```

use {
  adaptive_distance_field::{
    adf::{tree::{Kd, Orthant}, Primitive, ADF},
    geometry::{Point, Vector, VectorExt},
    sdf::{self, SDF},
  },
  rand::prelude::*,
  std::time::{Duration, Instant},
};

const QUERIES: usize = 120_000;
const QUERY_PASSES: usize = 3;

struct Cell {
  bucket: usize,
  splits: u8,
  build: Duration,
  query: Duration,
  nodes: usize,
  leaves: usize,
  occupancy: f64,
  levels: u8,
  bytes: usize,
  checksum: u64,
}

/// Build a field at one `(bucket, splits)` setting and time it.
macro_rules! run {
  ($D:literal, $L:ty, $bucket:expr, $splits:expr, $balls:expr, $subdiv:expr) => {
    run!($D, $L, $bucket, $splits, $balls, $subdiv, 1u8)
  };
  ($D:literal, $L:ty, $bucket:expr, $splits:expr, $balls:expr, $subdiv:expr, $round:expr) => {{
    const D: usize = $D;
    let mut field = ADF::<f64, D, $L>::new($splits, vec![Primitive::new(sdf::boundary_rect)])
      .with_prune_subdiv($subdiv)
      .with_bucket_size($bucket)
      .with_split_round($round);

    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xC0FFEE);
    let t0 = Instant::now();
    for _ in 0..$balls {
      let c = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.1..0.9)));
      let r = rng.random_range(0.03..0.10);
      field.insert_within(c, r, Primitive::new(move |p: Point<f64, D>| (p - c).length() - r));
    }
    let build = t0.elapsed();

    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xBEEF);
    let pts: Vec<Point<f64, D>> = (0..QUERIES)
      .map(|_| Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0))))
      .collect();
    let mut sum = 0.0;
    let mut query = Duration::MAX;
    for _ in 0..QUERY_PASSES {
      let t1 = Instant::now();
      sum = pts.iter().map(|&p| field.sdf(p)).sum();
      query = query.min(t1.elapsed());
    }

    let (mut slots, mut levels) = (0usize, 0u8);
    field.tree.traverse(&mut |n| { slots += n.data.len(); levels = levels.max(n.depth); Ok(()) }).ok();
    let leaves = field.tree.leaf_count();
    Cell {
      bucket: $bucket, splits: $splits, build, query,
      nodes: field.tree.node_count(), leaves,
      occupancy: slots as f64 / leaves.max(1) as f64,
      levels, bytes: field.memory_bytes(), checksum: sum.to_bits(),
    }
  }};
}

fn head(what: &str) {
  println!("\n  {what}");
  println!("  {:>6} {:>6} {:>10} {:>10} {:>8} {:>8} {:>6} {:>7} {:>10}",
    "bucket", "splits", "build", "query", "nodes", "leaves", "lvls", "occ", "memory");
}

fn line(c: &Cell) {
  println!("  {:>6} {:>6} {:>10.2?} {:>10.2?} {:>8} {:>8} {:>6} {:>7.1} {:>9.1} KiB",
    c.bucket, c.splits, c.build, c.query, c.nodes, c.leaves, c.levels, c.occupancy,
    c.bytes as f64 / 1024.0);
}

/// Fastest query, and the settings that got there.
fn best(cells: &[Cell], of: fn(&Cell) -> Duration) -> &Cell {
  cells.iter().min_by_key(|c| of(c)).unwrap()
}

fn verdict(name: &str, orthant: &[Cell], kd: &[Cell]) {
  // every cell of a dimension must hold the same field — capacity and depth are
  // performance knobs, so a differing checksum would mean one of them is not
  let all: Vec<u64> = orthant.iter().chain(kd).map(|c| c.checksum).collect();
  assert!(all.windows(2).all(|w| w[0] == w[1]),
    "{name}: a knob changed the field — {:?}", all);

  let (oq, kq) = (best(orthant, |c| c.query), best(kd, |c| c.query));
  let (ob, kb) = (best(orthant, |c| c.build), best(kd, |c| c.build));
  println!("\n  {name}: best of each layout on its own terms");
  println!("    query   orthant {:>9.2?} (bucket {}, splits {})   k-d {:>9.2?} (bucket {}, splits {})   ratio ×{:.2}",
    oq.query, oq.bucket, oq.splits, kq.query, kq.bucket, kq.splits,
    kq.query.as_secs_f64() / oq.query.as_secs_f64());
  println!("    build   orthant {:>9.2?} (bucket {}, splits {})   k-d {:>9.2?} (bucket {}, splits {})   ratio ×{:.2}",
    ob.build, ob.bucket, ob.splits, kb.build, kb.bucket, kb.splits,
    kb.build.as_secs_f64() / ob.build.as_secs_f64());
  println!("    memory  orthant {:>7.1} KiB                    k-d {:>7.1} KiB                  ratio ×{:.2}",
    oq.bytes as f64 / 1024.0, kq.bytes as f64 / 1024.0, kq.bytes as f64 / oq.bytes as f64);
}

#[test] #[ignore] fn tune_d3() {
  const BALLS: usize = 1200;
  const SUBDIV: u32 = 3;

  let orthant = vec![
    run!(3, Orthant, 2, 5u8, BALLS, SUBDIV), run!(3, Orthant, 3, 5u8, BALLS, SUBDIV),
    run!(3, Orthant, 5, 5u8, BALLS, SUBDIV), run!(3, Orthant, 8, 5u8, BALLS, SUBDIV),
    run!(3, Orthant, 14, 5u8, BALLS, SUBDIV), run!(3, Orthant, 24, 5u8, BALLS, SUBDIV),
    run!(3, Orthant, 5, 3u8, BALLS, SUBDIV), run!(3, Orthant, 5, 4u8, BALLS, SUBDIV),
    run!(3, Orthant, 5, 6u8, BALLS, SUBDIV),
  ];
  let kd = vec![
    run!(3, Kd, 2, 5u8, BALLS, SUBDIV), run!(3, Kd, 3, 5u8, BALLS, SUBDIV),
    run!(3, Kd, 5, 5u8, BALLS, SUBDIV), run!(3, Kd, 8, 5u8, BALLS, SUBDIV),
    run!(3, Kd, 14, 5u8, BALLS, SUBDIV), run!(3, Kd, 24, 5u8, BALLS, SUBDIV),
    run!(3, Kd, 5, 3u8, BALLS, SUBDIV), run!(3, Kd, 5, 4u8, BALLS, SUBDIV),
    run!(3, Kd, 5, 6u8, BALLS, SUBDIV),
    run!(3, Kd, 8, 5u8, BALLS, SUBDIV, 3u8), run!(3, Kd, 14, 5u8, BALLS, SUBDIV, 3u8),
  ];

  println!("\nD = 3 — {BALLS} balls, {QUERIES} queries, prune_subdiv {SUBDIV}");
  head("orthant");
  orthant.iter().for_each(line);
  head("k-d");
  kd.iter().for_each(line);
  verdict("D = 3", &orthant, &kd);
}

#[test] #[ignore] fn tune_d6() {
  const BALLS: usize = 60;
  const SUBDIV: u32 = 2;

  let orthant = vec![
    run!(6, Orthant, 3, 2u8, BALLS, SUBDIV), run!(6, Orthant, 8, 2u8, BALLS, SUBDIV),
    run!(6, Orthant, 24, 2u8, BALLS, SUBDIV), run!(6, Orthant, 8, 1u8, BALLS, SUBDIV),
  ];
  let kd = vec![
    run!(6, Kd, 3, 2u8, BALLS, SUBDIV), run!(6, Kd, 8, 2u8, BALLS, SUBDIV),
    run!(6, Kd, 24, 2u8, BALLS, SUBDIV), run!(6, Kd, 8, 1u8, BALLS, SUBDIV),
    run!(6, Kd, 8, 3u8, BALLS, SUBDIV), run!(6, Kd, 24, 3u8, BALLS, SUBDIV),
    // a whole round of cuts per overflow, so the shrink test is applied at a cell
    // size where pruning can actually bite
    run!(6, Kd, 8, 2u8, BALLS, SUBDIV, 6u8), run!(6, Kd, 24, 2u8, BALLS, SUBDIV, 6u8),
  ];

  println!("\nD = 6 — {BALLS} balls, {QUERIES} queries, prune_subdiv {SUBDIV}");
  head("orthant");
  orthant.iter().for_each(line);
  head("k-d");
  kd.iter().for_each(line);
  verdict("D = 6", &orthant, &kd);
}
