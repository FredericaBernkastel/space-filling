//! Weight-ordered axes against round-robin, on domains whose extents decay.
//!
//! Roadmap step 2 of `doc/publications/infinite_dimensions`: subdivide in
//! descending weight so that cost follows the *effective* dimension rather than
//! the ambient one. Weights are the domain box's extents — `γᵢ = (i+1)^(−s)`,
//! normalised to `γ₀ = 1` — which is the ellipsoid `Ωₐ` of §2.3 in axis-aligned
//! clothing. `s = 0` is the cube, and the control: there the two layouts must
//! agree, since every axis weighs the same.
//!
//! The certificate clears a cell when the margin covers `(L_f + L_g)·h(R)` with
//! `h = ½√(Σ sᵢ²)`, so halving side `sᵢ` buys a reduction proportional to `sᵢ²`.
//! [`Widest`] takes that greedily; [`Cyclic`] spends every `D`th cut on an axis
//! that may contribute nothing to the diameter at all.
//!
//! ```text
//! cargo test -p adaptive-distance-field --release --test weights -- --ignored --nocapture
//! ```

use {
  adaptive_distance_field::{
    adf::{tree::{Kd, WeightedKd}, Layout, Primitive, ADF},
    geometry::{Aabb, Point, Vector, VectorExt},
    sdf::{self, SDF},
  },
  rand::prelude::*,
  std::time::Instant,
};

const D: usize = 6;
const BALLS: usize = 500;
const SPLITS: u8 = 3;
const PRUNE_SUBDIV: u32 = 2;
const QUERIES: usize = 200_000;

/// `γᵢ = (i+1)^(−s)`, normalised to `γ₀ = 1`. `s = 0` is the unit cube.
fn domain(s: f64) -> Aabb<f64, D> {
  let gamma = Vector::<f64, D>::from_fn(|i, _| ((i + 1) as f64).powf(-s));
  Aabb::new(Point::from(Vector::zeros()), Point::from(gamma))
}

/// A fixed insertion sequence, scaled into `dom`.
///
/// Deliberately not ascent-driven. `LineSearch` steps outside a thin domain
/// routinely, and outside it `sdf` falls back to the root's bucket — which the
/// two layouts prune differently, so an ascent would feed each tree a *different*
/// sequence and the comparison would stop being like for like. The same reason
/// `layouts_agree_on_the_field` drives insertions from a seed.
fn sequence(dom: Aabb<f64, D>) -> Vec<(Point<f64, D>, f64)> {
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0x5EED_0F_A5);
  let thinnest = (0..D).fold(f64::MAX, |a, i| a.min(dom.size()[i]));
  (0..BALLS)
    .map(|_| {
      let centre = Point::from(Vector::<f64, D>::from_fn(|i, _| {
        // kept clear of the walls, so a ball never pokes out of a thin axis
        (0.15 + rng.random_range(0.0..0.7)) * dom.size()[i]
      }));
      (centre, rng.random_range(0.04..0.12) * thinnest)
    })
    .collect()
}

struct Run {
  build: f64,
  query: f64,
  nodes: usize,
  leaves: usize,
  slots: usize,
  bytes: usize,
  placed: usize,
  checksum: u64,
}

fn run<L>(s: f64, policy: bool, seq: &[(Point<f64, D>, f64)]) -> Run
where
  L: Layout<D>,
  L::Children<adaptive_distance_field::adf::tree::Split<
    adaptive_distance_field::adf::Bucket<f64, D>, D, L>>: Send,
{
  let dom = domain(s);
  let mut field = ADF::<f64, D, L>::new_in(dom, SPLITS, vec![Primitive::new(sdf::boundary_box(dom))])
    .with_prune_subdiv(PRUNE_SUBDIV)
    .with_cut_must_prune(policy);
  let mut placed = 0;
  let t0 = Instant::now();
  for &(centre, radius) in seq {
    if field.insert_within(
      centre, radius,
      Primitive::new(move |p: Point<f64, D>| (p - centre).length() - radius))
    {
      placed += 1;
    }
  }
  let build = t0.elapsed().as_secs_f64() * 1e3;

  // min of three sweeps: one is bandwidth-bound and swings badly between runs
  let mut probe = rand_pcg::Pcg64::seed_from_u64(0xC0FFEE);
  let pts: Vec<_> = (0..QUERIES / 20)
    .map(|_| Point::from(Vector::<f64, D>::from_fn(|i, _| {
      probe.random_range(0.0..1.0) * dom.size()[i]
    })))
    .collect();
  let mut query = f64::MAX;
  let mut checksum = 0u64;
  for _ in 0..3 {
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..20 {
      for p in &pts {
        acc = acc.wrapping_mul(31).wrapping_add(field.sdf(*p).to_bits());
      }
    }
    query = query.min(t.elapsed().as_secs_f64() * 1e3);
    checksum = acc;
  }

  let (mut slots, mut leaves) = (0usize, 0usize);
  field.tree.traverse(&mut |n| {
    slots += n.data.len();
    if n.is_leaf() { leaves += 1 }
    Ok(())
  }).ok();

  Run {
    build, query,
    nodes: field.tree.node_count(),
    leaves, slots,
    bytes: field.memory_bytes(),
    placed, checksum,
  }
}

fn human(bytes: usize) -> String {
  const UNIT: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
  let (mut v, mut u) = (bytes as f64, 0);
  while v >= 1024.0 && u + 1 < UNIT.len() { v /= 1024.0; u += 1 }
  format!("{v:.1} {}", UNIT[u])
}

#[test] #[ignore] fn weight_ordered_axes() {
  println!("\nD = {D}, {BALLS} insertions, {SPLITS} full subdivisions, prune_subdiv {PRUNE_SUBDIV}");
  println!("weights γᵢ = (i+1)^(−s), the domain box's extents\n");
  println!("  {:>4} {:>11} {:>9} {:>9} {:>8} {:>8} {:>7} {:>9} {:>8}",
    "s", "layout", "build ms", "query ms", "nodes", "leaves", "occ", "memory", "placed");
  println!("  {}", "-".repeat(88));

  for &s in &[0.0f64, 0.5, 1.0, 2.0] {
    let seq = sequence(domain(s));
    let cyc = run::<Kd>(s, false, &seq);
    let wid = run::<WeightedKd>(s, false, &seq);

    // the field is the field: pruning is sound under either cut order
    assert_eq!(cyc.checksum, wid.checksum,
      "s = {s}: the two cut orders disagree on the field");
    assert_eq!(cyc.placed, wid.placed, "s = {s}: different ball counts");

    for (name, r) in [("k-d", &cyc), ("widest", &wid)] {
      println!("  {:>4} {:>11} {:>9.1} {:>9.1} {:>8} {:>8} {:>7.1} {:>9} {:>8}",
        format!("{s:.1}"), name, r.build, r.query, r.nodes, r.leaves,
        r.slots as f64 / r.leaves.max(1) as f64, human(r.bytes), r.placed);
    }
    println!("  {:>4} {:>11} {:>9} {:>9} {:>8} {:>8}",
      "", "widest ÷",
      format!("×{:.2}", wid.build / cyc.build),
      format!("×{:.2}", wid.query / cyc.query),
      format!("×{:.2}", wid.nodes as f64 / cyc.nodes as f64),
      format!("×{:.2}", wid.bytes as f64 / cyc.bytes as f64));
  }

  println!("\n  with the default split policy on (D = 6 ⇒ a cut must prune to be kept)");
  println!("  {:>4} {:>11} {:>9} {:>8}", "s", "layout", "nodes", "placed");
  println!("  {}", "-".repeat(40));
  for &s in &[0.0f64, 1.0, 2.0] {
    let seq = sequence(domain(s));
    for (name, r) in [("k-d", run::<Kd>(s, true, &seq)), ("widest", run::<WeightedKd>(s, true, &seq))] {
      println!("  {:>4} {:>11} {:>9} {:>8}", format!("{s:.1}"), name, r.nodes, r.placed);
    }
  }
}
