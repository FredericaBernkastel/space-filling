//! The structure's own soundness properties, expressed without a shape
//! catalogue — a ball is three lines of vector algebra, and relying on nothing
//! else keeps these tests inside the crate whose claims they check.
//!
//! The shape- and drawing-driven tests (and the `#[ignore]`d benchmarks) live in
//! `space-filling`, as `tests/adf_shapes.rs` and `tests/adf_drawing.rs`.

use {
  super::*,
  crate::{
    geometry::{DistPoint, Point, Vector, VectorExt, P2},
    line_search::LineSearch,
    sdf,
  },
  rand::prelude::*,
};

/// A ball as a bare field: `|p - centre| - radius`, exactly 1-Lipschitz.
fn ball<const D: usize>(
  centre: Point<f64, D>,
  radius: f64,
) -> impl Fn(Point<f64, D>) -> f64 + Send + Sync + 'static {
  move |p| (p - centre).length() - radius
}

/// One random restart of the ascent, reporting the peak it reached and the
/// clearance there. Stands in for `space-filling`'s batched parallel search;
/// climbing one at a time is if anything the stricter test, since no placement
/// is ever made against a stale snapshot.
fn climb<const D: usize>(
  f: impl Fn(Point<f64, D>) -> f64,
  rng: &mut impl Rng,
) -> DistPoint<f64, f64, D> {
  let start = Point::from(
    Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)));
  let point = LineSearch::default().optimize(&f, start);
  DistPoint { distance: f(point), point }
}

/// The whole pipeline in three dimensions: an octree-backed ADF over the unit
/// cube, N-dimensional gradient ascent, and the D*-pruned insertion walk. Each
/// ball takes half the clearance at its maximum, so no two may ever intersect —
/// after inserting a ball of radius `r₁` at `p₁`, any later peak satisfies
/// `d₂ ≤ |p₂ - p₁| - r₁`, and `r₂ = d₂/2 ≤ d₂`. An overlap would therefore prove
/// a corrupted field or a broken walk.
#[test] fn sphere_packing_3d() {
  let mut adf = ADF::<f64, 3, tree::Orthant>::new(4, vec![Primitive::new(sdf::boundary_rect)]);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut spheres: Vec<(Point<f64, 3>, f64)> = vec![];

  let mut attempts = 0;
  while spheres.len() < 100 {
    attempts += 1;
    assert!(attempts < 10_000, "field saturated after {} spheres", spheres.len());
    let peak = climb(|p| adf.sdf(p), &mut rng);
    if peak.distance <= 1e-9 { continue; }
    let r = peak.distance / 2.0;
    if adf.insert_at_maximum(peak, Primitive::new(ball(peak.point, r))) {
      spheres.push((peak.point, r));
    }
  }

  for i in 0..spheres.len() {
    for j in i + 1..spheres.len() {
      let ((c1, r1), (c2, r2)) = (spheres[i], spheres[j]);
      let gap = (c1 - c2).length() - (r1 + r2);
      assert!(gap > -1e-9, "spheres {i} and {j} intersect by {:.3e}", -gap);
    }
  }
  // and the field itself remembers them: at each centre it reads exactly -r
  for &(c, r) in &spheres {
    assert!((adf.sdf(c) + r).abs() < 1e-9);
  }
}

/// Constructive proof that no constant-sized insertion domain is sound for local
/// maxima. Three point-like obstacles at 90°, 210°, 330° around `x₀` form a
/// strict local maximum (contact gaps 120° < 180°). Along the escape bisector
/// `w = 270°` the field grows as `g(x₀ + R·w) = √(R² − R·d + d²) ≈ R − d/2`,
/// while a primitive confined to `B̄(x₀, d)` reaches only `f(v) ≥ R − d` —
/// strictly below `g` for every `R`, until the outer boundary caps it. The update
/// region therefore extends arbitrarily many multiples of `d`, and the historical
/// `4√2·d` square provably leaves stale field behind, where
/// [`ADF::update_domain`] covers it exactly.
#[test] fn insertion_domain() {
  let x0 = P2::new(0.5, 0.75);
  let d = 0.05;
  let full = Aabb::unit();

  let mut adf = ADF::<f64, 2, tree::Orthant>::new(7, vec![Primitive::new(sdf::boundary_rect)]);
  for angle in [90f64, 210., 330.] {
    let (s, c) = angle.to_radians().sin_cos();
    let obstacle = ball(P2::new(x0.x + c * d, x0.y + s * d), 1e-4);
    adf.insert_sdf_domain(full, Arc::new(obstacle));
  }

  let local_max = DistPoint { point: x0, distance: adf.sdf(x0) };
  // pipeline-style placement: a ball inside the maximal ball, pushed toward `w`
  let r = 0.01;
  let placed = P2::new(x0.x, x0.y - (local_max.distance - r));
  let f: Arc<dyn Fn(P2<f64>) -> f64 + Send + Sync> = Arc::new(ball(placed, r));

  let probe = P2::new(0.48, 0.45); // R = 6·d down the escape ray
  let truth = f(probe).min(adf.sdf(probe));
  assert!(f(probe) < adf.sdf(probe), "the insertion must lower the field at the probe");

  { // the historical `4·√2·d` square misses the probe's leaf → stale field
    let mut adf = adf.clone();
    let half = local_max.distance * 2.0 * 2f64.sqrt();
    let empirical = Aabb::new(
      P2::new(x0.x - half, x0.y - half),
      P2::new(x0.x + half, x0.y + half));
    adf.insert_sdf_domain(empirical, f.clone());
    assert!(adf.sdf(probe) - truth > 0.02,
      "expected the historical constant domain to corrupt the field");
  }

  { // the adaptive domain covers D* → exact field
    let mut adf = adf.clone();
    adf.insert_sdf_domain(adf.update_domain(local_max), f.clone());
    assert!((adf.sdf(probe) - truth).abs() < 1e-12);
  }

  { // the fused D*-pruned walk is exact as well
    let mut adf = adf.clone();
    adf.insert_at_maximum(local_max, Primitive { f: f.clone(), lipschitz: 1.0 });
    assert!((adf.sdf(probe) - truth).abs() < 1e-12);
  }
}

/// The two layouts must represent the *same field*, not merely similar ones.
///
/// Pruning is sound in both — a primitive is dropped only when provably
/// redundant over the whole node — so the stored `min` equals the true `min` over
/// everything inserted, at every point. The trees differ in shape and in which
/// primitives each bucket keeps; the field they answer with cannot differ at all.
/// Agreement to the last bit is therefore the right assertion, and any k-d
/// descent or child-cell error shows up here immediately.
#[test] fn layouts_agree_on_the_field() {
  // A literal `D` per instantiation: generic code here would need the whole
  // `Branching`/`Send` bound cascade spelled out for no gain.
  macro_rules! check {($D:literal) => {{
    const D: usize = $D;
    let mut orthant = ADF::<f64, D, tree::Orthant>::new(4, vec![Primitive::new(sdf::boundary_rect)]);
    let mut kd = ADF::<f64, D, tree::Kd>::new(4, vec![Primitive::new(sdf::boundary_rect)]);

    // identical insertion sequence, driven by a fixed seed rather than by either
    // tree's own maxima, so the two are compared on exactly the same input
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xf1e1d);
    for _ in 0..40 {
      let centre = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.15..0.85)));
      let radius = rng.random_range(0.02..0.09);
      orthant.insert_within(centre, radius, Primitive::new(ball(centre, radius)));
      kd.insert_within(centre, radius, Primitive::new(ball(centre, radius)));
    }

    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xf1e1e);
    let mut worst = 0.0f64;
    for _ in 0..4000 {
      let p = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)));
      worst = worst.max((orthant.sdf(p) - kd.sdf(p)).abs());
    }
    assert!(worst == 0.0, "D = {}: layouts disagree by {:e}", D, worst);

    // ... and the k-d tree really is binary, with the deeper level count that
    // buys the same resolution
    assert!(kd.tree.node_count() > 1, "D = {}: k-d tree never subdivided", D);
  }}}
  check!(2);
  check!(3);
}

/// A k-d ADF works in a dimension where the orthant layout cannot be
/// instantiated at all: `Dim<8>` has no `Branching` impl, and a 256-way branch
/// would be useless even if it did.
#[test] fn kd_lifts_the_dimension_ceiling() {
  let mut field = ADF::<f64, 8, tree::Kd>::new(2, vec![Primitive::new(sdf::boundary_rect)]);
  let centre = Point::from([0.5; 8]);
  assert!(field.insert_within(centre, 0.2, Primitive::new(ball(centre, 0.2))));
  // inside the inserted ball the field is negative, and the boundary still bounds
  assert!(field.sdf(centre) < 0.0);
  assert!(field.sdf(Point::from([0.001; 8])) < 0.01);
  assert_eq!(field.layout_name(), "k-d");
}

/// Outside the domain there is no leaf to descend to, so `sdf` answers from the
/// root's own bucket. That makes the root the one node whose primitives must
/// outlive its subdivision — and it is easy to lose, because every test that
/// probes inside the cube passes regardless.
///
/// Losing it is not a small error: an empty bucket reads `+MAX/2`, the ascent
/// climbs out of the domain toward it, and a ball of radius `MAX/4` is inserted
/// at that "maximum". Pin the behaviour instead.
#[test] fn out_of_domain_reads_the_seed_field() {
  let mut adf = ADF::<f64, 3, tree::Orthant>::new(3, vec![Primitive::new(sdf::boundary_rect)]);
  let outside = [
    Point::from([-0.25, 0.5, 0.5]),
    Point::from([0.5, 1.75, 0.5]),
    Point::from([-1.0, -1.0, -1.0]),
  ];
  let seed: Vec<f64> = outside.iter().map(|&p| sdf::boundary_rect(p)).collect();
  assert!(seed.iter().all(|&v| v < 0.0), "the seed must be negative outside");

  // enough insertions to divide the root several times over
  let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
  for _ in 0..30 {
    let c = Point::from(Vector::<f64, 3>::from_fn(|_, _| rng.random_range(0.2..0.8)));
    adf.insert_within(c, 0.05, Primitive::new(ball(c, 0.05)));
  }
  assert!(adf.tree.node_count() > 9, "the root should have divided by now");

  for (&p, &want) in outside.iter().zip(&seed) {
    let got = adf.sdf(p);
    assert!(got <= want + 1e-12,
      "outside the domain the field must still be bounded by the seed:        got {got:e} at {p:?}, seed says {want:e}");
    assert!(got.is_finite() && got.abs() < 1e3, "outside reads a sentinel: {got:e}");
  }
}

/// A multi-level split — one overflow dividing through a whole round of axis cuts
/// — must represent the same field as the one-level default. It is off by default
/// because it does not pay (see CHANGELOG.md), so without this the recursive graft
/// would go untested.
#[test] fn a_split_round_changes_the_tree_but_not_the_field() {
  const D: usize = 3;
  let seed = vec![Primitive::new(sdf::boundary_rect)];
  let mut flat = ADF::<f64, D, tree::Kd>::new(3, seed.clone());
  let mut round = ADF::<f64, D, tree::Kd>::new(3, seed)
    .with_split_round(<tree::Kd as Layout<D>>::LEVELS_PER_SPLIT as u8);

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0xa11ce);
  for _ in 0..60 {
    let c = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.1..0.9)));
    let r = rng.random_range(0.02..0.08);
    flat.insert_within(c, r, Primitive::new(ball(c, r)));
    round.insert_within(c, r, Primitive::new(ball(c, r)));
  }

  // the round really does build a different tree ...
  assert!(round.tree.node_count() > flat.tree.node_count(),
    "a full round should divide further: {} vs {}",
    round.tree.node_count(), flat.tree.node_count());

  // ... and answers identically everywhere, which is the only thing that matters
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0xa11ce2);
  let mut worst = 0.0f64;
  for _ in 0..3000 {
    let p = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)));
    worst = worst.max((flat.sdf(p) - round.sdf(p)).abs());
  }
  assert!(worst == 0.0, "split round changed the field by {worst:e}");
}

/// Refusing a division that prunes nothing cannot move the field by a single bit:
/// it only declines to store two copies of the same bucket.
///
/// `D = 12` is above [`CUT_MUST_PRUNE_MIN_DIMS`], so the default is `true` here and
/// the comparison is against an explicitly disabled field. `prune_subdiv` is 1
/// because the default of 8 costs seconds per insertion at this dimension.
#[test] fn cut_must_prune_leaves_the_field_alone() {
  const D: usize = 12;
  let build = |require: bool| {
    let mut field = ADF::<f64, D, tree::Kd>::new(3, vec![Primitive::new(sdf::boundary_rect)])
      .with_prune_subdiv(1)
      .with_cut_must_prune(require);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xC0_FF_EE);
    for _ in 0..12 {
      let c = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.3..0.7)));
      field.insert_within(c, 0.05, Primitive::new(ball(c, 0.05)));
    }
    field
  };
  let (fat, deep) = (build(true), build(false));

  // The default at this dimension is the policy, and it is the policy that pays.
  assert!(ADF::<f64, D, tree::Kd>::new(1, vec![]).cut_must_prune);
  assert!(fat.tree.node_count() < deep.tree.node_count() / 8,
    "the refused cuts should collapse the arena: {} against {}",
    fat.tree.node_count(), deep.tree.node_count());

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0x5EED);
  for _ in 0..2000 {
    let p = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)));
    assert_eq!(fat.sdf(p).to_bits(), deep.sdf(p).to_bits(),
      "the field moved at {p:?}: {} against {}", fat.sdf(p), deep.sdf(p));
  }
}

/// Every node's depth, cell and occupancy in traversal order — a structural
/// fingerprint, strictly stronger than agreeing on the field.
fn fingerprint<const D: usize, L: tree::Layout<D>>(
  f: &ADF<f64, D, L>,
) -> Vec<(u8, [u64; D], [u64; D], usize)> {
  let mut out = vec![];
  f.tree.traverse(&mut |n| {
    out.push((
      n.depth,
      std::array::from_fn(|a| n.rect.min[a].to_bits()),
      std::array::from_fn(|a| n.rect.max[a].to_bits()),
      n.data.len(),
    ));
    Ok(())
  }).ok();
  out
}

fn pack_cube<const D: usize, L: tree::Layout<D>>(subdiv: u32) -> ADF<f64, D, L>
where
  L::Children<tree::Split<Bucket<f64, D>, D, L>>: Send,
{
  let mut field = ADF::<f64, D, L>::new(3, vec![Primitive::new(sdf::boundary_rect)])
    // forced off, or D = 5 collapses to a single leaf and the comparison is vacuous
    .with_cut_must_prune(false)
    .with_prune_subdiv(subdiv);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0xB0A);
  for _ in 0..40 {
    let c = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.2..0.8)));
    field.insert_within(c, 0.06, Primitive::new(ball(c, 0.06)));
  }
  field
}

/// On a cube the widest axis is whichever was cut least recently, so [`Widest`]
/// walks `0, 1, .., D-1` and back — exactly [`Cyclic`].
///
/// Pinned on the schedule itself, and on the tree only while the proof is barred
/// from refining, because with `prune_subdiv > 0` the *trees* legitimately
/// diverge: [`sdf_geq_everywhere_in`] subdivides through the same [`Layout`], and
/// the box it starts from is the node's cell — which is not a cube at any depth
/// that is not a multiple of `D`. [`Widest`] cuts those along their longest axis,
/// so the proofs land differently and prune differently. That is a gain, not a
/// discrepancy: it is the same greedy reduction of `h` applied to the branch and
/// bound. What can never differ is the field, pruning being sound either way.
#[test] fn widest_reduces_to_cyclic_on_a_cube() {
  const D: usize = 5;

  // the schedule proper, level for level over three full rounds
  let mut cell = Aabb::<f64, D>::unit();
  for depth in 0..3 * D as u8 {
    let w = <Widest as CutPolicy<D>>::axis(&cell, depth);
    assert_eq!(w, <Cyclic as CutPolicy<D>>::axis(&cell, depth),
      "the cut schedules part company at depth {depth}");
    cell.max[w] = cell.center()[w];
  }

  // with the proof unable to refine, the trees are identical node for node
  let (cyclic, widest) = (pack_cube::<D, Kd>(0), pack_cube::<D, WeightedKd>(0));
  assert!(cyclic.tree.node_count() > 100, "the comparison needs a real tree");
  let (a, b) = (fingerprint(&cyclic), fingerprint(&widest));
  if let Some((i, (x, y))) = a.iter().zip(&b).enumerate().find(|(_, (x, y))| x != y) {
    let cell = |v: &[u64; D]| v.map(f64::from_bits);
    panic!("node {i} of {} differs
  cyclic depth {} {:?}..{:?} len {}
  widest depth {} {:?}..{:?} len {}",
      a.len().min(b.len()),
      x.0, cell(&x.1), cell(&x.2), x.3,
      y.0, cell(&y.1), cell(&y.2), y.3);
  }
  assert_eq!(a.len(), b.len(), "same cells, different node counts");

  // let the proof refine and the trees may part; the field still may not
  let (cyclic, widest) = (pack_cube::<D, Kd>(2), pack_cube::<D, WeightedKd>(2));
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0x5EED);
  for _ in 0..2000 {
    let p = Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(0.0..1.0)));
    assert_eq!(cyclic.sdf(p).to_bits(), widest.sdf(p).to_bits(),
      "the field moved at {p:?}");
  }
}

/// The whole point: on an elongated domain the cuts go where the diameter is,
/// spending four levels on the long axis before touching either short one.
#[test] fn widest_spends_its_cuts_on_the_long_axis() {
  const D: usize = 3;
  let mut cell = Aabb::new(Point::from([0.0; D]), Point::from([8.0, 1.0, 1.0]));
  let mut schedule = vec![];
  for depth in 0..7u8 {
    let a = <Widest as CutPolicy<D>>::axis(&cell, depth);
    schedule.push(a);
    cell.max[a] = cell.center()[a];
  }
  // 8 → 4 → 2 → 1 on axis 0 (the last of those a tie it wins by index), then the
  // two short axes, then round-robin again now that the box is a cube
  assert_eq!(schedule, vec![0, 0, 0, 0, 1, 2, 0]);

  // Cyclic is indifferent to all of that
  let cyclic: Vec<_> = (0..7u8)
    .map(|d| <Cyclic as CutPolicy<D>>::axis(&cell, d))
    .collect();
  assert_eq!(cyclic, vec![0, 1, 2, 0, 1, 2, 0]);
}

/// An anisotropic domain needs its walls to match, or the seed field bounds the
/// wrong box.
#[test] fn a_domain_box_bounds_itself() {
  const D: usize = 3;
  let domain = Aabb::new(Point::from([0.0; D]), Point::from([8.0, 1.0, 1.0]));
  let walls = sdf::boundary_box(domain);
  let field = ADF::<f64, D, WeightedKd>::new_in(domain, 2, vec![Primitive::new(walls)]);

  assert_eq!(field.tree.root().rect.min, domain.min);
  assert_eq!(field.tree.root().rect.max, domain.max);
  // the centre of the long box is half a unit from the near walls, not four
  assert!((field.sdf(Point::from([4.0, 0.5, 0.5])) - 0.5).abs() < 1e-12);
  assert!(field.sdf(Point::from([0.02, 0.5, 0.5])) > 0.0, "just inside");
  assert!(field.sdf(Point::from([-0.5, 0.5, 0.5])) < 0.0, "outside the near wall");

  // and it agrees with `boundary_rect` when the box *is* the unit cube
  let unit = sdf::boundary_box(Aabb::<f64, D>::unit());
  for p in [[0.5; D], [0.1, 0.9, 0.5], [0.0; D], [1.5, 0.2, 0.2]] {
    let p = Point::from(p);
    assert_eq!(unit(p).to_bits(), sdf::boundary_rect(p).to_bits());
  }
}

/// A field over a non-cubic domain is still the exact `min` over everything
/// inserted, checked against brute force rather than against another tree.
///
/// The descent used to start from `Aabb::unit()` whatever the root actually was,
/// so every anisotropic domain silently resolved to the wrong leaf: 44 627 of
/// 200 000 probes wrong, by as much as 0.1. Two trees compared against each other
/// would not have caught it — both were wrong — which is why this one is pinned
/// against the primitives themselves.
#[test] fn an_anisotropic_domain_reads_exactly() {
  const D: usize = 3;
  fn check<L: tree::Layout<D>>(extents: [f64; D])
  where
    L::Children<tree::Split<Bucket<f64, D>, D, L>>: Send,
  {
    let dom = Aabb::new(Point::from([0.0; D]), Point::from(extents));
    let walls = sdf::boundary_box(dom);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
    let balls: Vec<_> = (0..30)
      .map(|_| {
        let c = Point::from(Vector::<f64, D>::from_fn(|i, _| {
          (0.15 + rng.random_range(0.0..0.7)) * dom.size()[i]
        }));
        (c, rng.random_range(0.01..0.03))
      })
      .collect();

    let mut field = ADF::<f64, D, L>::new_in(dom, 3, vec![Primitive::new(walls)])
      .with_prune_subdiv(2)
      .with_cut_must_prune(false);
    for &(c, r) in &balls {
      field.insert_within(c, r, Primitive::new(ball(c, r)));
    }

    let mut rng = rand_pcg::Pcg64::seed_from_u64(99);
    for _ in 0..20_000 {
      let p = Point::from(
        Vector::<f64, D>::from_fn(|i, _| rng.random_range(0.0..1.0) * dom.size()[i]));
      let truth = balls.iter()
        .fold(walls(p), |acc, &(c, r)| acc.min((p - c).length() - r));
      assert!((field.sdf(p) - truth).abs() < 1e-12,
        "{} on {extents:?}: {} against a true {truth} at {p:?}",
        L::NAME, field.sdf(p));
    }
  }
  for extents in [[1.0, 1.0, 1.0], [1.0, 0.5, 0.25], [1.0, 0.1, 0.02]] {
    check::<Kd>(extents);
    check::<WeightedKd>(extents);
    check::<tree::Orthant>(extents);
  }
}
