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
