//! SDF shape primitives, grouped into submodules by the number of dimensions
//! they support.
//!
//! - [`dn`] — any dimension count: [`Hypersphere`], [`Hyperrect`],
//!   [`Hypersquare`], [`Line`], [`Ring`], [`Moon`], [`Kakera`], [`Cross`],
//!   [`Polytope`].
//! - [`d2`] — the plane only, by construction rather than by omission: the
//!   regular-polygon and star families fold `p` by its polar angle, and
//!   [`Polygon`] signs its distance by planar winding parity.
//!
//! Everything is re-exported here, so `geometry::Hypersphere` and
//! `geometry::shapes::dn::Hypersphere` name the same type — import from this
//! module unless you deliberately want to restrict yourself to one tier. A
//! further tier (`d3`, …) joins the list if a shape ever caps out at that
//! dimension; there is none today, since every shape here that generalizes at
//! all generalizes the whole way.

pub mod d2;
pub mod dn;

pub use d2::*;
pub use dn::*;

use num_traits::Float;

pub(crate) fn clamp<T: Float>(mut x: T, min: T, max: T) -> T {
  if x < min { x = min; }
  if x > max { x = max; }
  x
}

#[cfg(test)]
mod tests {
  use {
    super::*,
    crate::{
      geometry::{Point, Shape, Vector, VectorExt, P2, V2},
      sdf::{self, Lipschitz, SDF},
    },
    nalgebra::Rotation2,
    rand::prelude::*,
  };

  /// Numerically verify `|f(p) − f(q)| ≤ L·|p − q|` over random global and
  /// tightly-spaced point pairs (the latter stress the local gradient), plus
  /// finiteness everywhere. Every stored primitive relies on an honest
  /// Lipschitz bound: the redundancy test certifies with it (`sdf_geq_everywhere`),
  /// and the D*-pruned insertion walk skips subtrees with it.
  fn check_lipschitz<const D: usize>(name: &str, l: f64, f: impl Fn(Point<f64, D>) -> f64) {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let span = 2.5;
    let rand_pt = |rng: &mut rand_pcg::Pcg64| Point::from(
      Vector::<f64, D>::from_fn(|_, _| rng.random_range(-span..span)));
    for i in 0..20000 {
      let p = rand_pt(&mut rng);
      let q = if i % 2 == 0 {
        rand_pt(&mut rng)
      } else {
        // short-range pair: within 1e-4..1e-2 of p
        let dir = Vector::<f64, D>::from_fn(|_, _| rng.random_range(-1.0..1.0f64))
          .robust_normalize();
        p + dir * 10f64.powf(rng.random_range(-4.0..-2.0))
      };
      let (fp, fq) = (f(p), f(q));
      assert!(fp.is_finite() && fq.is_finite(), "{name}: non-finite field at {p:?} / {q:?}");
      let dist = (p - q).length();
      assert!(
        (fp - fq).abs() <= l * dist * (1.0 + 1e-9) + 1e-12,
        "{name} (D={D}): Lipschitz bound {l} violated: |f({p:?}) − f({q:?})| = {} > {l}·{dist}",
        (fp - fq).abs()
      );
    }
  }

  /// [`check_lipschitz`] pinned to the plane: the shapes with a defaulted `D`
  /// only resolve it from the point type, which a generic call cannot supply.
  fn check_2d(name: &str, l: f64, f: impl Fn(P2<f64>) -> f64) {
    check_lipschitz::<2>(name, l, f)
  }

  /// Random points over `[-span, span]^D`, for the equivalence/exactness checks.
  fn sample<const D: usize>(n: usize, span: f64, mut body: impl FnMut(Point<f64, D>)) {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
    for _ in 0..n {
      body(Point::from(Vector::<f64, D>::from_fn(|_, _| rng.random_range(-span..span))));
    }
  }

  #[test] fn lipschitz_shapes() {
    check_2d("Hypersphere", 1.0, |p| Hypersphere.sdf(p));
    check_2d("Hyperrect", 1.0, |p| Hyperrect { size: V2::new(1.5, 0.8) }.sdf(p));
    check_2d("Hypersquare", 1.0, |p| Hypersquare.sdf(p));
    check_2d("Line", 1.0, |p| Line
      { a: P2::new(-0.8, -0.3), b: P2::new(0.6, 0.9), thickness: 0.2 }.sdf(p));
    check_2d("Triangle", 1.0, |p| NGonC::<3>.sdf(p));
    check_2d("Pentagon", 1.0, |p| NGonC::<5>.sdf(p));
    check_2d("Heptagon", 1.0, |p| NGonC::<7>.sdf(p));
    check_2d("NGonR(5)", 1.0, |p| NGonR { n: 5 }.sdf(p));
    check_2d("Star(5, 10/3)", 1.0, |p| Star { n: 5, m: 10.0 / 3.0 }.sdf(p));
    check_2d("Star(7, 3)", 1.0, |p| Star { n: 7, m: 3.0 }.sdf(p));
    check_2d("Pentagram", 1.0, |p| Pentagram.sdf(p));
    check_2d("Hexagram", 1.0, |p| Hexagram.sdf(p));
    for phase in [-1.0, -0.7, 0.0, 0.3, 1.0] {
      check_2d(&format!("Moon({phase})"), 1.0, |p| Moon { phase }.sdf(p));
    }
    check_2d("Kakera", 1.0, |p| Kakera { width: 0.5 }.sdf(p));
    check_2d("Cross", 1.0, |p| Cross { thickness: 0.3 }.sdf(p));
    check_2d("Ring", 1.0, |p| Ring { inner_r: 0.5 }.sdf(p));
    check_2d("Polygon", 1.0, |p| Polygon { vertices: [
      P2::new(-0.9, -0.5), P2::new(0.8, -0.7),
      P2::new(0.5, 0.9), P2::new(-0.3, 0.4),
    ]}.sdf(p));
    check_2d("HolyCross", 1.0, |p| HolyCross.sdf(p));
    check_2d("boundary_rect", 1.0, sdf::boundary_rect::<f64, 2>);
  }

  /// The same honesty check for the [`dn`] tier above the plane — an
  /// understated constant here would corrupt the ADF's pruning in 3D.
  #[test] fn lipschitz_shapes_nd() {
    check_lipschitz("Hypersphere", 1.0, |p| Hypersphere::<3>.sdf(p));
    check_lipschitz("Hypersphere", 1.0, |p| Hypersphere::<4>.sdf(p));
    check_lipschitz("Hyperrect", 1.0, |p| Hyperrect { size: Vector::from([1.5, 0.8, 2.0]) }.sdf(p));
    check_lipschitz("Hyperrect", 1.0, |p| Hyperrect { size: Vector::from([1.0, 2.0, 0.5, 1.2]) }.sdf(p));
    check_lipschitz("Hypersquare", 1.0, |p| Hypersquare::<3>.sdf(p));
    check_lipschitz("Hypersquare", 1.0, |p| Hypersquare::<4>.sdf(p));
    check_lipschitz("Line", 1.0, |p| Line {
      a: Point::from([-0.8, -0.3, 0.2]), b: Point::from([0.6, 0.9, -0.4]), thickness: 0.2,
    }.sdf(p));
    check_lipschitz("Ring", 1.0, |p| Ring::<f64, 3> { inner_r: 0.5 }.sdf(p));
    check_lipschitz("Ring", 1.0, |p| Ring::<f64, 4> { inner_r: 0.5 }.sdf(p));
    for phase in [-1.0, -0.7, 0.0, 0.3, 1.0] {
      check_lipschitz(&format!("Moon({phase})"), 1.0,
        |p: Point<f64, 3>| Moon { phase }.sdf(p));
      check_lipschitz(&format!("Moon({phase})"), 1.0,
        |p: Point<f64, 4>| Moon { phase }.sdf(p));
    }
    for width in [0.3, 0.5, 1.0, 1.7] {
      check_lipschitz(&format!("Kakera({width})"), 1.0,
        |p: Point<f64, 3>| Kakera { width }.sdf(p));
      check_lipschitz(&format!("Kakera({width})"), 1.0,
        |p: Point<f64, 4>| Kakera { width }.sdf(p));
    }
    for thickness in [0.05, 0.3, 0.8, 1.0] {
      check_lipschitz(&format!("Cross({thickness})"), 1.0,
        |p: Point<f64, 3>| Cross { thickness }.sdf(p));
      check_lipschitz(&format!("Cross({thickness})"), 1.0,
        |p: Point<f64, 5>| Cross { thickness }.sdf(p));
    }
    check_lipschitz("Polytope", 1.0, |p: Point<f64, 3>| cube_polytope().sdf(p));
    check_lipschitz("boundary_rect", 1.0, sdf::boundary_rect::<f64, 3>);
    check_lipschitz("boundary_rect", 1.0, sdf::boundary_rect::<f64, 5>);
    // combinators, one dimension up
    check_lipschitz("union 3D", 1.0, |p: Point<f64, 3>| Hypersphere::<3>
      .translate(Vector::from([0.4, 0.0, 0.0]))
      .union(Hypersquare::<3>.scale(0.6)).sdf(p));
    check_lipschitz("smooth_min 3D", 1.0, |p: Point<f64, 3>| Hypersphere::<3>
      .smooth_min(Hypersquare::<3>.scale(0.5), 32.0).sdf(p));
    check_lipschitz("rotate 3D", 1.0, |p: Point<f64, 3>|
      Hyperrect { size: Vector::from([1.5, 0.8, 2.0]) }
        .rotate(nalgebra::Rotation3::from_euler_angles(0.3, 0.6, -0.2)).sdf(p));
  }

  // ---- the generalized shapes must not have moved in the plane ----
  //
  // Reference implementations, transcribed from the 2D-only versions these
  // three replaced. `Moon` and `Kakera` now reduce through `revolve`, and
  // `Cross` computes its fold over the two largest coordinates instead of a
  // swap; at D = 2 all three must agree with the originals bit-for-bit-close.

  fn moon_ref(phase: f64, p: P2<f64>) -> f64 {
    let pixel = V2::new(p.x, p.y.abs());
    let d = phase * 2.0;
    let a = d / 2.0;
    let b = (1.0 - a * a).max(0.0).sqrt();
    if d * (pixel.x * b - pixel.y * a) > d * d * (b - pixel.y).max(0.0) {
      (pixel - V2::new(a, b)).length()
    } else {
      (pixel.length() - 1.0).max(-((pixel - V2::new(d, 0.0)).length() - 1.0))
    }
  }

  fn kakera_ref(width: f64, p: P2<f64>) -> f64 {
    let ndot = |a: V2<f64>, b: V2<f64>| a.x * b.x - a.y * b.y;
    let b = V2::new(width, 1.0);
    let q = p.coords.abs();
    let h = ((-2.0 * ndot(q, b) + ndot(b, b)) / b.dot(&b)).clamp(-1.0, 1.0);
    let d = (q - V2::new(1.0 - h, 1.0 + h).component_mul(&b) / 2.0).length();
    d * (q.x * b.y + q.y * b.x - b.x * b.y).signum()
  }

  fn cross_ref(thickness: f64, p: P2<f64>) -> f64 {
    let mut pixel = p.coords.abs();
    pixel = if pixel.y > pixel.x { V2::new(pixel.y, pixel.x) } else { pixel };
    let q = pixel - V2::new(1.0, thickness);
    let k = q.x.max(q.y);
    let w = if k > 0.0 { q } else { V2::new(thickness - pixel.x, -k) };
    k.signum() * w.map(|x| x.max(0.0)).length()
  }

  #[test] fn generalized_shapes_unchanged_in_2d() {
    for phase in [-1.0, -0.7, -0.2, 0.0, 0.3, 0.9, 1.0] {
      sample::<2>(20000, 2.5, |p| {
        let (new, old) = (Moon { phase }.sdf(p), moon_ref(phase, p));
        assert!((new - old).abs() < 1e-12, "Moon({phase}) at {p:?}: {new} vs {old}");
      });
    }
    for width in [0.2, 0.5, 1.0, 1.7] {
      sample::<2>(20000, 2.5, |p| {
        let (new, old) = (Kakera { width }.sdf(p), kakera_ref(width, p));
        assert!((new - old).abs() < 1e-12, "Kakera({width}) at {p:?}: {new} vs {old}");
      });
    }
    // `cross_ref` (Quílez's `sdCross`) is only valid for arms at most as thick
    // as they are long — inside the central cube it pairs `t − q₁` with
    // `min(1 − q₁, t − q₂)`, and once `t > 1/2` the latter can select the arm
    // cap `1 − q₁`, combining two lengths measured along the *same* axis. Below
    // that threshold `t − q₂ ≤ t ≤ 1 − t ≤ 1 − q₁` always holds, so the two
    // agree; `cross_exact_in_nd` covers the thick regime, where only the new
    // formula is right.
    for thickness in [0.05, 0.3, 0.5] {
      sample::<2>(20000, 2.5, |p| {
        let (new, old) = (Cross { thickness }.sdf(p), cross_ref(thickness, p));
        assert!((new - old).abs() < 1e-12, "Cross({thickness}) at {p:?}: {new} vs {old}");
      });
    }
    // and the divergence past the threshold is real, not a tolerance artifact:
    // from here the cap at x = 1 is nearer than the reflex ridge at (0.7, 0.7)
    let p = P2::new(-0.041533680486002034, 0.45933037512186825);
    assert!((Cross { thickness: 0.7 }.sdf(p) + 0.5406696248781317).abs() < 1e-12);
    assert!(cross_ref(0.7, p) < -0.59); // the old formula over-reports the depth
  }

  /// Outside a union of convex sets, the exact distance is the `min` of the
  /// exact per-set distances — so the cross's exterior can be checked against
  /// its `D` arm boxes directly. (Inside, that `min` only underestimates the
  /// depth, which is precisely why `Cross` does not use it.)
  fn cross_exterior_ref<const D: usize>(t: f64, p: Point<f64, D>) -> f64 {
    (0..D).map(|arm| Hyperrect {
      size: Vector::<f64, D>::from_fn(|a, _| if a == arm { 2.0 } else { 2.0 * t })
    }.sdf(p)).fold(f64::MAX, f64::min)
  }

  /// Inside, the distance to the complement, by explicit enumeration of every
  /// way out: past any single arm cap, or past any *pair* of axes both leaving
  /// the core. Independent of `Cross`'s shortcut of only tracking the two
  /// largest coordinates.
  fn cross_interior_ref<const D: usize>(t: f64, p: Point<f64, D>) -> f64 {
    let mut best = (0..D).fold(f64::MAX, |acc, a| acc.min(1.0 - p[a].abs()));
    for i in 0..D {
      for j in i + 1..D {
        let (a, b) = ((t - p[i].abs()).max(0.0), (t - p[j].abs()).max(0.0));
        best = best.min((a * a + b * b).sqrt());
      }
    }
    best
  }

  fn cross_exact<const D: usize>(t: f64) {
    let check = |p: Point<f64, D>| {
      let got = Cross { thickness: t }.sdf(p);
      let want = if got > 0.0 { cross_exterior_ref(t, p) } else { -cross_interior_ref(t, p) };
      assert!((got - want).abs() < 1e-12,
        "Cross({t}) D={D} at {p:?}: {got} vs {want}");
      got
    };
    // uniform over a box that comfortably contains the cross — mostly exterior,
    // increasingly so as the arms thin out
    let mut outside = 0;
    sample::<D>(20000, 1.8, |p| if check(p) > 0.0 { outside += 1 });
    // interior by construction, so coverage does not depend on `t` or `D`: one
    // long axis anywhere in [-1, 1], every other inside the core
    let mut rng = rand_pcg::Pcg64::seed_from_u64(11);
    let mut inside = 0;
    for _ in 0..4000 {
      let arm = rng.random_range(0..D);
      let p = Point::from(Vector::<f64, D>::from_fn(|a, _| if a == arm {
        rng.random_range(-1.0..1.0)
      } else {
        rng.random_range(-t..t)
      }));
      if check(p) <= 0.0 { inside += 1 }
    }
    assert!(inside > 3500 && outside > 2000, "D={D} t={t}: {inside} in, {outside} out");
  }

  #[test] fn cross_exact_in_nd() {
    for t in [0.05, 0.3, 0.5, 0.7, 1.0] {
      cross_exact::<2>(t);
      cross_exact::<3>(t);
      cross_exact::<4>(t);
    }
    // anchors: the origin's nearest exit is the reflex ridge at (t, t, ·);
    // (0.5, 0.5, 0.5) must bring its two largest axes down to t
    let t = 0.3;
    assert!((Cross { thickness: t }.sdf(Point::from([0.0; 3])) + (2.0 * t * t).sqrt()).abs() < 1e-12);
    assert!((Cross { thickness: t }.sdf(Point::from([0.5, 0.5, 0.5])) - (2.0 * 0.04f64).sqrt()).abs() < 1e-12);
  }

  /// `Moon` and `Kakera` in 3D are solids of revolution: their field must equal
  /// the 2D generator's, evaluated at `(x, |(y, z)|)`.
  #[test] fn revolution_matches_generator_in_3d() {
    for phase in [-0.7, 0.0, 0.4, 1.0] {
      sample::<3>(20000, 2.0, |p| {
        let gen2 = P2::new(p.x, (p.y * p.y + p.z * p.z).sqrt());
        let (got, want) = (Moon { phase }.sdf(p), moon_ref(phase, gen2));
        assert!((got - want).abs() < 1e-12, "Moon 3D at {p:?}: {got} vs {want}");
      });
    }
    for width in [0.3, 1.0, 1.7] {
      sample::<3>(20000, 2.0, |p| {
        let gen2 = P2::new(p.x, (p.y * p.y + p.z * p.z).sqrt());
        let (got, want) = (Kakera { width }.sdf(p), kakera_ref(width, gen2));
        assert!((got - want).abs() < 1e-12, "Kakera 3D at {p:?}: {got} vs {want}");
      });
    }
    // the 3D bicone reaches ±width along x and radius 1 across it
    let k = Kakera { width: 0.5 };
    assert!(SDF::<f64, 3>::sdf(&k, Point::from([0.5, 0.0, 0.0])).abs() < 1e-12);
    assert!(SDF::<f64, 3>::sdf(&k, Point::from([0.0, 0.0, 1.0])).abs() < 1e-12);
  }

  /// The unit cube as an intersection of its six face half-spaces.
  fn cube_polytope() -> Polytope<[HalfSpace<f64, 3>; 6]> {
    let mut hs = [HalfSpace::new(Vector::from([1.0, 0.0, 0.0]), 1.0); 6];
    for a in 0..3 {
      let mut n = Vector::<f64, 3>::zeros();
      n[a] = 1.0;
      hs[2 * a] = HalfSpace::new(n, 1.0);
      hs[2 * a + 1] = HalfSpace::new(n.map(|x| -x), 1.0);
    }
    Polytope { half_spaces: hs }
  }

  #[test] fn polytope_is_exact_inside_conservative_outside() {
    let cube = cube_polytope();
    let reference = Hypersquare::<3>;
    let (mut inside, mut outside) = (0, 0);
    sample::<3>(40000, 2.0, |p| {
      let (got, want) = (cube.sdf(p), reference.sdf(p));
      if want <= 0.0 {
        // interior: the nearest boundary point lies on a face plane, so the
        // half-space max is the exact distance
        assert!((got - want).abs() < 1e-12, "Polytope interior at {p:?}: {got} vs {want}");
        inside += 1;
      } else {
        // exterior: conservative — never claims to be farther than it is
        assert!(got <= want + 1e-12, "Polytope exterior at {p:?}: {got} > {want}");
        outside += 1;
      }
    });
    assert!(inside > 1000 && outside > 1000, "coverage: {inside} in, {outside} out");

    // the documented underestimate: at a corner it reports the face-plane
    // distance (1), not the true √3
    let corner = Point::from([2.0, 2.0, 2.0]);
    assert!((cube.sdf(corner) - 1.0).abs() < 1e-12);
    assert!((reference.sdf(corner) - 3f64.sqrt()).abs() < 1e-12);

    // an empty half-space list is "no shape", as with `Polygon`
    let empty: Polytope<[HalfSpace<f64, 3>; 0]> = Polytope { half_spaces: [] };
    assert_eq!(empty.sdf(Point::from([0.0; 3])), f64::MAX / 2.0);
  }

  #[test] fn lipschitz_combinators() {
    let star = || Star { n: 5, m: 10.0 / 3.0 };
    // translate / rotate / scale: precomposition with an isometry (or a
    // similarity whose value re-scale cancels the coordinate re-scale)
    // preserves the constant exactly
    check_2d("translate", 1.0, |p| star().translate(V2::new(0.3, -0.2)).sdf(p));
    check_2d("rotate", 1.0, |p| star().rotate(Rotation2::new(37f64.to_radians())).sdf(p));
    check_2d("scale(0.35)", 1.0, |p| star().scale(0.35).sdf(p));
    check_2d("scale(2.5)", 1.0, |p| star().scale(2.5).sdf(p));
    // boolean ops: min/max of L-Lipschitz fields is max(L₁, L₂)-Lipschitz
    check_2d("union", 1.0, |p| Hypersphere.translate(V2::new(0.4, 0.0))
      .union(Hypersquare.scale(0.6)).sdf(p));
    check_2d("subtraction", 1.0, |p| Shape::<f64, 2>::subtraction(
      Hypersquare, Hypersphere.scale(0.7).translate(V2::new(0.5, 0.5))).sdf(p));
    check_2d("intersection", 1.0, |p| Shape::<f64, 2>::intersection(
      Hypersphere, Hypersquare.rotate(Rotation2::new(20f64.to_radians()))).sdf(p));
    // smooth_min: ∇ = w·∇f + (1−w)·∇g with w ∈ (0, 1) — a convex combination
    check_2d("smooth_min", 1.0, |p| Hypersphere.translate(V2::new(-0.4, 0.1))
      .smooth_min(Hypersquare.scale(0.5).translate(V2::new(0.5, 0.0)), 32.0).sdf(p));
  }

  // the trait-derived constants agree with the numerically-verified bounds
  #[test] fn lipschitz_trait() {
    // a bare `Lipschitz` call is dimension-independent, so `D` must be named
    assert_eq!(Lipschitz::<f64>::lipschitz(&Hypersphere::<2>), 1.0);
    assert_eq!(Lipschitz::<f64>::lipschitz(&Hypersphere::<3>), 1.0);
    assert_eq!(Lipschitz::<f64>::lipschitz(&HolyCross), 1.0);
    let chain = Star { n: 5, m: 10.0 / 3.0 }
      .scale(0.35)
      .rotate(Rotation2::new(37f64.to_radians()))
      .translate(V2::new(0.3, -0.2));
    assert_eq!(chain.lipschitz(), 1.0);
    let boolean = Shape::<f64, 2>::union(
      Hypersphere, Shape::<f64, 2>::subtraction(Hypersquare, Ring::<f64, 2> { inner_r: 0.5 }));
    assert_eq!(boolean.lipschitz(), 1.0);
    // and one dimension up, through the N-D shapes
    let nd = Shape::<f64, 3>::union(
      Hypersphere::<3>.scale(0.5),
      Shape::<f64, 3>::subtraction(Cross { thickness: 0.2 }, Kakera { width: 0.4 }));
    assert_eq!(nd.lipschitz(), 1.0);
  }
}
