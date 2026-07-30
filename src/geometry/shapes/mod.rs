//! SDF shape primitives, grouped into submodules by the number of dimensions
//! they support.
//!
//! - [`dn`] — any dimension count. The unit shapes ([`Hypersphere`],
//!   [`Hyperrect`], [`Hypersquare`], [`Orthoplex`], [`LpBall`]), the swept and
//!   revolved ones ([`Line`], [`Polyline`], [`Ring`], [`Torus`], [`Moon`],
//!   [`Kakera`]), [`Cross`], the product family [`ProductBall`], the convex
//!   [`Polytope`] with its [`simplex`] and [`permutohedron`] constructors, and
//!   the periodic [`Gyroid`].
//! - [`d2`] — the plane only, by construction rather than by omission: the
//!   regular-polygon and star families fold `p` by its polar angle, and
//!   [`Polygon`] signs its distance by planar winding parity.
//! - [`d3`] — the two Platonic solids with no analogue elsewhere
//!   ([`dodecahedron`], [`icosahedron`]), the two polyhedra that tile 3-space
//!   alone ([`truncated_octahedron`], [`rhombic_dodecahedron`]), and
//!   [`torus_knot`].
//! - [`d4`] — the three exceptional regular 4-polytopes ([`cell_24`],
//!   [`cell_120`], [`cell_600`]) and the shapes needing a 2 + 2 split of the
//!   axes ([`duocylinder`], [`CliffordTorus`]).
//!
//! Everything is re-exported here, so `geometry::Hypersphere` and
//! `geometry::shapes::dn::Hypersphere` name the same type — import from this
//! module unless you deliberately want to restrict yourself to one tier.
//!
//! # Building shapes out of shapes
//!
//! The catalogue above is deliberately small, because most interesting shapes
//! are compositions. Beyond the transforms and booleans on [`Shape`](crate::geometry::Shape), four
//! combinators change a shape's *character*, and all four are exact and
//! Lipschitz-preserving:
//!
//! | Combinator | Field | Effect |
//! |---|---|---|
//! | [`shell`](crate::geometry::Combinator::shell) | <code>&#124;sdf&#124; - w</code> | solid → hollow surface |
//! | [`offset`](crate::geometry::Combinator::offset) | `sdf - r` | grows and rounds every corner |
//! | [`extrude`](crate::geometry::Combinator::extrude) | box of `(sdf, &#124;p_last&#124; - h)` | lifts `D-1` → `D` |
//! | [`revolve`](crate::geometry::Combinator::revolve) | `sdf(p₀, ‖p⊥‖ - r)` | sweeps a 2D profile around axis 0 |
//!
//! The last two are dimension *lifts*, which is what makes the plane-only tier
//! useful in space: every star, polygon and n-gon becomes either a prism or a
//! ring of that cross-section.
//!
//! ```
//! # use space_filling::{geometry::*, sdf::SDF};
//! // a star-sectioned torus: 2D profile, held off the axis, swept around it
//! let star_ring = Pentagram.scale(0.25f64).revolve(0.7);
//!
//! // a hollow icosahedron frame — rounded so the field is exact everywhere
//! let frame = icosahedron::<f64>().offset(0.02).shell(0.01);
//!
//! // a gyroid labyrinth: a minimal surface, thickened, clipped to a container
//! let labyrinth = Shape::<f64, 3>::intersection(
//!   Hypersquare::<3>, Gyroid { frequency: 9.0 }.shell(0.03));
//!
//! // the shell of a cube minus an inscribed ball: a cube's edges, as it were
//! let cage = Shape::<f64, 3>::subtraction(
//!   Hypersquare::<3>.shell(0.05), Hypersphere::<3>.scale(0.95));
//!
//! # for s in [&star_ring as &dyn SDF<f64, 3>, &frame, &labyrinth, &cage] {
//! #   assert!(s.sdf(Point::from([0.31, -0.22, 0.13])).is_finite());
//! # }
//! ```
//!
//! (Note the last two. A [`Shape`](crate::geometry::Shape) combinator whose return type mentions neither
//! the scalar nor `D` — `union`, `subtraction`, `intersection`, `smooth_min`,
//! `scale` — infers both from its *receiver*, and a wrapper such as `shell(..)`
//! supplies neither; hand it to `Shape::<f64, 3>::subtraction` instead of making
//! it the receiver. The four combinators in the table never have this problem,
//! which is exactly why they live on the dimension-free [`Combinator`](crate::geometry::Combinator).)
//!
//! Two shapes are worth calling out as *sources* rather than results.
//! [`Polytope`] takes any set of half-spaces, so a polytope you can tabulate is
//! a polytope you can render — every constructor in [`d3`] and [`d4`] is one
//! call to [`convex_hull`] over a vertex orbit. And [`Gyroid`] is not a distance
//! field at all but a level set, normalized by its own gradient bound so that
//! the result is still an honest 1-Lipschitz underestimate: the pattern to copy
//! whenever you want an implicit surface the solvers can actually chew on.

pub mod d2;
pub mod d3;
pub mod d4;
pub mod dn;

pub use d2::*;
pub use d3::*;
pub use d4::*;
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
      geometry::{Combinator, Point, Shape, Vector, VectorExt, P2, V2},
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
    // a `D`-generic receiver cannot infer `D` for a `Shape<T, D>` method whose
    // return type does not mention it, hence the turbofish
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
    assert_eq!(Lipschitz::<f64>::lipschitz(&boolean), 1.0);
    // and one dimension up, through the N-D shapes
    let nd = Shape::<f64, 3>::union(
      Hypersphere::<3>.scale(0.5),
      Shape::<f64, 3>::subtraction(Cross { thickness: 0.2 }, Kakera { width: 0.4 }));
    assert_eq!(Lipschitz::<f64>::lipschitz(&nd), 1.0);
  }

  // ---- the four combinators ----------------------------------------------

  /// `extrude` of a disc must be a cylinder — and [`ProductBall`] computes that
  /// same cylinder by a completely different route (per-block radial clamping),
  /// so agreement pins both down. Both claim exactness, so they must agree
  /// everywhere, inside and out.
  #[test] fn extrude_matches_product_ball() {
    let (r, h) = (0.6, 0.35);
    let cylinder = Hypersphere.scale(r).extrude(h);
    let reference = ProductBall { spec: [(2, r), (1, h)] };
    sample::<3>(20000, 1.6, |p| {
      let (got, want) = (cylinder.sdf(p), reference.sdf(p));
      assert!((got - want).abs() < 1e-12, "cylinder at {p:?}: {got} vs {want}");
    });
    // and the extrusion of a square is just a box
    let bar = Hypersquare.scale(0.4).extrude(0.7);
    let boxed = Hyperrect { size: Vector::from([0.8, 0.8, 1.4]) };
    sample::<3>(20000, 1.6, |p| {
      let (got, want) = (bar.sdf(p), boxed.sdf(p));
      assert!((got - want).abs() < 1e-12, "bar at {p:?}: {got} vs {want}");
    });
  }

  /// `revolve` of a disc must be a torus, which [`Torus`] states in closed form.
  #[test] fn revolve_matches_torus() {
    let (major, minor) = (0.7, 0.25);
    let swept = Hypersphere.scale(minor).revolve(major);
    let reference = Torus { major, minor };
    for d in 0..2 {
      // check in 3D and 4D: the revolution is defined for any perpendicular span
      if d == 0 {
        sample::<3>(20000, 1.6, |p| {
          let (got, want) = (swept.sdf(p), reference.sdf(p));
          assert!((got - want).abs() < 1e-12, "torus 3D at {p:?}: {got} vs {want}");
        });
      } else {
        sample::<4>(20000, 1.6, |p| {
          let (got, want) = (swept.sdf(p), reference.sdf(p));
          assert!((got - want).abs() < 1e-12, "torus 4D at {p:?}: {got} vs {want}");
        });
      }
    }
    // a torus in the plane degenerates to the two discs at (0, ±major)
    assert!((reference.sdf(P2::new(0.0, major)) + minor).abs() < 1e-12);
    assert!((reference.sdf(P2::new(0.0, -major)) + minor).abs() < 1e-12);
  }

  /// The offset of a convex body is exactly its field minus the radius, so a
  /// rounded box's surface must sit exactly `r` outside the box's.
  #[test] fn offset_and_shell_algebra() {
    let base = Hyperrect { size: V2::new(1.2, 0.7) };
    let (r, w) = (0.15, 0.05);
    sample::<2>(20000, 2.0, |p| {
      assert!((base.offset(r).sdf(p) - (base.sdf(p) - r)).abs() < 1e-15);
      assert!((base.shell(w).sdf(p) - (base.sdf(p).abs() - w)).abs() < 1e-15);
      // a shell is the subtraction of the eroded body from the dilated one
      let by_hand = base.offset(w).sdf(p).max(-base.offset(-w).sdf(p));
      assert!((base.shell(w).sdf(p) - by_hand).abs() < 1e-12);
    });
    // the hollow interior really is outside the shell
    assert!(Hypersquare.scale(0.5).shell(0.02).sdf(P2::new(0.0, 0.0)) > 0.0);
  }

  #[test] fn lipschitz_combinator_lifts() {
    check_2d("shell", 1.0, |p| Star { n: 5, m: 10.0 / 3.0 }.shell(0.05).sdf(p));
    check_2d("offset", 1.0, |p| Cross { thickness: 0.2 }.offset(0.05).sdf(p));
    check_lipschitz("extrude(pentagram)", 1.0,
      |p: Point<f64, 3>| Pentagram.extrude(0.3).sdf(p));
    check_lipschitz("extrude(extrude)", 1.0,
      |p: Point<f64, 4>| Hexagram.extrude(0.3).extrude(0.2).sdf(p));
    check_lipschitz("revolve(pentagram)", 1.0,
      |p: Point<f64, 3>| Pentagram.scale(0.25).revolve(0.7).sdf(p));
    check_lipschitz("revolve(moon)", 1.0,
      |p: Point<f64, 3>| Moon { phase: 0.4 }.revolve(1.6).sdf(p));
    check_lipschitz("shell(gyroid)", 1.0,
      |p: Point<f64, 3>| Gyroid { frequency: 6.0 }.shell(0.05).sdf(p));
  }

  // ---- the new N-dimensional shapes --------------------------------------

  /// Each of these is an independent route to a shape the crate already has.
  #[test] fn nd_shapes_agree_with_their_special_cases() {
    // ℓ² ball == hypersphere
    sample::<3>(20000, 2.0, |p| {
      let (got, want) = (LpBall::<f64, 3> { p: 2.0 }.sdf(p), Hypersphere::<3>.sdf(p));
      assert!((got - want).abs() < 1e-12, "LpBall(2) at {p:?}: {got} vs {want}");
    });
    // ℓ¹ ball == orthoplex, up to the facet normalization: both are the max over
    // the same planes, so they agree exactly inside and on the boundary
    sample::<3>(20000, 1.2, |p| {
      let (l1, ortho) = (LpBall::<f64, 3> { p: 1.0 }.sdf(p), Orthoplex::<3>.sdf(p));
      assert_eq!(l1 <= 0.0, ortho <= 0.0, "sign disagreement at {p:?}");
    });
    // a two-point polyline == a capsule
    let (a, b) = (Point::from([-0.6, 0.2, 0.1]), Point::from([0.5, -0.3, 0.4]));
    sample::<3>(20000, 1.6, |p| {
      let got = Polyline { vertices: [a, b], thickness: 0.2 }.sdf(p);
      let want = Line { a, b, thickness: 0.2 }.sdf(p);
      assert!((got - want).abs() < 1e-12, "polyline at {p:?}: {got} vs {want}");
    });
    // an all-singleton product of balls == a box
    let size = Vector::from([1.2, 0.6, 1.8]);
    let product = ProductBall { spec: [(1, 0.6), (1, 0.3), (1, 0.9)] };
    sample::<3>(20000, 2.0, |p| {
      let (got, want) = (product.sdf(p), Hyperrect { size }.sdf(p));
      assert!((got - want).abs() < 1e-12, "product box at {p:?}: {got} vs {want}");
    });
  }

  #[test] fn orthoplex_and_lp_ball_anchors() {
    // vertices sit on the boundary, the centre `1/√D` deep
    for a in 0..3 {
      let v = Point::from(Vector::<f64, 3>::from_fn(|k, _| if k == a { 1.0 } else { 0.0 }));
      assert!(Orthoplex::<3>.sdf(v).abs() < 1e-12);
    }
    assert!((Orthoplex::<3>.sdf(Point::from([0.0; 3])) + 1.0 / 3f64.sqrt()).abs() < 1e-12);
    // the ℓᵖ family grows monotonically toward the cube as p rises
    let corner = Point::from([0.9, 0.9, 0.9]);
    let mut previous = f64::MAX;
    for p in [1.0, 1.5, 2.0, 4.0, 8.0, 16.0] {
      let d = LpBall::<f64, 3> { p }.sdf(corner);
      assert!(d < previous, "LpBall({p}) not monotone at the corner");
      previous = d;
    }
    // and the declared constant is honest for every p, including p < 2
    for p in [1.0, 1.25, 1.5, 2.0, 4.0] {
      let ball = LpBall::<f64, 3> { p };
      let l = Lipschitz::<f64>::lipschitz(&ball);
      assert!(l >= 1.0);
      check_lipschitz(&format!("LpBall({p})"), l, |q: Point<f64, 3>| ball.sdf(q));
    }
  }

  #[test] fn lipschitz_new_nd_shapes() {
    check_lipschitz("Orthoplex", 1.0, |p: Point<f64, 3>| Orthoplex::<3>.sdf(p));
    check_lipschitz("Orthoplex", 1.0, |p: Point<f64, 4>| Orthoplex::<4>.sdf(p));
    check_lipschitz("Torus", 1.0,
      |p: Point<f64, 3>| Torus { major: 0.7, minor: 0.25 }.sdf(p));
    check_lipschitz("ProductBall", 1.0,
      |p: Point<f64, 4>| ProductBall { spec: [(2, 0.5), (2, 0.3)] }.sdf(p));
    check_lipschitz("Polyline", 1.0, |p: Point<f64, 3>| Polyline {
      vertices: [
        Point::from([-0.6, 0.2, 0.1]), Point::from([0.5, -0.3, 0.4]),
        Point::from([0.1, 0.7, -0.5]), Point::from([-0.4, -0.4, 0.6]),
      ],
      thickness: 0.15,
    }.sdf(p));
    check_lipschitz("simplex", 1.0, |p: Point<f64, 3>| simplex::<f64, 3>().sdf(p));
    check_lipschitz("permutohedron", 1.0, |p: Point<f64, 3>| permutohedron::<f64, 3>().sdf(p));
    check_lipschitz("CliffordTorus", 1.0, |p: Point<f64, 4>|
      CliffordTorus { r1: 0.5, r2: 0.5, thickness: 0.2 }.sdf(p));
  }

  /// The gyroid is the one field here that is not a distance function — it is a
  /// level set divided by its own gradient bound. That bound is the whole basis
  /// of its soundness, so it gets checked hard, at several frequencies and
  /// dimensions.
  #[test] fn gyroid_is_honestly_1_lipschitz() {
    for frequency in [1.0, 4.0, 9.0, 20.0] {
      check_lipschitz(&format!("Gyroid({frequency}) 3D"), 1.0,
        |p: Point<f64, 3>| Gyroid { frequency }.sdf(p));
      check_lipschitz(&format!("Gyroid({frequency}) 4D"), 1.0,
        |p: Point<f64, 4>| Gyroid { frequency }.sdf(p));
    }
    // it really does separate space into two labyrinths: both signs occur, and
    // the zero set is dense enough that a coarse walk finds crossings
    let g = Gyroid { frequency: 8.0 };
    let (mut pos, mut neg) = (0, 0);
    sample::<3>(4000, 1.0, |p| if g.sdf(p) > 0.0 { pos += 1 } else { neg += 1 });
    assert!(pos > 1000 && neg > 1000, "gyroid is lopsided: {pos} / {neg}");
  }

  // ---- polytopes: geometry, not just Lipschitz ---------------------------

  /// Every vertex of a hull must land exactly on the hull's boundary, and the
  /// centre must be strictly inside — the sharpest available check that the
  /// support-function offsets are right.
  fn check_hull<const D: usize>(
    name: &str,
    hull: &Polytope<Vec<HalfSpace<f64, D>>>,
    vertices: &[Vector<f64, D>],
    facets: usize,
  ) {
    assert_eq!(hull.half_spaces.len(), facets, "{name}: facet count");
    let far = vertices.iter().fold(0.0f64, |acc, v| acc.max(v.length()));
    assert!((far - 1.0).abs() < 1e-9, "{name}: circumradius {far}");
    for v in vertices {
      let d = hull.sdf(Point::from(*v));
      assert!(d.abs() < 1e-9, "{name}: vertex {v:?} off the boundary by {d}");
    }
    let centre = hull.sdf(Point::from(Vector::<f64, D>::zeros()));
    assert!(centre < -1e-6, "{name}: centre not interior ({centre})");
    // every unit normal, so the field is 1-Lipschitz by construction
    for h in &hull.half_spaces {
      assert!((h.normal.length() - 1.0).abs() < 1e-12, "{name}: non-unit normal");
    }
  }

  #[test] fn simplex_geometry() {
    // the defining Gram matrix: unit vertices, pairwise dot −1/D, summing to 0
    fn check<const D: usize>() {
      let v = simplex_vertices::<f64, D>();
      assert_eq!(v.len(), D + 1);
      let sum = v.iter().fold(Vector::<f64, D>::zeros(), |acc, x| acc + x);
      assert!(sum.length() < 1e-9, "D={D}: vertices do not sum to zero");
      for (i, a) in v.iter().enumerate() {
        assert!((a.length() - 1.0).abs() < 1e-12, "D={D}: vertex {i} not unit");
        for (j, b) in v.iter().enumerate() {
          if i == j { continue; }
          let want = -1.0 / D as f64;
          assert!((a.dot(b) - want).abs() < 1e-9, "D={D}: dot {i}·{j} = {}", a.dot(b));
        }
      }
      check_hull(&format!("simplex<{D}>"), &simplex::<f64, D>(), &v, D + 1);
      // with unit circumradius the facets sit 1/D from the centre
      let inradius = -simplex::<f64, D>().sdf(Point::from(Vector::<f64, D>::zeros()));
      assert!((inradius - 1.0 / D as f64).abs() < 1e-9, "D={D}: inradius {inradius}");
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<5>();
  }

  #[test] fn permutohedron_geometry() {
    // facets are indexed by proper non-empty subsets of D+1 coordinates
    assert_eq!(permutohedron::<f64, 2>().half_spaces.len(), 6);
    assert_eq!(permutohedron::<f64, 3>().half_spaces.len(), 14);
    assert_eq!(permutohedron::<f64, 4>().half_spaces.len(), 30);
    // in the plane it is the regular hexagon, so its inradius is cos(π/6)
    let inradius = -permutohedron::<f64, 2>().sdf(P2::new(0.0, 0.0));
    let want = (std::f64::consts::PI / 6.0).cos();
    assert!((inradius - want).abs() < 1e-9, "hexagon inradius {inradius} vs {want}");
    // in 3-space it is the truncated octahedron: same facet count, same inradius
    let a = -permutohedron::<f64, 3>().sdf(Point::from([0.0; 3]));
    let b = -truncated_octahedron::<f64>().sdf(Point::from([0.0; 3]));
    assert!((a - b).abs() < 1e-9, "3D permutohedron inradius {a} vs {b}");
  }

  #[test] fn platonic_and_tiling_polyhedra() {
    check_hull("icosahedron", &icosahedron::<f64>(),
      &icosahedron_vertices::<f64>(), 20);
    check_hull("dodecahedron", &dodecahedron::<f64>(),
      &dodecahedron_vertices::<f64>(), 12);
    assert_eq!(icosahedron_vertices::<f64>().len(), 12);
    assert_eq!(dodecahedron_vertices::<f64>().len(), 20);

    // the two space-filling polyhedra; both have vertex orbits at two radii, so
    // only the outermost land on the unit sphere
    let trunc = truncated_octahedron::<f64>();
    assert_eq!(trunc.half_spaces.len(), 14);
    assert!(trunc.sdf(Point::from([0.0; 3])) < -1e-6);
    let rhombic = rhombic_dodecahedron::<f64>();
    assert_eq!(rhombic.half_spaces.len(), 12);
    assert!(rhombic.sdf(Point::from([0.0; 3])) < -1e-6);
    // the rhombic dodecahedron's axial vertices are its farthest points
    for a in 0..3 {
      let v = Point::from(Vector::<f64, 3>::from_fn(|k, _| if k == a { 1.0 } else { 0.0 }));
      assert!(trunc.sdf(v) > -1e-6, "truncated octahedron: axis vertex is interior");
      assert!(rhombic.sdf(v).abs() < 1e-9, "rhombic dodecahedron: axis vertex off boundary");
    }
    check_lipschitz("dodecahedron", 1.0, |p: Point<f64, 3>| dodecahedron::<f64>().sdf(p));
    check_lipschitz("rhombic dodecahedron", 1.0,
      |p: Point<f64, 3>| rhombic_dodecahedron::<f64>().sdf(p));
  }

  /// The three exceptional regular 4-polytopes. The orbit counts and the equal
  /// vertex norms are what validate the tabulated golden-ratio constants: get a
  /// single entry wrong and either the count or the radius breaks.
  #[test] fn exceptional_4_polytopes() {
    let v24 = cell_24_vertices::<f64>();
    let v600 = cell_600_vertices::<f64>();
    let v120 = cell_120_vertices::<f64>();
    assert_eq!(v24.len(), 24, "24-cell vertices");
    assert_eq!(v600.len(), 120, "600-cell vertices");
    assert_eq!(v120.len(), 600, "120-cell vertices");
    for (name, set) in [("24", &v24), ("600", &v600), ("120", &v120)] {
      for v in set.iter() {
        assert!((v.length() - 1.0).abs() < 1e-9,
          "{name}-cell: vertex {v:?} at radius {}", v.length());
      }
    }
    // the derivation of the 120-cell's vertices asserts its own correctness:
    // exactly 600 cliques, all landing at one radius
    // the 24-cell's vertices are the D₄ roots: 24 of them, and self-duality
    // means its 24 facets match its 24 vertices in number
    check_hull("24-cell", &cell_24::<f64>(), &v24, 24);
    check_hull("120-cell", &cell_120::<f64>(), &v120, 120);
    check_hull("600-cell", &cell_600::<f64>(), &v600, 600);
    check_lipschitz("24-cell", 1.0, |p: Point<f64, 4>| cell_24::<f64>().sdf(p));
  }

  #[test] fn duocylinder_and_clifford_torus() {
    // the duocylinder is a product of two discs, exactly
    let duo = duocylinder(0.6, 0.4);
    sample::<4>(10000, 1.4, |p| {
      let r1 = (p[0] * p[0] + p[1] * p[1]).sqrt() - 0.6;
      let r2 = (p[2] * p[2] + p[3] * p[3]).sqrt() - 0.4;
      let want = V2::new(r1.max(0.0), r2.max(0.0)).length() + r1.max(r2).min(0.0);
      assert!((duo.sdf(p) - want).abs() < 1e-12, "duocylinder at {p:?}");
    });
    // the Clifford torus is the ridge where the duocylinder's two faces meet:
    // on it, both block radii are exact, so the field is −thickness/2
    let ct = CliffordTorus { r1: 0.5, r2: 0.5, thickness: 0.2 };
    let on = Point::from([0.5, 0.0, 0.0, 0.5]);
    assert!((ct.sdf(on) + 0.1).abs() < 1e-12);
    let off = Point::from([0.5, 0.0, 0.0, 0.9]);
    assert!((ct.sdf(off) - (0.4 - 0.1)).abs() < 1e-12);
    // r1 = r2 = 1/√2 places it on the unit 3-sphere
    let flat = CliffordTorus { r1: 0.5f64.sqrt(), r2: 0.5f64.sqrt(), thickness: 0.0 };
    let p = Point::from([0.5, 0.5, 0.5, 0.5]);
    assert!((p.coords.length() - 1.0).abs() < 1e-12);
    assert!(flat.sdf(p).abs() < 1e-12);
  }

  #[test] fn torus_knot_is_a_closed_curve() {
    let knot = torus_knot::<f64>(2, 3, 128, 0.06);
    assert_eq!(knot.vertices.len(), 129, "closed by repeating the first sample");
    assert_eq!(knot.vertices[0], knot.vertices[128]);
    let far = knot.vertices.iter().fold(0.0f64, |acc, v| acc.max(v.coords.length()));
    assert!(far <= 1.0 + 1e-12, "escapes the unit sphere: {far}");
    // the sampled points lie on the tube's core, hence `thickness/2` deep
    for v in knot.vertices.iter() {
      assert!((knot.sdf(*v) + 0.03).abs() < 1e-9);
    }
    check_lipschitz("torus_knot", 1.0, |p: Point<f64, 3>| knot.sdf(p));
  }

}
