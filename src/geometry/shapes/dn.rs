//! Shapes defined for **any** dimension count `D`.
//!
//! Their fields are built from vector algebra alone — norms, per-axis
//! `abs`/`max`, dot products — so nothing in here is tied to the plane. Two of
//! them ([`Moon`], [`Kakera`]) are *solids of revolution*: the point is first
//! reduced to the `(axial, radial)` half-plane about axis 0, then measured
//! against the same 2D generator, which is exact (see [`revolve`]).

use {
  super::clamp,
  crate::{
    geometry::{Aabb, BoundingBox, Point, Real, Vector, VectorExt, V2},
    sdf::{Lipschitz, SDF},
  },
  num_traits::Float,
};

/// Unit sphere (a circle in 2D — the default; `Hypersphere::<3>` for a ball, …).
///
/// The dimension lives on the type so that combinator chains starting from a
/// bare `Hypersphere` stay inferable; it defaults to 2 in type positions.
///
/// `sdf(p) = |p| - 1` — the exact signed distance in any dimension, hence
/// 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Hypersphere<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Hypersphere<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl <T: Real, const D: usize> SDF<T, D> for Hypersphere<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    pixel.coords.length() - T::one()
  }
}

/// Axis-aligned box with center at the origin (a rectangle in 2D, a cuboid in
/// 3D).
///
/// With `q = |p| - size/2` (componentwise): `sdf(p) = |max(q, 0)| + min(max_a q_a, 0)`
/// — the exact signed distance to the box in any dimension, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Hyperrect<T, const D: usize> {
  pub size: Vector<T, D>
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Hyperrect<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let two = T::one() + T::one();
    Aabb {
      min: Point::from(self.size / -two),
      max: Point::from(self.size / two),
    }}}

impl<T: Real, const D: usize> SDF<T, D> for Hyperrect<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    let dist = pixel.coords.abs() - self.size / two;
    let outside_dist = dist
      .map(|x| x.max(T::zero()))
      .length();
    let inside_dist = dist.iter()
      .fold(T::neg_infinity(), |a, &b| a.max(b))
      .min(T::zero());
    outside_dist + inside_dist
  }}

/// `= Hyperrect { size: [2.0; D] }` — exact SDF, 1-Lipschitz (see
/// [`Hyperrect`]). Any dimension; defaults to 2.
#[derive(Debug, Copy, Clone)]
pub struct Hypersquare<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Hypersquare<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Hypersquare<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    Hyperrect { size: Vector::repeat(two) }.sdf(pixel)
  }}

/// Line segment from `a` to `b` with round caps (a capsule), any dimension.
///
/// `sdf(p) = dist(p, [a, b]) - thickness/2`, where `dist` projects `p` onto the
/// segment with a clamped parameter — the exact signed distance, hence
/// 1-Lipschitz. Degenerate for `a = b` (0/0).
#[derive(Debug, Copy, Clone)]
pub struct Line<T: nalgebra::Scalar, const D: usize> {
  pub a: Point<T, D>,
  pub b: Point<T, D>,
  pub thickness: T,
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Line<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let two = T::one() + T::one();
    let ret = Aabb::from_points([self.a, self.b]);
    let t = Vector::repeat(self.thickness / two);
    Aabb::new(ret.min - t, ret.max + t)
  }}

impl<T: Real, const D: usize> SDF<T, D> for Line<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let ba = self.b - self.a;
    let pa = pixel - self.a;
    let h = clamp(pa.dot(&ba) / ba.dot(&ba), T::zero(), T::one());
    (pa - ba * h).length() - self.thickness / (T::one() + T::one())
  }
}

/// Annulus: unit sphere with a concentric hole of radius `inner_r` (a spherical
/// shell in 3D; any dimension, defaults to 2).
///
/// `sdf(p) = max(|p| - 1, inner_r - |p|)` — the boolean subtraction of two
/// concentric spheres, which for a shell happens to be the exact signed
/// distance everywhere; 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Ring<T, const D: usize = 2> {
  pub inner_r: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Ring<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Ring<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let outer = pixel.coords.length() - T::one();
    let inner = pixel.coords.length() - self.inner_r;
    outer.max(-inner)
  }
}

/// Reduce `p` to the `(axial, radial)` half-plane of a solid of revolution
/// about axis 0: `axial = p₀`, `radial = |p₁..p_D|`.
///
/// A solid of revolution's signed distance *equals* the 2D signed distance of
/// its generator, measured in this half-plane: for any candidate `q` in the
/// solid, `|p − q| ≥ dist₂(reduce(p), reduce(q))`, with equality when `q`'s
/// perpendicular component is aligned with `p`'s — so the reduction loses
/// nothing, and the reduced 2D field is exact whenever the generator's is.
/// The map is 1-Lipschitz, so it preserves the generator's constant as well.
/// At `D = 2` it degenerates to `(p.x, |p.y|)`, hence these shapes are
/// unchanged in the plane.
#[inline]
fn revolve<T: Real, const D: usize>(p: Point<T, D>) -> V2<T> {
  let radial = (1..D)
    .fold(T::zero(), |acc, a| acc + p[a] * p[a])
    .sqrt();
  V2::new(p[0], radial)
}

/// Crescent moon, revolved about axis 0 in higher dimensions; `phase` in
/// `-1..=1`.
///
/// A unit sphere minus a unit sphere offset by `d = 2·phase` along axis 0.
/// Near the cusps the field is the distance to the cusp ridge; elsewhere
/// `sdf(p) = max(|p| - 1, 1 - |p - d·e₀|)` — together the exact signed
/// distance, hence 1-Lipschitz. At `phase = 0` the crescent degenerates to its
/// boundary sphere (`sdf = ||p| - 1|`, an empty-interior shape).
///
/// In 3D this is the crescent of revolution — a ball with a spherical bite
/// taken out along the x-axis; see [`revolve`].
#[derive(Debug, Copy, Clone)]
pub struct Moon<T> {
  pub phase: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Moon<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Moon<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    // (axial, radial) — at D = 2 this is exactly `(p.x, |p.y|)`
    let pixel = revolve(pixel);
    let d = self.phase * two;
    // algebraically d²/2d; written directly to avoid 0/0 at phase = 0
    let a = d / two;
    let b = (T::one() - a * a).max(T::zero()).sqrt();

    if d * (pixel.x * b - pixel.y * a) > d * d * (b - pixel.y).max(T::zero()) {
      (pixel - V2::new(a, b)).length()
    } else {
      (pixel.length() - T::one()).max(
        -((pixel - V2::new(d, T::zero())).length() - T::one())
      )
    }
  }
}

/// A shard: rhombus with half-diagonals `(width, 1)`, revolved about axis 0 in
/// higher dimensions — a bicone (spindle) in 3D with apexes at `±width·e₀` and
/// equatorial radius 1.
///
/// `p` is reduced to the first quadrant of the `(axial, radial)` half-plane,
/// then measured against the single edge segment, signed by the side of the
/// edge line: `sdf(p) = ±|q - proj(q)|` — the exact signed distance, hence
/// 1-Lipschitz. See [`revolve`].
#[derive(Debug, Copy, Clone)]
pub struct Kakera<T> {
  pub width: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Kakera<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let mut half = Vector::<T, D>::repeat(T::one());
    half[0] = self.width;
    Aabb::new(Point::from(half.map(|x| -x)), Point::from(half))
  }}

impl<T: Real, const D: usize> SDF<T, D> for Kakera<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    let ndot = |a: V2<T>, b: V2<T>| a.x * b.x - a.y * b.y;
    let b = V2::new(self.width, T::one());
    // |axial| paired with the (already non-negative) radial coordinate; at
    // D = 2 this is exactly `p.coords.abs()`
    let q = revolve(pixel);
    let q = V2::new(q.x.abs(), q.y);
    let mut h = (-two * ndot(q, b) + ndot(b, b)) / b.dot(&b);
    h = clamp(h, -T::one(), T::one());
    let d = (q - V2::new(T::one() - h, T::one() + h).component_mul(&b) / two).length();
    d * (q.x * b.y + q.y * b.x - b.x * b.y).signum()
  }
}

/// Axis-aligned cross: `D` orthogonal arms of half-length 1 and half-width
/// `thickness` (a plus sign in 2D, a 3-armed jack in 3D). Assumes
/// `thickness ≤ 1`.
///
/// A point is inside iff its largest `|pₐ|` is `≤ 1` **and** its second
/// largest is `≤ thickness` — that is, at most one axis may leave the central
/// cube. Writing `q₁ ≥ q₂ ≥ …` for `|pₐ|` sorted descending (a permutation,
/// hence an isometry, so distances survive the sort):
///
/// ```text
/// outside:  √( max(q₁−1, 0)² + Σ_{a≥2} max(qₐ−t, 0)² )
/// inside:  −min( 1 − q₁,  √( max(t−q₁, 0)² + max(t−q₂, 0)² ) )
/// ```
///
/// The two interior terms are the two ways out: through an arm's end cap, or
/// past a reflex corner by lifting the two largest coordinates above `t`. Both
/// branches are the exact signed distance, hence 1-Lipschitz — a plain `min`
/// of `D` boxes would instead underestimate the interior depth.
///
/// Exact for every `thickness ≤ 1`, thick arms included. The classic 2D fold
/// (Quílez's `sdCross`, which this replaces) takes the second interior term
/// unconditionally and so mis-measures the depth once `thickness > 1/2`, where
/// a point in the central cube can be nearer to an arm cap than to a reflex
/// corner; see `shapes::tests::generalized_shapes_unchanged_in_2d`.
#[derive(Debug, Copy, Clone)]
pub struct Cross<T> {
  pub thickness: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Cross<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Cross<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let t = self.thickness;
    // the two largest |pₐ|, and Σ max(|pₐ| − t, 0)² over every axis
    let (mut q1, mut q2) = (T::neg_infinity(), T::neg_infinity());
    let mut excess = T::zero();
    for a in 0..D {
      let v = pixel[a].abs();
      if v > q1 { q2 = q1; q1 = v; } else if v > q2 { q2 = v; }
      let e = (v - t).max(T::zero());
      excess = excess + e * e;
    }

    if q1 > T::one() || q2 > t {
      let end = (q1 - T::one()).max(T::zero());
      // drop the largest axis' own term: it is allowed to exceed `t`
      let own = (q1 - t).max(T::zero());
      let side = (excess - own * own).max(T::zero());
      (end * end + side).sqrt()
    } else {
      let cap = T::one() - q1;
      let (a, b) = ((t - q1).max(T::zero()), (t - q2).max(T::zero()));
      let reflex = (a * a + b * b).sqrt();
      -cap.min(reflex)
    }
  }
}

/// A closed half-space `{ v : normal·v ≤ offset }`, the building block of
/// [`Polytope`]. Prefer [`Self::new`], which normalizes.
#[derive(Debug, Copy, Clone)]
pub struct HalfSpace<T, const D: usize> {
  /// Outward normal; must be a **unit** vector for the field to be 1-Lipschitz.
  pub normal: Vector<T, D>,
  pub offset: T,
}

impl<T: Real, const D: usize> HalfSpace<T, D> {
  /// Scales `normal` to unit length (and `offset` with it, which leaves the
  /// half-space itself unchanged), so the resulting field is 1-Lipschitz
  /// whatever magnitude was passed in.
  pub fn new(normal: Vector<T, D>, offset: T) -> Self {
    let len = normal.length();
    if len > T::zero() {
      Self { normal: normal / len, offset: offset / len }
    } else {
      Self { normal, offset }
    }
  }
}

/// Convex polytope: the intersection of half-spaces — the N-dimensional
/// analogue of [`Polygon`](super::d2::Polygon), and the arbitrary-dimension
/// generalization of [`NGonC`](super::d2::NGonC)'s construction.
///
/// `sdf(p) = max_i (nᵢ·p - dᵢ)` over the bounding half-spaces. Exact in the
/// interior (where the nearest boundary point always lies on a face plane) and
/// beside each face; an *underestimate* outside past an edge or vertex, where
/// it reports the distance to the nearest face *plane* rather than to the
/// polytope — the same trade-off [`NGonC`](super::d2::NGonC) makes, and it errs
/// only toward reporting shapes as nearer than they are. A max of unit-gradient
/// linear fields, hence 1-Lipschitz provided the normals are unit (use
/// [`HalfSpace::new`]). An empty half-space list yields the constant "no shape"
/// field.
///
/// Convex only — a non-convex region is an intersection of no half-space set;
/// build one with [`Shape::subtraction`](crate::geometry::Shape::subtraction)
/// or [`union`](crate::geometry::Shape::union) instead.
///
/// [`Self::bounding_box`] reports the unit box, following this module's
/// unit-shape convention: define the polytope inside the unit sphere (offsets
/// `≤ 1`, as `NGonC` does with `cos(π/n)`) and position it with
/// [`scale`](crate::geometry::Shape::scale) / etc. A polytope that reaches
/// beyond `[-1, 1]^D` still has the correct field, but under-reports its
/// bounds — which affects only drawing and the transform pivots, never the
/// solvers.
#[derive(Debug, Copy, Clone)]
pub struct Polytope<U> {
  pub half_spaces: U
}

impl<T, U, const D: usize> BoundingBox<T, D> for Polytope<U>
  where T: Real,
        U: AsRef<[HalfSpace<T, D>]> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T, U, const D: usize> SDF<T, D> for Polytope<U>
  where T: Real,
        U: AsRef<[HalfSpace<T, D>]> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let hs = self.half_spaces.as_ref();
    if hs.is_empty() {
      return T::max_value() / (T::one() + T::one());
    }
    hs.iter()
      .map(|h| h.normal.dot(&pixel.coords) - h.offset)
      .fold(T::neg_infinity(), T::max)
  }
}

// Every shape above is either an exact SDF or a 1-Lipschitz underestimate
// assembled from unit-gradient pieces (see each shape's doc), so the honest
// bound is `1`. Combinators propagate these automatically; see
// [`crate::sdf::Lipschitz`].
impl<T: Float, const D: usize> Lipschitz<T> for Hypersphere<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Hyperrect<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Hypersquare<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float + nalgebra::Scalar, const D: usize> Lipschitz<T> for Line<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Ring<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Moon<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Kakera<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Cross<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U> Lipschitz<T> for Polytope<U> { fn lipschitz(&self) -> T { T::one() } }
