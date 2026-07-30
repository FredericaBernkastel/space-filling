//! Geometry vocabulary: N-dimensional points, vectors and axis-aligned boxes.
//!
//! Coordinates are [`nalgebra`] points with a compile-time dimension count.
//! World coordinates are normalized with the origin in the minimal corner and
//! every axis growing positive (for images: top-left origin, y-axis down);
//! the solvers operate over the unit hypercube `[0, 1]^D`. Pixel coordinates
//! are the same points over an integer scalar — the scalar type alone
//! distinguishes the two spaces.
//!
//! Fields are positioned by the [`Combinator`] algebra, which lives in
//! [`sdf`](crate::sdf) alongside the traits it composes.

use {
  nalgebra::{
    Scalar, ClosedAddAssign, ClosedSubAssign, ClosedMulAssign, ClosedDivAssign,
    Rotation as NaRotation,
  },
  num_traits::{Float, Signed, Zero},
  crate::sdf::{
    Lipschitz, Union, Subtraction, Intersection, SmoothMin,
    Shell, Offset, Extrude, Revolve
  }
};

/// An N-dimensional point; `Point<T, 2>` replaces the previous euclid points.
pub type Point<T, const D: usize> = nalgebra::Point<T, D>;
/// An N-dimensional vector.
pub type Vector<T, const D: usize> = nalgebra::SVector<T, D>;
/// 2D convenience alias.
pub type P2<T> = Point<T, 2>;
/// 2D convenience alias.
pub type V2<T> = Vector<T, 2>;

/// The crate-wide scalar bound: [`Float`] supplies *all* scalar math (deliberately
/// not nalgebra's `RealField`, whose method names collide with `Float`); the
/// nalgebra bounds admit the point/vector containers and their operators.
pub trait Real:
  Float + Signed + Scalar
  + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign + ClosedDivAssign {}
impl<T> Real for T where T:
  Float + Signed + Scalar
  + ClosedAddAssign + ClosedSubAssign + ClosedMulAssign + ClosedDivAssign {}

/// `Float`-bounded vector length; nalgebra's own `norm` demands `RealField`,
/// which cannot be in scope next to `Float` (colliding method names).
pub trait VectorExt<T> {
  fn length(&self) -> T;
  /// `self / self.length()`, or zero when the length vanishes.
  fn robust_normalize(&self) -> Self;
}

impl<T: Real, const D: usize> VectorExt<T> for Vector<T, D> {
  #[inline]
  fn length(&self) -> T {
    self.iter().fold(T::zero(), |acc, &x| acc + x * x).sqrt()
  }
  #[inline]
  fn robust_normalize(&self) -> Self {
    let len = self.length();
    if len > T::zero() { self / len } else { Self::zeros() }
  }
}

/// An axis-aligned box, `min` the minimal corner. The one box type used
/// everywhere: shape bounds, insertion domains, and the ADF tree cells.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Aabb<T: Scalar, const D: usize> {
  pub min: Point<T, D>,
  pub max: Point<T, D>,
}

impl<T: Scalar, const D: usize> Aabb<T, D> {
  #[inline]
  pub fn new(min: Point<T, D>, max: Point<T, D>) -> Self {
    Self { min, max }
  }
}

// available for integer (pixel) boxes as well
impl<T: Scalar + ClosedSubAssign, const D: usize> Aabb<T, D> {
  #[inline]
  pub fn size(&self) -> Vector<T, D> {
    &self.max - &self.min
  }
}

impl<T: Real, const D: usize> Aabb<T, D> {
  /// The unit hypercube `[0, 1]^D`.
  #[inline]
  pub fn unit() -> Self {
    Self {
      min: Point::from(Vector::repeat(T::zero())),
      max: Point::from(Vector::repeat(T::one())),
    }
  }
  /// `[-half, half]^D`.
  #[inline]
  pub fn symmetric(half: T) -> Self {
    Self {
      min: Point::from(Vector::repeat(-half)),
      max: Point::from(Vector::repeat(half)),
    }
  }
  /// The smallest box containing all `points`; degenerate at the origin for an
  /// empty input.
  pub fn from_points(points: impl IntoIterator<Item = Point<T, D>>) -> Self {
    let mut iter = points.into_iter();
    let first = match iter.next() {
      Some(p) => p,
      None => Point::from(Vector::repeat(T::zero())),
    };
    let init = Self { min: first, max: first };
    iter.fold(init, |acc, p| Self {
      min: acc.min.coords.zip_map(&p.coords, |a, b| a.min(b)).into(),
      max: acc.max.coords.zip_map(&p.coords, |a, b| a.max(b)).into(),
    })
  }
  #[inline]
  pub fn center(&self) -> Point<T, D> {
    let two = T::one() + T::one();
    self.min + (self.max - self.min) / two
  }
  /// Half-open containment: `min[a] <= pt[a] < max[a]` on every axis
  /// (`false` for NaN).
  #[inline]
  pub fn contains(&self, pt: &Point<T, D>) -> bool {
    (0..D).all(|a| self.min[a] <= pt[a] && pt[a] < self.max[a])
  }
  /// Open-interval overlap: empty boxes intersect nothing.
  #[inline]
  pub fn intersects(&self, other: &Self) -> bool {
    (0..D).all(|a| self.min[a] < other.max[a] && other.min[a] < self.max[a])
  }
  /// The closest point of the box to `pt` (per-axis clamp).
  #[inline]
  pub fn clamp_point(&self, pt: &Point<T, D>) -> Point<T, D> {
    Point::from(std::array::from_fn(|a| pt[a].max(self.min[a]).min(self.max[a])))
  }
  /// The smallest box containing both — named `merge` rather than `union` so
  /// that the blanket [`Combinator::union`] cannot shadow it (a by-value method
  /// call probes trait methods taking `self` before inherent ones taking
  /// `&self`).
  pub fn merge(&self, other: &Self) -> Self {
    Self {
      min: self.min.coords.zip_map(&other.min.coords, |a, b| a.min(b)).into(),
      max: self.max.coords.zip_map(&other.max.coords, |a, b| a.max(b)).into(),
    }
  }
  /// The overlap of the two boxes, or `None` if they are disjoint. Named `clip`
  /// rather than `intersection` for the same reason [`Self::merge`] is.
  pub fn clip(&self, other: &Self) -> Option<Self> {
    let ret = Self {
      min: self.min.coords.zip_map(&other.min.coords, |a, b| a.max(b)).into(),
      max: self.max.coords.zip_map(&other.max.coords, |a, b| a.min(b)).into(),
    };
    (0..D).all(|a| ret.min[a] < ret.max[a]).then_some(ret)
  }
  /// Slide the box by `offset` — named `shift` rather than `translate` for the
  /// same reason [`Self::merge`] is not `union`.
  #[inline]
  pub fn shift(&self, offset: Vector<T, D>) -> Self {
    Self { min: self.min + offset, max: self.max + offset }
  }
  /// Grow by `amount` on every side (shrink, for a negative `amount`).
  #[inline]
  pub fn inflate(&self, amount: T) -> Self {
    let v = Vector::repeat(amount);
    Self { min: self.min - v, max: self.max + v }
  }
}

pub trait BoundingBox<T: Scalar, const D: usize> {
  fn bounding_box(&self) -> Aabb<T, D>;
}

/// The SDF algebra: every way to transform or combine a field into another
/// field. Each combinator preserves or `max`-combines its operands' Lipschitz
/// constants, so a composed chain reports an honest bound to the ADF (see
/// [`Lipschitz`]).
///
/// The trait carries no dimension of its own. Every method takes its scalar and
/// dimension as *method* generics, pinned by the arguments — which is what lets
/// a field whose type does not name its dimension be wrapped directly:
///
/// ```
/// # use adaptive_distance_field::{geometry::*, sdf::SDF};
/// # #[derive(Copy, Clone)] struct Ball;
/// # impl<const D: usize> SDF<f64, D> for Ball {
/// #   fn sdf(&self, p: Point<f64, D>) -> f64 { p.coords.length() - 1.0 } }
/// # impl<const D: usize> BoundingBox<f64, D> for Ball {
/// #   fn bounding_box(&self) -> Aabb<f64, D> { Aabb::symmetric(1.0) } }
/// // `Ball` is a field in *every* dimension, yet this chain needs no turbofish
/// let carved = Ball.offset(0.05).subtraction(Ball.scale(0.4));
/// assert!(carved.sdf(Point::from([1.02, 0.0, 0.0])) < 0.0);
/// ```
///
/// Were these parameterized by `D`, that chain could not compile: nothing would
/// pin `D` at the `offset` call, since neither `Ball` nor `Offset<S, T>` names
/// it, and the eventual `sdf` call constrains a different inference variable.
/// [`extrude`](Self::extrude) could not be stated at all, its base and target
/// dimensions differing.
///
/// Blanket-implemented, so it applies to every field (and, harmlessly, to
/// everything else); bring it into scope with
/// `use adaptive_distance_field::geometry::*`. One consequence of that reach: a
/// by-value method call probes trait methods before inherent ones, so
/// `union`/`intersection`/`scale` here would shadow same-named `&self` methods
/// on unrelated types. That is why [`Aabb`] calls its set operations
/// [`merge`](Aabb::merge) and [`clip`](Aabb::clip), and why
/// `num_complex::Complex::scale` needs UFCS when this trait is in scope.
pub trait Combinator: Sized {
  /// Translate by `offset`.
  ///
  /// `sdf'(p) = sdf(p - offset)` — precomposition with an isometry; preserves
  /// the Lipschitz constant exactly.
  fn translate<T: Scalar, const D: usize>(self, offset: Vector<T, D>)
    -> Translation<Self, T, D> {
    Translation { shape: self, offset }
  }
  /// Rotate around the center of the shape's bounding box; any
  /// [`nalgebra::Rotation`] — for 2D, `Rotation2::new(angle)`.
  ///
  /// `sdf'(p) = sdf(R(p - c) + c)` — precomposition with an isometry;
  /// preserves the Lipschitz constant exactly.
  fn rotate<T: Scalar, const D: usize>(self, rotation: NaRotation<T, D>)
    -> Rotation<Self, T, D> {
    Rotation { shape: self, rotation }
  }
  /// Scale around the center of shape's bounding box.
  ///
  /// `sdf'(p) = s · sdf((p - c)/s + c)` — the value re-scale cancels the
  /// coordinate re-scale (`s·L·δ/s = L·δ`), preserving the Lipschitz constant
  /// exactly. Requires `s > 0` (a negative `s` flips the field's sign
  /// semantics).
  fn scale<T>(self, scale: T) -> Scale<Self, T> {
    Scale { shape: self, scale }
  }
  /// Union of two SDFs.
  ///
  /// `sdf'(p) = min(s1, s2)` — exact in free space, underestimates interior
  /// depth where the operands overlap; `max(L₁, L₂)`-Lipschitz, since `min`
  /// of Lipschitz fields never steepens.
  fn union<U>(self, other: U) -> Union<Self, U> {
    Union { s1: self, s2: other }
  }
  /// Subtraction of two SDFs. Note that this operation is *not* commutative,
  /// i.e. `Subtraction {a, b} =/= Subtraction {b, a}`.
  ///
  /// `sdf'(p) = max(s1, -s2)` — a conservative bound of the true distance
  /// (an underestimate near the carved boundary), not the exact SDF;
  /// negation and `max` both preserve constants, so `max(L₁, L₂)`-Lipschitz.
  fn subtraction<U>(self, other: U) -> Subtraction<Self, U> {
    Subtraction { s1: self, s2: other }
  }
  /// Intersection of two SDFs.
  ///
  /// `sdf'(p) = max(s1, s2)` — a conservative bound (underestimates the
  /// distance outside re-entrant corners), not the exact SDF;
  /// `max(L₁, L₂)`-Lipschitz.
  fn intersection<U>(self, other: U) -> Intersection<Self, U> {
    Intersection { s1: self, s2: other }
  }
  /// Takes the minimum of two SDFs, smoothing between them when they are close.
  ///
  /// `k` controls the radius/distance of the smoothing. 32 is a good default value.
  ///
  /// `sdf'(p) = -log2(2^(-k·s1) + 2^(-k·s2)) / k` — its gradient is the convex
  /// combination `w·∇s1 + (1-w)·∇s2`, `w ∈ (0, 1)`, hence `max(L₁, L₂)`-Lipschitz.
  /// The value dips below `min(s1, s2)` by at most `1/k` (shapes read slightly
  /// inflated near the blend).
  fn smooth_min<T, U>(self, other: U, k: T) -> SmoothMin<T, Self, U> {
    SmoothMin { s1: self, s2: other, k }
  }

  /// Hollow the shape out into a `2·half_width`-thick shell around its
  /// boundary: `sdf'(p) = |sdf(p)| - half_width`.
  ///
  /// Turns any solid into a surface: an annulus from a disc, a frame from a cube,
  /// a thickened sheet from an implicit surface. `|·|` is 1-Lipschitz, so the
  /// constant is preserved exactly, and the result is the exact signed distance
  /// wherever the operand's is.
  fn shell<T>(self, half_width: T) -> Shell<Self, T> {
    Shell { shape: self, half_width }
  }
  /// Grow the shape by `radius` in every direction: `sdf'(p) = sdf(p) - radius`.
  ///
  /// Rounds off corners, since the offset surface of a corner is a sphere arc —
  /// the standard way to get a rounded box, rounded cross or rounded polytope.
  /// Exact for `radius ≥ 0` when the operand is exact, and Lipschitz-preserving
  /// either way (a constant shift has zero gradient). A negative `radius`
  /// erodes instead, which can empty the shape entirely.
  ///
  /// It also repairs the conservative-outside underestimate of a field built as a
  /// maximum over half-spaces, such as a convex polytope: offsetting moves the
  /// true surface out to where the face-plane maximum is exact, so a rounded
  /// polytope is exact everywhere its rounding radius reaches.
  fn offset<T>(self, radius: T) -> Offset<Self, T> {
    Offset { shape: self, radius }
  }
  /// Lift a `(D-1)`-dimensional shape into `D` dimensions by giving it
  /// `2·half_height` of thickness along the last axis — a prism.
  ///
  /// The field is the box construction with the base field standing in for one
  /// axis, so it is the exact signed distance whenever the base's is, and
  /// `max(L, 1)`-Lipschitz (base and axial terms act on disjoint axes, so their
  /// gradients are orthogonal).
  ///
  /// This is what makes a field defined only in the plane usable in 3D and above:
  /// every 2D profile becomes a prism of that cross-section, and extrusions nest,
  /// one axis at a time. Implemented for base/target pairs up to `(5, 6)`.
  fn extrude<T>(self, half_height: T) -> Extrude<Self, T> {
    Extrude { shape: self, half_height }
  }
  /// Sweep a 2D profile around axis 0, offset `radius` from it — a solid of
  /// revolution in any dimension.
  ///
  /// `self` is read as a profile in the `(axial, radial)` half-plane: its `x`
  /// is the coordinate along the axis of revolution, its `y` the distance from
  /// that axis. Formally `sdf'(p) = sdf(p₀, |p₁..p_D| - radius)`.
  ///
  /// That reduction loses nothing: for any candidate `q` in the swept solid,
  /// `|p - q| >= dist2(reduce(p), reduce(q))`, with equality when the
  /// perpendicular component of `q` is aligned with that of `p` — so the field is
  /// the *exact* signed distance whenever the profile's is. The map is itself
  /// 1-Lipschitz, so the constant survives as well.
  ///
  /// At `radius = 0` the profile is centred on the axis, giving spindles and lens
  /// shapes; at `radius > 0` it is held clear of the axis and sweeps a torus of
  /// that profile — a circular profile gives the ordinary torus, a star-shaped
  /// one a star-sectioned ring.
  ///
  /// Exact while `radius` keeps the profile clear of the axis (`radius ≥` the
  /// profile's inward reach). Push it closer and the swept solid passes through
  /// itself; the field then *underestimates* near the axis — conservative, so
  /// still sound for the solvers, but no longer a true distance.
  fn revolve<T>(self, radius: T) -> Revolve<Self, T> {
    Revolve { shape: self, radius }
  }
}

impl<S> Combinator for S {}

/// See [`Combinator::translate`]. Lipschitz-preserving (isometry).
#[derive(Debug, Copy, Clone)]
pub struct Translation<S, T: Scalar, const D: usize> {
  pub shape: S,
  pub offset: Vector<T, D>
}
impl <S, T, const D: usize> BoundingBox<T, D> for Translation<S, T, D>
  where S: BoundingBox<T, D>,
        T: Real {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.shape.bounding_box().shift(self.offset)
  }
}
impl <S, T: Scalar, const D: usize> Lipschitz<T> for Translation<S, T, D>
  where S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }
}

/// Rotate around the center of shape's bounding box.
/// See [`Combinator::rotate`]. Lipschitz-preserving (isometry).
#[derive(Debug, Copy, Clone)]
pub struct Rotation<S, T: Scalar, const D: usize> {
  pub shape: S,
  pub rotation: NaRotation<T, D>
}
impl <S, T, const D: usize> BoundingBox<T, D> for Rotation<S, T, D>
  where S: BoundingBox<T, D>,
        T: Real
{
  fn bounding_box(&self) -> Aabb<T, D> {
    // A box rotated around its own centre hulls to `|R|·half` — exact, and
    // independent of the rotation's sign, so it also bounds the sampled
    // `sdf(R(p − c) + c)` field's support.
    let bounding = self.shape.bounding_box();
    let pivot = bounding.center();
    let two = T::one() + T::one();
    let half = bounding.size() / two;
    let half = self.rotation.matrix().abs() * half;
    Aabb { min: pivot - half, max: pivot + half }
  }
}
impl <S, T: Scalar, const D: usize> Lipschitz<T> for Rotation<S, T, D>
  where S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }
}

/// Scale around the center of shape's bounding box.
/// See [`Combinator::scale`]. Lipschitz-preserving (`s > 0`; the value re-scale
/// cancels the coordinate re-scale).
#[derive(Debug, Copy, Clone)]
pub struct Scale<S, T> {
  pub shape: S,
  pub scale: T
}
impl <S, T, const D: usize> BoundingBox<T, D> for Scale<S, T>
  where S: BoundingBox<T, D>,
        T: Real
{
  fn bounding_box(&self) -> Aabb<T, D> {
    let bounding = self.shape.bounding_box();
    let c = bounding.center();
    Aabb {
      min: c + (bounding.min - c) * self.scale,
      max: c + (bounding.max - c) * self.scale,
    }
  }
}
impl <S, T> Lipschitz<T> for Scale<S, T>
  where S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }
}

/// A field value paired with the point it was sampled at; `distance` and the
/// point scalar may differ (e.g. an `f32` field over integer pixel coordinates).
#[derive(Copy, Clone, Debug)]
pub struct DistPoint<T, P: Scalar, const D: usize> {
  pub distance: T,
  pub point: Point<P, D>
}

impl<T: Float, P: Scalar + Zero, const D: usize> Default for DistPoint<T, P, D> {
  fn default() -> Self {
    Self {
      distance: T::max_value() / (T::one() + T::one()),
      point: Point::from(Vector::zeros())
    }
  }
}

impl<T: PartialEq, P: Scalar, const D: usize> PartialEq for DistPoint<T, P, D> {
  fn eq(&self, other: &Self) -> bool {
    self.distance.eq(&other.distance)
  }
}

impl<T: PartialOrd, P: Scalar, const D: usize> PartialOrd for DistPoint<T, P, D> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.distance.partial_cmp(&other.distance)
  }
}

impl<T: PartialEq, P: Scalar, const D: usize> Eq for DistPoint<T, P, D> {}

impl<P: Scalar, const D: usize> std::cmp::Ord for DistPoint<f32, P, D> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    self.distance.total_cmp(&other.distance)
  }
}
