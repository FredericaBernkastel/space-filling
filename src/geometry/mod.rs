//! Geometry vocabulary: N-dimensional points, boxes, the [`Shape`] trait with
//! its transform/boolean combinators, and the SDF shape primitives.
//!
//! Coordinates are [`nalgebra`] points with a compile-time dimension count.
//! World coordinates are normalized with the origin in the minimal corner and
//! every axis growing positive (for images: top-left origin, y-axis down);
//! the solvers operate over the unit hypercube `[0, 1]^D`. Pixel coordinates
//! are the same points over an integer scalar — the scalar type alone
//! distinguishes the two spaces.
//!
//! Each primitive is a *unit* shape inscribed in the unit sphere (spanning
//! `[-1, 1]`, centred at the origin), then positioned with
//! [`Shape::translate`] / [`Shape::scale`] / [`Shape::rotate`]. Every
//! combinator preserves or `max`-combines its operands' Lipschitz bounds, so a
//! composed shape reports an honest constant to the ADF (see [`Lipschitz`]).

use {
  nalgebra::{
    Scalar, ClosedAddAssign, ClosedSubAssign, ClosedMulAssign, ClosedDivAssign,
    Rotation as NaRotation,
  },
  num_traits::{Float, Signed, Zero},
  crate::sdf::{SDF, Lipschitz, Union, Subtraction, Intersection, SmoothMin}
};

pub mod shapes;
pub use shapes::*;

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
  pub fn union(&self, other: &Self) -> Self {
    Self {
      min: self.min.coords.zip_map(&other.min.coords, |a, b| a.min(b)).into(),
      max: self.max.coords.zip_map(&other.max.coords, |a, b| a.max(b)).into(),
    }
  }
  /// `None` if the boxes do not overlap.
  pub fn intersection(&self, other: &Self) -> Option<Self> {
    let ret = Self {
      min: self.min.coords.zip_map(&other.min.coords, |a, b| a.max(b)).into(),
      max: self.max.coords.zip_map(&other.max.coords, |a, b| a.min(b)).into(),
    };
    (0..D).all(|a| ret.min[a] < ret.max[a]).then_some(ret)
  }
  #[inline]
  pub fn translate(&self, offset: Vector<T, D>) -> Self {
    Self { min: self.min + offset, max: self.max + offset }
  }
}

pub trait BoundingBox<T: Scalar, const D: usize> {
  fn bounding_box(&self) -> Aabb<T, D>;
}

/// Something inside a rectangular area.
///
/// Every combinator below preserves the Lipschitz constant of its operands:
/// a chain's honest bound is the *maximum* over its leaf primitives — exactly
/// how the ADF composes per-bucket bounds.
pub trait Shape<T: Scalar, const D: usize>: SDF<T, D> + BoundingBox<T, D> {
  /// Translate by `offset`.
  ///
  /// `sdf'(p) = sdf(p - offset)` — precomposition with an isometry; preserves
  /// the Lipschitz constant exactly.
  fn translate(self, offset: Vector<T, D>) -> Translation<Self, T, D> where Self: Sized {
    Translation { shape: self, offset }
  }
  /// Rotate around the center of the shape's bounding box; any
  /// [`nalgebra::Rotation`] — for 2D, `Rotation2::new(angle)`.
  ///
  /// `sdf'(p) = sdf(R(p - c) + c)` — precomposition with an isometry;
  /// preserves the Lipschitz constant exactly.
  fn rotate(self, rotation: NaRotation<T, D>) -> Rotation<Self, T, D> where Self: Sized {
    Rotation { shape: self, rotation }
  }
  /// Scale around the center of shape's bounding box.
  ///
  /// `sdf'(p) = s · sdf((p - c)/s + c)` — the value re-scale cancels the
  /// coordinate re-scale (`s·L·δ/s = L·δ`), preserving the Lipschitz constant
  /// exactly. Requires `s > 0` (a negative `s` flips the field's sign
  /// semantics).
  fn scale(self, scale: T) -> Scale<Self, T> where Self: Sized {
    Scale { shape: self, scale }
  }
  /// Union of two SDFs.
  ///
  /// `sdf'(p) = min(s1, s2)` — exact in free space, underestimates interior
  /// depth where the operands overlap; `max(L₁, L₂)`-Lipschitz, since `min`
  /// of Lipschitz fields never steepens.
  fn union<U>(self, other: U) -> Union<Self, U> where Self: Sized {
    Union { s1: self, s2: other }
  }
  /// Subtraction of two SDFs. Note that this operation is *not* commutative,
  /// i.e. `Subtraction {a, b} =/= Subtraction {b, a}`.
  ///
  /// `sdf'(p) = max(s1, -s2)` — a conservative bound of the true distance
  /// (an underestimate near the carved boundary), not the exact SDF;
  /// negation and `max` both preserve constants, so `max(L₁, L₂)`-Lipschitz.
  fn subtraction<U>(self, other: U) -> Subtraction<Self, U> where Self: Sized {
    Subtraction { s1: self, s2: other }
  }
  /// Intersection of two SDFs.
  ///
  /// `sdf'(p) = max(s1, s2)` — a conservative bound (underestimates the
  /// distance outside re-entrant corners), not the exact SDF;
  /// `max(L₁, L₂)`-Lipschitz.
  fn intersection<U>(self, other: U) -> Intersection<Self, U> where Self: Sized {
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
  fn smooth_min<U>(self, other: U, k: T) -> SmoothMin<T, Self, U> where Self: Sized {
    SmoothMin { s1: self, s2: other, k }
  }
  #[cfg(feature = "drawing")]
  #[cfg_attr(docsrs, doc(cfg(feature = "drawing")))]
  fn texture<Tex>(self, texture: Tex) -> crate::drawing::Texture<Self, Tex> where Self: Sized {
    crate::drawing::Texture { shape: self, texture }
  }
}
impl <T: Scalar, Sh, const D: usize> Shape<T, D> for Sh where Sh: SDF<T, D> + BoundingBox<T, D> {}

/// See [`Shape::translate`]. Lipschitz-preserving (isometry).
#[derive(Debug, Copy, Clone)]
pub struct Translation<S, T: Scalar, const D: usize> {
  pub shape: S,
  pub offset: Vector<T, D>
}
impl <S, T, const D: usize> BoundingBox<T, D> for Translation<S, T, D>
  where S: BoundingBox<T, D>,
        T: Real {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.shape.bounding_box().translate(self.offset)
  }
}
impl <S, T: Scalar, const D: usize> Lipschitz<T> for Translation<S, T, D>
  where S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }
}

/// Rotate around the center of shape's bounding box.
/// See [`Shape::rotate`]. Lipschitz-preserving (isometry).
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
/// See [`Shape::scale`]. Lipschitz-preserving (`s > 0`; the value re-scale
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
