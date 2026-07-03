//! .
//!
//! The origin of coordinate system is in top-left corner. Most of shapes are represented in the
//! interval `[-1, 1]`, and center in the origin.

use {
  std::ops::Add,
  euclid::{Point2D, Box2D, Vector2D as V2, Rotation2D, Angle},
  num_traits::Float,
  crate::sdf::{SDF, Lipschitz, Union, Subtraction, Intersection, SmoothMin}
};

pub mod shapes;
pub use shapes::*;

/// Pixel coordinate basis
#[derive(Debug, Copy, Clone)]
pub struct PixelSpace;
/// Normalized coordinate basis
#[derive(Debug, Copy, Clone)]
pub struct WorldSpace;

pub type P2<P> = Point2D<P, WorldSpace>;

pub trait BoundingBox<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace>;
}

/// Something inside a rectangular area.
///
/// Every combinator below preserves the Lipschitz constant of its operands:
/// a chain's honest bound is the *maximum* over its leaf primitives — exactly
/// how the ADF composes per-bucket bounds.
pub trait Shape<T>: SDF<T> + BoundingBox<T> {
  /// Translate by `offset`.
  ///
  /// `sdf'(p) = sdf(p - offset)` — precomposition with an isometry; preserves
  /// the Lipschitz constant exactly.
  fn translate(self, offset: V2<T, WorldSpace>) -> Translation<Self, T> where Self: Sized {
    Translation { shape: self, offset }
  }
  /// Rotate around the center of shape's bounding box.
  ///
  /// `sdf'(p) = sdf(R_θ(p - c) + c)` — precomposition with an isometry;
  /// preserves the Lipschitz constant exactly.
  fn rotate(self, angle: Angle<T>) -> Rotation<Self, T> where Self: Sized {
    Rotation { shape: self, angle }
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
impl <T, Sh> Shape<T> for Sh where Sh: SDF<T> + BoundingBox<T> {}

/// See [`Shape::translate`]. Lipschitz-preserving (isometry).
#[derive(Debug, Copy, Clone)]
pub struct Translation<S, T> {
  pub shape: S,
  pub offset: V2<T, WorldSpace>
}
impl <S, P> BoundingBox<P> for Translation<S, P>
  where S: BoundingBox<P>,
        P: Copy + Add<Output = P> {
  fn bounding_box(&self) -> Box2D<P, WorldSpace> {
    self.shape.bounding_box().translate(self.offset)
  }
}
impl <S, P> Lipschitz<P> for Translation<S, P>
  where S: Lipschitz<P> {
  fn lipschitz(&self) -> P {
    self.shape.lipschitz()
  }
}

/// Rotate around the center of shape's bounding box.
/// See [`Shape::rotate`]. Lipschitz-preserving (isometry).
#[derive(Debug, Copy, Clone)]
pub struct Rotation<S, T> {
  pub shape: S,
  pub angle: Angle<T>
}
impl <T, S> BoundingBox<T> for Rotation<S, T>
  where S: BoundingBox<T>,
        T: Float
{
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let bounding = self.shape.bounding_box();
    let pivot = bounding.center();
    let rot = |point: Point2D<_, _>| Rotation2D::new(self.angle)
      .transform_point( (point - pivot).to_point())
      + pivot.to_vector();
    update_bounding_box(bounding, rot)
  }
}
impl <S, T> Lipschitz<T> for Rotation<S, T>
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
impl <T, S> BoundingBox<T> for Scale<S, T>
  where S: BoundingBox<T>,
        T: Float
{
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let c = self.shape.bounding_box().center().to_vector();
    self.shape.bounding_box()
      .translate(-c)
      .scale(self.scale, self.scale)
      .translate(c)
  }
}
impl <S, T> Lipschitz<T> for Scale<S, T>
  where S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }
}

fn update_bounding_box<T>(
  bounding: Box2D<T, WorldSpace>,
  morphism: impl Fn(Point2D<T, WorldSpace>) -> Point2D<T, WorldSpace>
) -> Box2D<T, WorldSpace>
  where T: Float
{
  let pts = [
    [bounding.min.x, bounding.min.y],
    [bounding.max.x, bounding.min.y],
    [bounding.max.x, bounding.max.y],
    [bounding.min.x, bounding.max.y],
  ];
  let pts = pts.iter().cloned()
    .map(|p| morphism(p.into()));
  Box2D::from_points(pts)
}

#[derive(Copy, Clone, Debug)]
pub struct DistPoint<D, P, Space> {
  pub distance: D,
  pub point: Point2D<P, Space>
}

impl<F: Float, P: Default, S> Default for DistPoint<F, P, S> {
  fn default() -> Self {
    Self {
      distance: F::max_value() / (F::one() + F::one()),
      point: Point2D::default()
    }
  }
}

impl<D: PartialEq, P, S> PartialEq for DistPoint<D, P, S> {
  fn eq(&self, other: &Self) -> bool {
    self.distance.eq(&other.distance)
  }
}

impl<D: PartialOrd, P, S> PartialOrd for DistPoint<D, P, S> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.distance.partial_cmp(&other.distance)
  }
}

impl<D: PartialEq, P, S> Eq for DistPoint<D, P, S> {}

impl<P, S> std::cmp::Ord for DistPoint<f32, P, S> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // waiting for #![feature(total_cmp)]
    fn total_cmp(left: f32, right: f32) -> std::cmp::Ordering {
      let mut left = left.to_bits() as i32;
      let mut right = right.to_bits() as i32;
      left ^= (((left >> 31) as u32) >> 1) as i32;
      right ^= (((right >> 31) as u32) >> 1) as i32;

      left.cmp(&right)
    }
    total_cmp(self.distance, other.distance)
  }
}