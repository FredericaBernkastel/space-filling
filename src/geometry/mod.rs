//! .
//!
//! The origin of coordinate system is in top-left corner. Most of shapes are represented in the
//! interval `[-1, 1]`, and center in the origin.

use {
  std::ops::Mul,
  euclid::{Point2D, Box2D, Vector2D as V2, Size2D, Rotation2D, Angle},
  num_traits::NumCast,
  crate::sdf::{SDF, Union, Subtraction, Intersection, SmoothMin}
};

pub mod shapes;
pub use shapes::*;

/// Pixel coordinate basis
#[derive(Debug, Copy, Clone)]
pub struct PixelSpace;
/// Normalized coordinate basis
#[derive(Debug, Copy, Clone)]
pub struct WorldSpace;

pub trait BoundingBox<T, S> {
  fn bounding_box(&self) -> Box2D<T, S>;
}

/// Something inside a rectangular area.
pub trait Shape: SDF<f32> + BoundingBox<f32, WorldSpace> {
  fn translate<T>(self, offset: V2<T, WorldSpace>) -> Translation<Self, T> where Self: Sized {
    Translation { shape: self, offset }
  }
  /// Rotate around the center of shape's bounding box
  fn rotate<T>(self, angle: Angle<T>) -> Rotation<Self, T> where Self: Sized {
    Rotation { shape: self, angle }
  }
  /// Scale around the center of shape's bounding box
  fn scale<T>(self, scale: V2<T, WorldSpace>) -> Scale<Self, T> where Self: Sized {
    Scale { shape: self, scale }
  }
  /// Union of two SDFs.
  fn union<U>(self, other: U) -> Union<Self, U> where Self: Sized {
    Union { s1: self, s2: other }
  }
  /// Subtracion of two SDFs. Note that this operation is *not* commutative,
  /// i.e. `Subtraction {a, b} =/= Subtraction {b, a}`.
  fn subtraction<U>(self, other: U) -> Subtraction<Self, U> where Self: Sized {
    Subtraction { s1: self, s2: other }
  }
  /// Intersection of two SDFs.
  fn intersection<U>(self, other: U) -> Intersection<Self, U> where Self: Sized {
    Intersection { s1: self, s2: other }
  }
  /// Takes the minimum of two SDFs, smoothing between them when they are close.
  ///
  /// `k` controls the radius/distance of the smoothing. 32 is a good default value.
  fn smooth_min<T, U>(self, other: U, k: T) -> SmoothMin<T, Self, U> where Self: Sized {
    SmoothMin { s1: self, s2: other, k }
  }
  #[cfg(feature = "drawing")]
  #[cfg_attr(doc, doc(cfg(feature = "drawing")))]
  fn texture<T>(self, texture: T) -> crate::drawing::Texture<Self, T> where Self: Sized {
    crate::drawing::Texture { shape: self, texture }
  }
}
impl <T> Shape for T where T: SDF<f32> + BoundingBox<f32, WorldSpace> {}

#[derive(Debug, Copy, Clone)]
pub struct Translation<S, T> {
  pub shape: S,
  pub offset: V2<T, WorldSpace>
}
impl <S> BoundingBox<f32, WorldSpace> for Translation<S, f32>
  where S: BoundingBox<f32, WorldSpace> {
  fn bounding_box(&self) -> Box2D<f32, WorldSpace> {
    self.shape.bounding_box().translate(self.offset)
  }
}

/// Rotate around the center of shape's bounding box
#[derive(Debug, Copy, Clone)]
pub struct Rotation<S, T> {
  pub shape: S,
  pub angle: Angle<T>
}
impl <S> BoundingBox<f32, WorldSpace> for Rotation<S, f32>
  where S: BoundingBox<f32, WorldSpace> {
  fn bounding_box(&self) -> Box2D<f32, WorldSpace> {
    let bounding = self.shape.bounding_box();
    let pivot = bounding.center();
    let rot = |point: Point2D<_, _>| Rotation2D::new(self.angle)
      .transform_point( (point - pivot).to_point())
      + pivot.to_vector();
    update_bounding_box(bounding, rot)
  }
}

/// Scale around the center of shape's bounding box
#[derive(Debug, Copy, Clone)]
pub struct Scale<S, T> {
  pub shape: S,
  pub scale: V2<T, WorldSpace>
}
impl <S> BoundingBox<f32, WorldSpace> for Scale<S, f32>
  where S: BoundingBox<f32, WorldSpace> {
  fn bounding_box(&self) -> Box2D<f32, WorldSpace> {
    let c = self.shape.bounding_box().center().to_vector();
    self.shape.bounding_box()
      .translate(-c)
      .scale(self.scale.x, self.scale.y)
      .translate(c)
  }
}

pub fn to_world_space<T: NumCast + Copy>(
  point: Point2D<T, PixelSpace>,
  resolution: Size2D<T, PixelSpace>
) -> Point2D<f32, WorldSpace> {
  point.to_f32().to_vector()
    .component_div(resolution.to_f32().to_vector())
    .cast_unit()
    .to_point()
}

pub fn to_pixel_space<T: NumCast + Copy + Mul<Output = T>>(
  point: Point2D<T, WorldSpace>,
  resolution: Size2D<u32, PixelSpace>
) -> Point2D<u32, PixelSpace> {
  point.to_vector().component_mul(resolution.to_vector().cast().cast_unit())
    .cast_unit()
    .to_point()
    .to_u32()
}

fn update_bounding_box(
  bounding: Box2D<f32, WorldSpace>,
  morphism: impl Fn(Point2D<f32, WorldSpace>) -> Point2D<f32, WorldSpace>
) -> Box2D<f32, WorldSpace> {
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

