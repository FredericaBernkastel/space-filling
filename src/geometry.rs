use std::ops::{Add, Mul, Div};
use euclid::{Point2D, Box2D, Vector2D as V2, Size2D};
use num_traits::NumCast;

/// Pixel coordinate
#[derive(Debug, Copy, Clone)]
pub struct PixelSpace;
/// Normalized coordinate
#[derive(Debug, Copy, Clone)]
pub struct WorldSpace;

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

pub trait BoundingBox<T, S> {
  fn bounding_box(&self) -> Box2D<T, S>;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Circle<T, U> {
  pub xy: Point2D<T, U>,
  pub r: T,
}

impl<S> BoundingBox<f32, S> for Circle<f32, S> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      (self.xy.to_vector() - V2::splat(self.r)).to_point(),
      (self.xy.to_vector() + V2::splat(self.r)).to_point()
    )
  }
}

impl<T: Copy + Mul<Output = T>, U> Mul<T> for Circle<T, U> {
  type Output = Self;

  fn mul(self, rhs: T) -> Self::Output {
    Circle {
      xy: self.xy * rhs,
      r: self.r * rhs
    }
  }
}

impl<T: Copy + Div<Output = T>, U> Div<T> for Circle<T, U> {
  type Output = Self;

  fn div(self, rhs: T) -> Self::Output {
    Circle {
      xy: self.xy / rhs,
      r: self.r / rhs
    }
  }
}

impl<T: Add<Output = T> + Copy, S> BoundingBox<T, S> for euclid::Rect<T, S> {
  fn bounding_box(&self) -> Box2D<T, S> {
    self.to_box2d()
  }
}