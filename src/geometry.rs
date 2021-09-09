use std::ops::{Mul, Div};
use euclid::{Point2D};

#[derive(Debug, Copy, Clone)]
pub struct PixelSpace;
#[derive(Debug, Copy, Clone)]
pub struct WorldSpace;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Circle<T, U> {
  pub xy: Point2D<T, U>,
  pub r: T,
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