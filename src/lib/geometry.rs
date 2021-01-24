use crate::geometry;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point<T> {
  pub x: T,
  pub y: T,
}

#[derive(Debug, Copy, Clone)]
pub struct Circle {
  pub xy: Point<f32>,
  pub r: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Rect<T> {
  pub center: Point<T>,
  pub size: T,
}

/// top left bottom right
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TLBR<T> {
  pub tl: Point<T>,
  pub br: Point<T>
}

impl<T: std::ops::Add<Output = T>> std::ops::Add for Point<T> {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    Self {
      x: self.x + other.x,
      y: self.y + other.y,
    }
  }
}

impl<T: std::ops::Sub<Output = T>> std::ops::Sub for Point<T> {
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    Self {
      x: self.x - other.x,
      y: self.y - other.y,
    }
  }
}

impl<RHS: Copy, T: std::ops::Mul<RHS, Output = T>> std::ops::Mul<RHS> for Point<T> {
  type Output = Self;

  fn mul(self, rhs: RHS) -> Self::Output {
    Self {
      x: self.x * rhs,
      y: self.y * rhs
    }
  }
}

impl<RHS: Copy, T: std::ops::Div<RHS, Output = T>> std::ops::Div<RHS> for Point<T> {
  type Output = Self;

  fn div(self, rhs: RHS) -> Self::Output {
    Self {
      x: self.x / rhs,
      y: self.y / rhs
    }
  }
}

impl Point<f32> {
  /// vector length
  pub fn length(self) -> f32 {
    (self.x * self.x + self.y * self.y).sqrt()
  }
  /// vector offset
  pub fn translate(self, offset: Self) -> Self {
    self - offset
  }
}

impl<T: std::cmp::PartialOrd> Point<T> {
  /// determine whether a point is inside a rectangle
  pub fn in_rect(self, rect: geometry::TLBR<T>) -> bool {
    self.x >= rect.tl.x && self.x < rect.br.x &&
      self.y >= rect.tl.y && self.y < rect.br.y
  }
}

impl From<Point<u32>> for Point<f32> {
  fn from(pt: Point<u32>) -> Self {
    Point {
      x: pt.x as f32,
      y: pt.y as f32
    }
  }
}

impl std::ops::Mul<f32> for Circle {
  type Output = Self;

  fn mul(self, rhs: f32) -> Self::Output {
    Circle {
      xy: self.xy * rhs,
      r: self.r * rhs
    }
  }
}

impl std::ops::Div<f32> for Circle {
  type Output = Self;

  fn div(self, rhs: f32) -> Self::Output {
    Circle {
      xy: self.xy / rhs,
      r: self.r / rhs
    }
  }
}

impl Into<TLBR<f32>> for Rect<f32> {
  fn into(self) -> TLBR<f32> {
    let size_half = self.size / 2.0;
    TLBR {
      tl: Point { x: self.center.x - size_half, y: self.center.y - size_half },
      br: Point { x: self.center.x + size_half, y: self.center.y + size_half },
    }
  }
}

impl Into<Rect<f32>> for TLBR<f32> {
  fn into(self) -> Rect<f32> {
    Rect {
      center: Point { x: (self.tl.x + self.br.x) / 2.0, y: (self.tl.y + self.br.y) / 2.0 },
      size: (self.br.x - self.tl.x).min(self.br.y - self.tl.y)
    }
  }
}

impl Into<Rect<u32>> for TLBR<u32> {
  fn into(self) -> Rect<u32> {
    Rect {
      center: Point { x: (self.tl.x + self.br.x) / 2, y: (self.tl.y + self.br.y) / 2 },
      size: (self.br.x - self.tl.x).min(self.br.y - self.tl.y)
    }
  }
}

impl<T> TLBR<T> {
  pub fn bl(self) -> Point<T> {
    Point { x: self.tl.x, y: self.br.y }
  }
  pub fn tr(self) -> Point<T> {
    Point { x: self.br.x, y: self.tl.y }
  }
}

pub trait Intersect {
  type Rhs;
  fn intersects(self, rhs: Self::Rhs) -> bool;
}

impl Intersect for Rect<f32> {
  type Rhs = Circle;

  fn intersects(self, c: Self::Rhs) -> bool {
    let dist = Point {
      x: (c.xy.x - self.center.x).abs(),
      y: (c.xy.y - self.center.y).abs(),
    };

    !(dist.x > (self.size / 2.0 + c.r)) &&
      !(dist.y > (self.size / 2.0 + c.r)) && (
      dist.x <= self.size / 2.0 ||
        dist.y <= self.size / 2.0 || {
        let corner_dist_sq =
          (dist.x - self.size / 2.0).powf(2.0) +
            (dist.y - self.size / 2.0).powf(2.0);
        corner_dist_sq <= c.r.powf(2.0)
      }
    )
  }
}

impl Intersect for Circle {
  type Rhs = Rect<f32>;
  fn intersects(self, r: Self::Rhs) -> bool {
    r.intersects(self)
  }
}