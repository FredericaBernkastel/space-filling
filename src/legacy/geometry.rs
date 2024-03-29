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

impl<T: Copy + std::ops::Mul<T, Output = T>> std::ops::Mul<T> for Point<T> {
  type Output = Self;

  fn mul(self, rhs: T) -> Self::Output {
    Self {
      x: self.x * rhs,
      y: self.y * rhs
    }
  }
}

impl<T: Copy + std::ops::Div<T, Output = T>> std::ops::Div<T> for Point<T> {
  type Output = Self;

  fn div(self, rhs: T) -> Self::Output {
    Self {
      x: self.x / rhs,
      y: self.y / rhs
    }
  }
}

impl<T: Copy + std::ops::Mul<T, Output = T>> std::ops::Mul<Point<T>> for Point<T> {
  type Output = Self;

  fn mul(self, rhs: Point<T>) -> Self::Output {
    Self {
      x: self.x * rhs.x,
      y: self.y * rhs.y
    }
  }
}

impl<T: Copy + std::ops::Div<T, Output = T>> std::ops::Div<Point<T>> for Point<T> {
  type Output = Self;

  fn div(self, rhs: Point<T>) -> Self::Output {
    Self {
      x: self.x / rhs.x,
      y: self.y / rhs.y
    }
  }
}

impl<T: std::cmp::PartialOrd> Point<T> {
  /// determine whether a point is inside a rectangle
  pub fn in_rect(self, rect: TLBR<T>) -> bool {
    self.x >= rect.tl.x && self.x < rect.br.x &&
      self.y >= rect.tl.y && self.y < rect.br.y
  }
}

pub trait Intersect<Rhs> {
  fn intersects(self, rhs: Rhs) -> bool;
}

impl Intersect<Circle> for Rect<f32> {
  fn intersects(self, c: Circle) -> bool {
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

impl Intersect<Self> for Rect<f32> {
  fn intersects(self, rhs: Self) -> bool {
    let a: TLBR<f32> = self.into();
    let b: TLBR<f32> = rhs.into();

    [a.tl, a.tr(), a.bl(), a.br]
      .iter()
      .any(|pt| pt.in_rect(b)) ||
      [b.tl, b.tr(), b.bl(), b.br]
        .iter()
        .any(|pt| pt.in_rect(a))
  }
}

impl Intersect<Rect<f32>> for Circle {
  fn intersects(self, r: Rect<f32>) -> bool {
    r.intersects(self)
  }
}

/// Rectangle (center, size)
#[derive(Copy, Clone, Debug)]
pub struct Rect<T> {
  pub center: Point<T>,
  pub size: T,
}

/// Rectangle (top-left, bottom-right)
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TLBR<T> {
  pub tl: Point<T>,
  pub br: Point<T>
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
