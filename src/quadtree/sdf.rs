use crate::lib::Point;

#[derive(Debug, Copy, Clone)]
pub struct Circle {
  pub xy: Point,
  pub r: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Rect {
  pub center: Point,
  pub size: f32,
}

/// top left bottom right
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TLBR {
  pub tl: Point,
  pub br: Point
}

impl TLBR {
  pub fn bl(self) -> Point {
    Point { x: self.tl.x, y: self.br.y }
  }
  pub fn tr(self) -> Point {
    Point { x: self.br.x, y: self.tl.y }
  }
}

impl Into<TLBR> for Rect {
  fn into(self) -> TLBR {
    let size_half = self.size / 2.0;
    TLBR {
      tl: Point { x: self.center.x - size_half, y: self.center.y - size_half },
      br: Point { x: self.center.x + size_half, y: self.center.y + size_half },
    }
  }
}

impl Into<Rect> for TLBR {
  fn into(self) -> Rect {
    Rect {
      center: Point { x: (self.tl.x + self.br.x) / 2.0, y: (self.tl.y + self.br.y) / 2.0 },
      size: (self.br.x - self.tl.x).min(self.br.y - self.tl.y)
    }
  }
}

pub trait Intersect {
  type Rhs;
  fn intersects(self, rhs: Self::Rhs) -> bool;
}

impl Intersect for Rect {
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
  type Rhs = Rect;
  fn intersects(self, r: Self::Rhs) -> bool {
    r.intersects(self)
  }
}

/// Circle SDF
pub fn circle(sample: Point, circle: Circle) -> f32 {
  sample.translate(circle.xy).length() - circle.r
}
