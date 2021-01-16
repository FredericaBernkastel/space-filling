use crate::lib::Point;

#[derive(Debug, Copy, Clone)]
pub struct Circle {
  pub xy: Point,
  pub r: f32,
}

/// Circle SDF
pub fn circle(sample: Point, circle: Circle) -> f32 {
  sample.translate(circle.xy).length() - circle.r
}