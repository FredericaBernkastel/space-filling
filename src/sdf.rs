use crate::geometry::{Point, Circle, Rect};

pub trait SDF {
  fn sdf(self, pixel: Point<f32>) -> f32;
}

impl SDF for Circle {
  fn sdf(self, pixel: Point<f32>) -> f32 {
    pixel.translate(self.xy).length() - self.r
  }
}

impl SDF for Rect<f32> {
  fn sdf(self, pixel: Point<f32>) -> f32 {
    let pixel = pixel.translate(self.center);
    let component_wise_edge_distance = Point {
      x: pixel.x.abs() - self.size / 2.0,
      y: pixel.y.abs() - self.size / 2.0
    };
    let outside_distance = Point {
      x: component_wise_edge_distance.x.max(0.0),
      y: component_wise_edge_distance.y.max(0.0)
    }.length();
    let inside_distance = component_wise_edge_distance.x
      .max(component_wise_edge_distance.y)
      .min(0.0);
    outside_distance + inside_distance
  }
}

pub fn boundary_rect(pixel: Point<f32>) -> f32 {
  -Rect { center: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 }, size: 1.0 }
    .sdf(pixel)
}

