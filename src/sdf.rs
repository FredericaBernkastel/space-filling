use {
  std::ops::Add,
  num_traits::float::Float,
  euclid::{Point2D, Rect},
  crate::geometry::{Circle, WorldSpace}
};

pub trait SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T;
}

impl<T: Float + Add<Output = T>> SDF<T> for Circle<T, WorldSpace> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    (self.xy.to_vector() - pixel.to_vector()).length() - self.r
  }
}

impl SDF<f32> for Rect<f32, WorldSpace> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let pixel = self.center().to_vector() - pixel.to_vector();
    let dist = pixel.abs() - (self.size.to_vector() / 2.0);
    let outside_dist = dist
      .max([0.0, 0.0].into())
      .length();
    let inside_dist = dist.x
      .max(dist.y)
      .min(0.0);
    outside_dist + inside_dist
  }
}

pub fn boundary_rect(pixel: Point2D<f32, WorldSpace>) -> f32 {
  -Rect { origin: [0.0, 0.0].into(), size: [1.0, 1.0].into() }
    .sdf(pixel)
}
