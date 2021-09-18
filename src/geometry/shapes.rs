use {
  super::{BoundingBox, WorldSpace},
  crate::sdf::SDF,
  euclid::{Box2D, Point2D, Vector2D as V2}
};

/// Unit circle
#[derive(Debug, Copy, Clone)]
pub struct Circle;

#[derive(Debug, Copy, Clone)]
pub struct Square;

impl<S> BoundingBox<f32, S> for Circle {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl<S> BoundingBox<f32, S> for Square {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl SDF<f32> for Circle {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    pixel.to_vector().length() - 1.0
  }
}

impl SDF<f32> for Square {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    //let pixel = self.center().to_vector() - pixel.to_vector();
    //let dist = pixel.to_vector().abs() - (self.size.to_vector() / 2.0);
    let dist = pixel.to_vector().abs() - V2::splat(1.0);
    let outside_dist = dist
      .max(V2::splat(0.0))
      .length();
    let inside_dist = dist.x
      .max(dist.y)
      .min(0.0);
    outside_dist + inside_dist
  }
}
