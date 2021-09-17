use {
  euclid::{Point2D, Vector2D as V2, Rotation2D},
  crate::geometry::{self, WorldSpace, Shape, Rotation, Scale, Translation}
};

/// Signed distance function
pub trait SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T;
}

impl SDF<f32> for geometry::Circle {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    pixel.to_vector().length() - 0.5
  }
}

impl SDF<f32> for geometry::Square {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    //let pixel = self.center().to_vector() - pixel.to_vector();
    //let dist = pixel.to_vector().abs() - (self.size.to_vector() / 2.0);
    let dist = pixel.to_vector().abs() - V2::splat(0.5);
    let outside_dist = dist
      .max(V2::splat(0.0))
      .length();
    let inside_dist = dist.x
      .max(dist.y)
      .min(0.0);
    outside_dist + inside_dist
  }
}

impl <S> SDF<f32> for Translation<S, f32>
  where S: Shape {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    self.shape.sdf(pixel - self.offset)
  }
}

impl <S> SDF<f32> for Rotation<S, f32>
  where S: Shape {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let pivot = self.shape.bounding_box().center();
    let pixel = Rotation2D::new(self.angle)
      .transform_point( (pixel - pivot).to_point())
      + pivot.to_vector();

    self.shape.sdf(pixel)
  }
}

impl <S> SDF<f32> for Scale<S, f32>
  where S: Shape {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let c = self.shape.bounding_box().center();
    let pixel = ((pixel - c)
      .component_div(self.scale) + c.to_vector())
      .to_point();
    self.shape.sdf(pixel) * self.scale.x.min(self.scale.y)
  }
}

/// Distance to the edges of image.
pub fn boundary_rect(pixel: Point2D<f32, WorldSpace>) -> f32 {
  -geometry::Square.translate(V2::splat(0.5))
    .sdf(pixel)
}
