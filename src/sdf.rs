use {
  euclid::{Point2D, Vector2D as V2, Rotation2D, Box2D},
  crate::geometry::{self, WorldSpace, Shape, Rotation, Scale, Translation, BoundingBox},
  num_traits::Float
};

/// Signed distance function
pub trait SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T;
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
  -geometry::Square
    .translate(V2::splat(0.5))
    .scale(V2::splat(0.5))
    .sdf(pixel)
}

/// Union of two SDFs.
#[derive(Clone, Copy, Debug)]
pub struct Union<S1, S2> {
  pub s1: S1,
  pub s2: S2,
}

impl<T, S1, S2> SDF<T> for Union<S1, S2>
  where T: Float,
        S1: SDF<T>,
        S2: SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    self.s1.sdf(pixel).min(self.s2.sdf(pixel))
  }}

impl<T, S1, S2> BoundingBox<T, WorldSpace> for Union<S1, S2>
  where T: Copy + PartialOrd,
        S1: BoundingBox<T, WorldSpace>,
        S2: BoundingBox<T, WorldSpace> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}

/// Subtracion of two SDFs. Note that this operation is *not* commutative,
/// i.e. `Subtraction {a, b} =/= Subtraction {b, a}`.
#[derive(Clone, Copy, Debug)]
pub struct Subtraction<S1, S2> {
  pub s1: S1,
  pub s2: S2,
}

impl<T, S1, S2> SDF<T> for Subtraction<S1, S2>
  where T: Float,
    S1: SDF<T>,
    S2: SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    (-self.s2.sdf(pixel)).max(self.s1.sdf(pixel))
  }}

impl<T, S1, S2> BoundingBox<T, WorldSpace> for Subtraction<S1, S2>
  where T: Copy + PartialOrd,
    S1: BoundingBox<T, WorldSpace>,
    S2: BoundingBox<T, WorldSpace> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}

/// Intersection of two SDFs.
#[derive(Clone, Copy, Debug)]
pub struct Intersection<S1, S2> {
  pub s1: S1,
  pub s2: S2,
}

impl<T, S1, S2> SDF<T> for Intersection<S1, S2>
  where T: Float,
        S1: SDF<T>,
        S2: SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    self.s1.sdf(pixel).max(self.s2.sdf(pixel))
  }}

impl<T, S1, S2> BoundingBox<T, WorldSpace> for Intersection<S1, S2>
  where T: Copy + PartialOrd + num_traits::Zero,
        S1: BoundingBox<T, WorldSpace>,
        S2: BoundingBox<T, WorldSpace> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    self.s1.bounding_box()
      .intersection(&self.s2.bounding_box())
      .unwrap_or(Box2D::from_size([T::zero(), T::zero()].into()))
  }}

/// Takes the minimum of two SDFs, smoothing between them when they are close.
///
/// `k` controls the radius/distance of the smoothing. 32 is a good default value.
#[derive(Clone, Copy, Debug)]
pub struct SmoothMin<T, S1, S2> {
  pub s1: S1,
  pub s2: S2,
  pub k: T
}

impl<T, S1, S2> SDF<T> for SmoothMin<T, S1, S2>
  where T: Float,
        S1: SDF<T>,
        S2: SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let (s1, s2) = (self.s1.sdf(pixel), self.s2.sdf(pixel));
    let res = (-self.k * s1).exp2() + (-self.k * s2).exp2();
    -res.log2() / self.k
  }}

impl<T, S1, S2> BoundingBox<T, WorldSpace> for SmoothMin<T, S1, S2>
  where T: Copy + PartialOrd,
        S1: BoundingBox<T, WorldSpace>,
        S2: BoundingBox<T, WorldSpace> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}