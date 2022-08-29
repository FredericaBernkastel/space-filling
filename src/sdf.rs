use {
  euclid::{Point2D, Vector2D as V2, Rotation2D, Box2D},
  crate::geometry::{self, WorldSpace, Shape, Rotation, Scale, Translation, BoundingBox},
  num_traits::{Float, Signed},
  std::ops::{Neg, Sub}
};

/// Signed distance function
pub trait SDF<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T;
}

impl <S, P> SDF<P> for Translation<S, P>
  where S: Shape<P>,
        P: Clone + Sub<Output = P>  {
  fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    self.shape.sdf(pixel - self.offset.clone())
  }
}

impl <S, P> SDF<P> for Rotation<S, P>
  where S: Shape<P>,
        P: Float {
  fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    let pivot = self.shape.bounding_box().center();
    let pixel = Rotation2D::new(self.angle)
      .transform_point( (pixel - pivot).to_point())
      + pivot.to_vector();

    self.shape.sdf(pixel)
  }
}

impl <S, P> SDF<P> for Scale<S, P>
  where S: Shape<P>,
        P: Float {
  fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    let c = self.shape.bounding_box().center();
    let pixel = ((pixel - c) / self.scale + c.to_vector())
      .to_point();
    self.shape.sdf(pixel) * self.scale
  }
}

/// Distance to the edges of image.
pub fn boundary_rect<T: Float + Signed>(pixel: Point2D<T, WorldSpace>) -> T {
  let p5 = T::one() / (T::one() + T::one());
  -geometry::Rect { size: Point2D::splat(T::one()) }
    .translate(V2::splat(p5))
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

impl<T, S1, S2> BoundingBox<T> for Union<S1, S2>
  where T: Copy + PartialOrd,
        S1: BoundingBox<T>,
        S2: BoundingBox<T> {
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

impl<T, S1, S2> BoundingBox<T> for Subtraction<S1, S2>
  where T: Copy + PartialOrd,
    S1: BoundingBox<T>,
    S2: BoundingBox<T> {
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

impl<T, S1, S2> BoundingBox<T> for Intersection<S1, S2>
  where T: Copy + PartialOrd + num_traits::One + Neg<Output = T>,
        S1: BoundingBox<T>,
        S2: BoundingBox<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    self.s1.bounding_box()
      .intersection(&self.s2.bounding_box())
      .unwrap_or(Box2D {
        min: Point2D::splat(-T::one()),
        max: Point2D::splat(-T::one())
      })
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

impl<T, S1, S2> BoundingBox<T> for SmoothMin<T, S1, S2>
  where T: Copy + PartialOrd,
        S1: BoundingBox<T>,
        S2: BoundingBox<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}