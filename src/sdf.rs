use {
  crate::{
    geometry::{self, Point, Vector, Real, Aabb, Shape, Rotation, Scale, Translation, BoundingBox},
  },
  nalgebra::Scalar,
  num_traits::Float,
};

/// Signed distance function over `D`-dimensional points.
pub trait SDF<T: Scalar, const D: usize> {
  fn sdf(&self, p: Point<T, D>) -> T;
}

/// An upper bound of the field's Lipschitz constant:
/// `|sdf(p) - sdf(q)| ≤ lipschitz() · |p - q|` for all `p`, `q`.
///
/// The ADF redundancy test certifies with this bound
/// ([`sdf_geq_everywhere`](crate::solver::adf)), and the `D*`-pruned insertion
/// walk skips subtrees with it — so it must be *honest*: an understated
/// constant can corrupt the field, an overstated one merely costs pruning
/// power. Exact SDFs return `1`; distance *estimators* declare their own
/// bound; combinators propagate `max` over their operands.
///
/// [`Primitive::from_shape`](crate::solver::Primitive::from_shape) derives the
/// stored bound from this trait automatically.
pub trait Lipschitz<T> {
  fn lipschitz(&self) -> T;
}

impl <S, T, const D: usize> SDF<T, D> for Translation<S, T, D>
  where S: Shape<T, D>,
        T: Real {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    self.shape.sdf(pixel - self.offset)
  }
}

impl <S, T, const D: usize> SDF<T, D> for Rotation<S, T, D>
  where S: Shape<T, D>,
        T: Real {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let pivot = self.shape.bounding_box().center();
    let pixel = self.rotation.matrix() * (pixel - pivot) + pivot.coords;
    self.shape.sdf(Point::from(pixel))
  }
}

impl <S, T, const D: usize> SDF<T, D> for Scale<S, T>
  where S: Shape<T, D>,
        T: Real {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let c = self.shape.bounding_box().center();
    let pixel = (pixel - c) / self.scale + c.coords;
    self.shape.sdf(Point::from(pixel)) * self.scale
  }
}

/// Distance to the edges of the image (the unit hypercube), positive inside.
///
/// The negation of the exact unit-cube SDF: `sdf(p) = -sdf_rect(p - 0.5)`.
/// Negation preserves the constant — 1-Lipschitz.
pub fn boundary_rect<T: Real, const D: usize>(pixel: Point<T, D>) -> T {
  let p5 = T::one() / (T::one() + T::one());
  -geometry::Rect { size: Vector::repeat(T::one()) }
    .translate(Vector::repeat(p5))
    .sdf(pixel)
}

/// Union of two SDFs: `min(s1, s2)`.
/// See [`Shape::union`]; `max(L₁, L₂)`-Lipschitz.
#[derive(Clone, Copy, Debug)]
pub struct Union<S1, S2> {
  pub s1: S1,
  pub s2: S2,
}

impl<T, S1, S2, const D: usize> SDF<T, D> for Union<S1, S2>
  where T: Real,
        S1: SDF<T, D>,
        S2: SDF<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    self.s1.sdf(pixel).min(self.s2.sdf(pixel))
  }}

impl<T, S1, S2, const D: usize> BoundingBox<T, D> for Union<S1, S2>
  where T: Real,
        S1: BoundingBox<T, D>,
        S2: BoundingBox<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}

impl<T, S1, S2> Lipschitz<T> for Union<S1, S2>
  where T: Float,
        S1: Lipschitz<T>,
        S2: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.s1.lipschitz().max(self.s2.lipschitz())
  }}

/// Subtraction of two SDFs: `max(s1, -s2)`. Note that this operation is *not*
/// commutative, i.e. `Subtraction {a, b} =/= Subtraction {b, a}`.
/// See [`Shape::subtraction`]; `max(L₁, L₂)`-Lipschitz.
#[derive(Clone, Copy, Debug)]
pub struct Subtraction<S1, S2> {
  pub s1: S1,
  pub s2: S2,
}

impl<T, S1, S2, const D: usize> SDF<T, D> for Subtraction<S1, S2>
  where T: Real,
    S1: SDF<T, D>,
    S2: SDF<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    (-self.s2.sdf(pixel)).max(self.s1.sdf(pixel))
  }}

impl<T, S1, S2, const D: usize> BoundingBox<T, D> for Subtraction<S1, S2>
  where T: Real,
    S1: BoundingBox<T, D>,
    S2: BoundingBox<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}

impl<T, S1, S2> Lipschitz<T> for Subtraction<S1, S2>
  where T: Float,
        S1: Lipschitz<T>,
        S2: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.s1.lipschitz().max(self.s2.lipschitz())
  }}

/// Intersection of two SDFs: `max(s1, s2)`.
/// See [`Shape::intersection`]; `max(L₁, L₂)`-Lipschitz.
#[derive(Clone, Copy, Debug)]
pub struct Intersection<S1, S2> {
  pub s1: S1,
  pub s2: S2,
}

impl<T, S1, S2, const D: usize> SDF<T, D> for Intersection<S1, S2>
  where T: Real,
        S1: SDF<T, D>,
        S2: SDF<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    self.s1.sdf(pixel).max(self.s2.sdf(pixel))
  }}

impl<T, S1, S2, const D: usize> BoundingBox<T, D> for Intersection<S1, S2>
  where T: Real,
        S1: BoundingBox<T, D>,
        S2: BoundingBox<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.s1.bounding_box()
      .intersection(&self.s2.bounding_box())
      .unwrap_or(Aabb {
        min: Point::from(Vector::repeat(-T::one())),
        max: Point::from(Vector::repeat(-T::one()))
      })
  }}

impl<T, S1, S2> Lipschitz<T> for Intersection<S1, S2>
  where T: Float,
        S1: Lipschitz<T>,
        S2: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.s1.lipschitz().max(self.s2.lipschitz())
  }}

/// Takes the minimum of two SDFs, smoothing between them when they are close.
///
/// `k` controls the radius/distance of the smoothing. 32 is a good default value.
/// See [`Shape::smooth_min`]; `max(L₁, L₂)`-Lipschitz (its gradient is a
/// convex combination of the operands' gradients).
#[derive(Clone, Copy, Debug)]
pub struct SmoothMin<T, S1, S2> {
  pub s1: S1,
  pub s2: S2,
  pub k: T
}

impl<T, S1, S2, const D: usize> SDF<T, D> for SmoothMin<T, S1, S2>
  where T: Real,
        S1: SDF<T, D>,
        S2: SDF<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let (s1, s2) = (self.s1.sdf(pixel), self.s2.sdf(pixel));
    let res = (-self.k * s1).exp2() + (-self.k * s2).exp2();
    -res.log2() / self.k
  }}

impl<T, S1, S2, const D: usize> BoundingBox<T, D> for SmoothMin<T, S1, S2>
  where T: Real,
        S1: BoundingBox<T, D>,
        S2: BoundingBox<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.s1.bounding_box().union(&self.s2.bounding_box())
  }}

impl<T, S1, S2> Lipschitz<T> for SmoothMin<T, S1, S2>
  where T: Float,
        S1: Lipschitz<T>,
        S2: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.s1.lipschitz().max(self.s2.lipschitz())
  }}
