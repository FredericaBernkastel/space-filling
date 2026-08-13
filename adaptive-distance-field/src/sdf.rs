use {
  crate::{
    geometry::{
      Point, Vector, Real, Aabb, Rotation, Scale, Translation,
      BoundingBox, VectorExt, P2, V2
    },
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
/// ([`sdf_geq_everywhere`](crate::adf)), and the `D*`-pruned insertion
/// walk skips subtrees with it — so it must be *honest*: an understated
/// constant can corrupt the field, an overstated one merely costs pruning
/// power. Exact SDFs return `1`; distance *estimators* declare their own
/// bound; combinators propagate `max` over their operands.
///
/// [`Primitive::from_shape`](crate::adf::Primitive::from_shape) derives the
/// stored bound from this trait automatically.
pub trait Lipschitz<T> {
  fn lipschitz(&self) -> T;
}

impl <S, T, const D: usize> SDF<T, D> for Translation<S, T, D>
  where S: SDF<T, D>,
        T: Real {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    self.shape.sdf(pixel - self.offset)
  }
}

impl <S, T, const D: usize> SDF<T, D> for Rotation<S, T, D>
  where S: SDF<T, D> + BoundingBox<T, D>,
        T: Real {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let pivot = self.shape.bounding_box().center();
    let pixel = self.rotation.matrix() * (pixel - pivot) + pivot.coords;
    self.shape.sdf(Point::from(pixel))
  }
}

impl <S, T, const D: usize> SDF<T, D> for Scale<S, T>
  where S: SDF<T, D> + BoundingBox<T, D>,
        T: Real {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let c = self.shape.bounding_box().center();
    let pixel = (pixel - c) / self.scale + c.coords;
    self.shape.sdf(Point::from(pixel)) * self.scale
  }
}

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/boundary_rect.avif" alt="the walls of the unit square, positive inside" width="200" style="display:block; margin: 0.3em 0 0.9em"> Distance to the walls of the unit hypercube `[0, 1]^D`, positive inside.
///
/// The negation of the exact cube SDF centred on `(½, …, ½)`, written out rather
/// than composed from a shape so that it stands alone: this is the canonical
/// seed for an [`ADF`](crate::adf::ADF).
/// Negation preserves the constant — 1-Lipschitz.
pub fn boundary_rect<T: Real, const D: usize>(pixel: Point<T, D>) -> T {
  let p5 = T::one() / (T::one() + T::one());
  // distance from the cube's centre, per axis, minus the half-extent
  let q = pixel.coords.map(|x| (x - p5).abs() - p5);
  let outside = q.map(|x| x.max(T::zero())).length();
  let inside = q.iter()
    .fold(T::neg_infinity(), |a, &b| a.max(b))
    .min(T::zero());
  -(outside + inside)
}

/// Distance to the walls of an arbitrary box, positive inside — the seed for an
/// [`ADF`](crate::adf::ADF) whose domain is not the unit cube.
///
/// [`boundary_rect`] is this at [`Aabb::unit`], and the two agree bit for bit
/// there. Anisotropic domains are how weights are expressed: a box of extent
/// `γ` makes axis `i` matter in proportion to `γᵢ`, which is what
/// [`Widest`](crate::adf::tree::Widest) orders its cuts by.
///
/// Negation preserves the constant — 1-Lipschitz.
pub fn boundary_box<T: Real + Send + Sync + 'static, const D: usize>(
  domain: Aabb<T, D>,
) -> impl Fn(Point<T, D>) -> T + Copy + Send + Sync + 'static {
  let two = T::one() + T::one();
  let centre = domain.center();
  let half = domain.size() / two;
  move |pixel: Point<T, D>| {
    // distance from the box's centre, per axis, minus that axis's half-extent
    let q = (pixel - centre).zip_map(&half, |x, h| x.abs() - h);
    let outside = q.map(|x| x.max(T::zero())).length();
    let inside = q.iter()
      .fold(T::neg_infinity(), |a, &b| a.max(b))
      .min(T::zero());
    -(outside + inside)
  }
}

/// Union of two SDFs: `min(s1, s2)`.
/// See [`Combinator::union`](crate::geometry::Combinator::union); `max(L₁, L₂)`-Lipschitz.
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
    self.s1.bounding_box().merge(&self.s2.bounding_box())
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
/// See [`Combinator::subtraction`](crate::geometry::Combinator::subtraction); `max(L₁, L₂)`-Lipschitz.
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
    self.s1.bounding_box().merge(&self.s2.bounding_box())
  }}

impl<T, S1, S2> Lipschitz<T> for Subtraction<S1, S2>
  where T: Float,
        S1: Lipschitz<T>,
        S2: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.s1.lipschitz().max(self.s2.lipschitz())
  }}

/// Intersection of two SDFs: `max(s1, s2)`.
/// See [`Combinator::intersection`](crate::geometry::Combinator::intersection); `max(L₁, L₂)`-Lipschitz.
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
      .clip(&self.s2.bounding_box())
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
/// See [`Combinator::smooth_min`](crate::geometry::Combinator::smooth_min); `max(L₁, L₂)`-Lipschitz (its gradient is a
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
    self.s1.bounding_box().merge(&self.s2.bounding_box())
  }}

impl<T, S1, S2> Lipschitz<T> for SmoothMin<T, S1, S2>
  where T: Float,
        S1: Lipschitz<T>,
        S2: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.s1.lipschitz().max(self.s2.lipschitz())
  }}

/// Hollow shape: the `half_width`-thick shell around another shape's boundary.
/// See [`Combinator::shell`](crate::geometry::Combinator::shell); `L`-Lipschitz, exactly as its operand.
#[derive(Clone, Copy, Debug)]
pub struct Shell<S, T> {
  pub shape: S,
  pub half_width: T
}

impl<T, S, const D: usize> SDF<T, D> for Shell<S, T>
  where T: Real,
        S: SDF<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    self.shape.sdf(pixel).abs() - self.half_width
  }}

impl<T, S, const D: usize> BoundingBox<T, D> for Shell<S, T>
  where T: Real,
        S: BoundingBox<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    // the shell reaches `half_width` outward from the original surface
    self.shape.bounding_box().inflate(self.half_width)
  }}

impl<T, S> Lipschitz<T> for Shell<S, T>
  where T: Float,
        S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }}

/// Uniformly grown (or shrunk) shape — rounded corners for free.
/// See [`Combinator::offset`](crate::geometry::Combinator::offset); `L`-Lipschitz, exactly as its operand.
#[derive(Clone, Copy, Debug)]
pub struct Offset<S, T> {
  pub shape: S,
  pub radius: T
}

impl<T, S, const D: usize> SDF<T, D> for Offset<S, T>
  where T: Real,
        S: SDF<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    self.shape.sdf(pixel) - self.radius
  }}

impl<T, S, const D: usize> BoundingBox<T, D> for Offset<S, T>
  where T: Real,
        S: BoundingBox<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    self.shape.bounding_box().inflate(self.radius)
  }}

impl<T, S> Lipschitz<T> for Offset<S, T>
  where T: Float,
        S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }}

/// A `(D−1)`-dimensional shape given thickness along the last axis — a prism.
/// See [`Combinator::extrude`](crate::geometry::Combinator::extrude).
///
/// Implemented for base/target dimension pairs up to `(5, 6)`; `max(L, 1)`-Lipschitz.
#[derive(Clone, Copy, Debug)]
pub struct Extrude<S, T> {
  pub shape: S,
  pub half_height: T
}

macro_rules! impl_extrude {($(($base:literal, $dim:literal))*) => {$(
  impl<T, S> SDF<T, $dim> for Extrude<S, T>
    where T: Real,
          S: SDF<T, $base> {
    fn sdf(&self, pixel: Point<T, $dim>) -> T {
      // exactly the box construction, with the base field standing in for one
      // of the two axes: exact whenever the base field is
      let base: Point<T, $base> = Point::from(std::array::from_fn(|a| pixel[a]));
      let w = V2::new(
        self.shape.sdf(base),
        pixel[$dim - 1].abs() - self.half_height);
      let outside = V2::new(w.x.max(T::zero()), w.y.max(T::zero())).length();
      let inside = w.x.max(w.y).min(T::zero());
      outside + inside
    }}

  impl<T, S> BoundingBox<T, $dim> for Extrude<S, T>
    where T: Real,
          S: BoundingBox<T, $base> {
    fn bounding_box(&self) -> Aabb<T, $dim> {
      let b = self.shape.bounding_box();
      let (mut min, mut max) = ([T::zero(); $dim], [T::zero(); $dim]);
      for a in 0..$base {
        min[a] = b.min[a];
        max[a] = b.max[a];
      }
      min[$dim - 1] = -self.half_height;
      max[$dim - 1] = self.half_height;
      Aabb::new(Point::from(min), Point::from(max))
    }}
)*}}
impl_extrude!((1, 2) (2, 3) (3, 4) (4, 5) (5, 6));

impl<T, S> Lipschitz<T> for Extrude<S, T>
  where T: Float,
        S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    // the base field and the `|p_last| − h` term act on disjoint axes, so their
    // gradients are orthogonal and the combined bound is the larger of the two
    self.shape.lipschitz().max(T::one())
  }}

/// A 2D profile swept around axis 0 — a solid of revolution, offset `radius`
/// from the axis. See [`Combinator::revolve`](crate::geometry::Combinator::revolve); `L`-Lipschitz, exactly as its operand.
#[derive(Clone, Copy, Debug)]
pub struct Revolve<S, T> {
  pub shape: S,
  pub radius: T
}

impl<T, S, const D: usize> SDF<T, D> for Revolve<S, T>
  where T: Real,
        S: SDF<T, 2> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    // (axial, radial − radius); the reduction is exact and 1-Lipschitz — see
    // `Combinator::revolve`
    let radial = (1..D)
      .fold(T::zero(), |acc, a| acc + pixel[a] * pixel[a])
      .sqrt();
    self.shape.sdf(P2::new(pixel[0], radial - self.radius))
  }}

impl<T, S, const D: usize> BoundingBox<T, D> for Revolve<S, T>
  where T: Real,
        S: BoundingBox<T, 2> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let b = self.shape.bounding_box();
    // the profile's radial span, shifted out by `radius`, sweeps a disc of this
    // radius across every axis but the first
    let reach = (b.min[1] + self.radius).abs()
      .max((b.max[1] + self.radius).abs());
    let (mut min, mut max) = ([-reach; D], [reach; D]);
    min[0] = b.min[0];
    max[0] = b.max[0];
    Aabb::new(Point::from(min), Point::from(max))
  }}

impl<T, S> Lipschitz<T> for Revolve<S, T>
  where T: Float,
        S: Lipschitz<T> {
  fn lipschitz(&self) -> T {
    self.shape.lipschitz()
  }}
