#![allow(non_upper_case_globals)] // `HolyCross`
//! Shapes defined for the **plane only** (`D = 2`).
//!
//! The regular-polygon and
//! star families fold `p` by its polar angle (`atan2` plus an angular sector
//! wrap), which has no direct counterpart in higher dimensions, and [`Polygon`]
//! signs its distance by planar winding-crossing parity over an ordered vertex
//! list. Their N-dimensional relatives are different objects living in
//! [`dn`](super::dn): a convex [`Polytope`](super::dn::Polytope) for the
//! half-space construction, or an extruded prism if what you want is the 2D
//! profile carried along a new axis
//! (`shape.intersection(Hyperrect { .. })`).

use {
  super::clamp,
  crate::{
    geometry::{Aabb, BoundingBox, Hyperrect, Real, Translation, VectorExt, P2, V2},
    sdf::{Lipschitz, Union, SDF},
  },
  num_traits::{Float, FloatConst},
};

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/ngon_c.avif" alt="a regular hexagon" width="200" style="display:block; margin: 0.3em 0 0.9em"> Regular polygon with N sides, inscribed in a unit circle. Partially evaluated at compile-time.
///
/// Note that `N` counts *sides*, not dimensions — `NGonC<6>` is a hexagon in
/// the plane. For the arbitrary-dimension half-space construction see
/// [`Polytope`](super::dn::Polytope).
///
/// `sdf(p) = max_i (p · n_i) - cos(π/N)` over the `N` edge normals — the field of
/// the intersection of half-planes: exact inside and beside the edges, an
/// *underestimate* beyond the vertices (distance to the edge line, not to the
/// vertex). A max of unit-gradient linear fields, hence 1-Lipschitz.
/// Degenerate for `N < 3`.
#[derive(Debug, Copy, Clone)]
pub struct NGonC<const N: usize>;

impl<T: Real, const N: usize> BoundingBox<T, 2> for NGonC<N> {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real + FloatConst, const N: usize> SDF<T, 2> for NGonC<N> {
  fn sdf(&self, pixel: P2<T>) -> T {
    let p5 = T::one() / (T::one() + T::one());
    let n = T::from(N).unwrap();
    let angle = pixel.y.atan2(pixel.x) + T::FRAC_PI_2();
    let split = T::TAU() / n;
    let r = (T::PI() / n).cos();
    pixel.coords.length() * (split * (angle / split + p5).floor() - angle).cos() - r
  }
}

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/ngon_r.avif" alt="a regular pentagon" width="200" style="display:block; margin: 0.3em 0 0.9em"> Regular polygon with N sides, inscribed in a unit circle. Evaluated at runtime.
///
/// Same field as [`NGonC`]: `max_i (p · n_i) - cos(π/n)` — exact inside,
/// underestimates beyond the vertices; 1-Lipschitz. Degenerate for `n < 3`.
#[derive(Debug, Copy, Clone)]
pub struct NGonR {
  pub n: u64
}

impl<T: Real> BoundingBox<T, 2> for NGonR {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real + FloatConst> SDF<T, 2> for NGonR {
  fn sdf(&self, pixel: P2<T>) -> T {
    let p5 = T::one() / (T::one() + T::one());
    let n = T::from(self.n).unwrap();
    let angle = pixel.y.atan2(pixel.x) + T::FRAC_PI_2();
    let split = T::TAU() / n;
    let r = (T::PI() / n).cos();
    pixel.coords.length() * (split * (angle / split + p5).floor() - angle).cos() - r
  }
}

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/star.avif" alt="a five-pointed star" width="200" style="display:block; margin: 0.3em 0 0.9em"> N-pointed regular star polygon, inscribed in a unit circle.
/// `m` is density, must be between `2..=n`.
///
/// `p` is folded into one angular sector (a piecewise isometry, continuous by
/// the shape's symmetry), then measured against that sector's edge segment:
/// `sdf(p) = ±|p' - proj(p')|` — the exact signed distance, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Star<T> {
  pub n: u64,
  pub m: T
}

impl<T: Real> BoundingBox<T, 2> for Star<T> {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real + FloatConst> SDF<T, 2> for Star<T> {
  fn sdf(&self, pixel: P2<T>) -> T {
    let module = |x: T, y: T| x - y * (x / y).floor();
    let n = T::from(self.n).unwrap();
    let an = T::PI() / n;
    let en = T::PI() / self.m;
    let acs = V2::new(an.cos(), an.sin());
    let ecs = V2::new(en.cos(), en.sin());

    let bn = module(pixel.x.atan2(pixel.y), (T::one() + T::one()) * an) - an;
    let mut p = V2::new(bn.cos(), bn.sin().abs())
      * pixel.coords.length()
      - acs;
    p += ecs * clamp(-p.dot(&ecs), T::zero(), acs.y / ecs.y);
    p.length() * p.x.signum()
  }
}

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/polygon.avif" alt="an irregular quadrilateral" width="200" style="display:block; margin: 0.3em 0 0.9em"> Arbitrary simple polygon given by its vertices — non-convex allowed.
///
/// `sdf(p) = s · min_i dist(p, e_i)` — the minimum of exact distances to the
/// edge segments, signed by winding-crossing parity: the exact signed distance
/// for simple polygons, hence 1-Lipschitz. An empty vertex list yields the
/// constant "no shape" field (`L = 0`).
///
/// The winding-parity sign is what pins this to the plane; the N-dimensional
/// analogue is the convex [`Polytope`](super::dn::Polytope).
#[derive(Debug, Copy, Clone)]
pub struct Polygon<T> {
  pub vertices: T
}

impl<T, U> BoundingBox<T, 2> for Polygon<U>
  where T: Real,
        U: AsRef<[P2<T>]> {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::from_points(self.vertices.as_ref().iter().copied())
  }}

impl<T, U> SDF<T, 2> for Polygon<U>
  where T: Real,
        U: AsRef<[P2<T>]> {
  fn sdf(&self, pixel: P2<T>) -> T {
    let v = self.vertices.as_ref();
    let mut d = match v.get(0) {
      Some(&v) => (pixel - v).dot(&(pixel - v)),
      None => return T::max_value() / (T::one() + T::one())
    };
    let mut s = T::one();
    let n = v.len();
    (0..n).zip(std::iter::once(n - 1).chain(0..n - 1))
      .for_each(|(i, j)| {
        let e = v[j] - v[i];
        let w = pixel - v[i];
        let b = w - e * clamp(w.dot(&e) / e.dot(&e), T::zero(), T::one());
        d = d.min(b.dot(&b));
        let c = [
          pixel.y >= v[i].y,
          pixel.y < v[j].y,
          e.x * w.y > e.y * w.x
        ];
        if c == [true; 3] || c == [false; 3] {
          s = -s;
        }
      });
    s * d.sqrt()
  }
}

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/pentagram.avif" alt="a pentagram" width="200" style="display:block; margin: 0.3em 0 0.9em"> `= Star { n: 5, m: 10.0 / 3.0 }` — exact SDF, 1-Lipschitz (see [`Star`]).
#[derive(Debug, Copy, Clone)]
pub struct Pentagram;

impl<T: Real> BoundingBox<T, 2> for Pentagram {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T> SDF<T, 2> for Pentagram
  where T: Real + FloatConst {
  fn sdf(&self, pixel: P2<T>) -> T {
    let two = T::one() + T::one();
    let three = two + T::one();
    let ten = three * three + T::one();
    Star { n: 5, m: ten / three }.sdf(pixel)
  }}

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/hexagram.avif" alt="a hexagram" width="200" style="display:block; margin: 0.3em 0 0.9em"> `= Star { n: 6, m: 3.0 }` — exact SDF, 1-Lipschitz (see [`Star`]).
#[derive(Debug, Copy, Clone)]
pub struct Hexagram;

impl<T: Real> BoundingBox<T, 2> for Hexagram {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T> SDF<T, 2> for Hexagram
  where T: Real + FloatConst {
  fn sdf(&self, pixel: P2<T>) -> T {
    let three = T::one() + T::one() + T::one();
    Star { n: 6, m: three }.sdf(pixel)
  }}

/// `= NGonC::<3>`
pub type Triangle = NGonC<3>;

/// `= NGonC::<5>`
pub type Pentagon = NGonC<5>;

/// `= NGonC::<6>`
pub type Hexagon = NGonC<6>;

/// `= NGonC::<7>`
pub type Heptagon = NGonC<7>;

/// `= NGonC::<8>`
pub type Octagon = NGonC<8>;

/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/holy_cross.avif" alt="a cross with a ring around its crossing" width="200" style="display:block; margin: 0.3em 0 0.9em"> Latin cross: the union of two rectangles.
///
/// `min` of two exact box fields — exact in free space, underestimates the
/// interior depth where the rectangles overlap; 1-Lipschitz (see
/// [`Combinator::union`](crate::geometry::Combinator::union)). Pinned to `f64` and the
/// plane because a `static` needs a concrete type; the [`Hyperrect`]s
/// underneath it are dimension-generic.
pub static HolyCross: Union <
  Hyperrect<f64, 2>,
  Translation < Hyperrect<f64, 2>, f64, 2 >
> = Union {
  s1: Hyperrect { size: nalgebra::vector![0.4, 2.0] },
  s2: Translation {
    shape: Hyperrect { size: nalgebra::vector![1.432, 0.4] },
    offset: nalgebra::vector![0.0, -0.3]
  }
};

// As in [`dn`](super::dn): every shape here is an exact SDF or a 1-Lipschitz
// underestimate assembled from unit-gradient pieces, so the honest bound is `1`.
impl<T: Float, const N: usize> Lipschitz<T> for NGonC<N> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for NGonR { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Star<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U> Lipschitz<T> for Polygon<U> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Pentagram { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Hexagram { fn lipschitz(&self) -> T { T::one() } }
