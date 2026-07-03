#![allow(non_upper_case_globals)]
use {
  super::{Shape, BoundingBox, WorldSpace, Translation},
  crate::sdf::{SDF, Lipschitz, Union},
  euclid::{Box2D, Point2D, Vector2D as V2},
  num_traits::{Float, Signed, FloatConst},
  std::marker::PhantomData
};

fn clamp<T: Float>(mut x: T, min: T, max: T) -> T {
  if x < min { x = min; }
  if x > max { x = max; }
  x
}

/// Unit circle.
///
/// `sdf(p) = |p| - 1` — the exact signed distance, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Circle;

impl<T: Float> BoundingBox<T> for Circle {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl <T: Float> SDF<T> for Circle {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    pixel.to_vector().length() - T::one()
  }
}

/// Rectangle with center at `[0, 0]`.
///
/// With `q = |p| - size/2` (componentwise): `sdf(p) = |max(q, 0)| + min(max(q.x, q.y), 0)`
/// — the exact signed distance to the box, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Rect<T, S> {
  pub size: Point2D<T, S>
}

impl<T: Float> BoundingBox<T> for Rect<T, WorldSpace> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let two = T::one() + T::one();
    Box2D::new(
      -self.size / two,
      self.size / two
    )}}

impl<T> SDF<T> for Rect<T, WorldSpace>
  where T: Float + Signed
{
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let two = T::one() + T::one();
    let dist = pixel.to_vector().abs() - (self.size.to_vector() / two);
    let outside_dist = dist
      .max(V2::splat(T::zero()))
      .length();
    let inside_dist = dist.x
      .max(dist.y)
      .min(T::zero());
    outside_dist + inside_dist
  }}

/// Line segment from `a` to `b` with round caps (a capsule).
///
/// `sdf(p) = dist(p, [a, b]) - thickness/2`, where `dist` projects `p` onto the
/// segment with a clamped parameter — the exact signed distance, hence
/// 1-Lipschitz. Degenerate for `a = b` (0/0).
#[derive(Debug, Copy, Clone)]
pub struct Line<T> {
  pub a: Point2D<T, WorldSpace>,
  pub b: Point2D<T, WorldSpace>,
  pub thickness: T,
}

impl<T: Float> BoundingBox<T> for Line<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let two = T::one() + T::one();
    let ret = Box2D::from_points([self.a, self.b]);
    let t = V2::splat(self.thickness / two);
    Box2D::new(ret.min - t, ret.max + t)
  }}

impl<T: Float> SDF<T> for Line<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let ba = self.b - self.a;
    let pa = pixel - self.a;
    let h = clamp(pa.dot(ba) / ba.dot(ba), T::zero(), T::one());
    (pa - ba * h).length() - self.thickness / (T::one() + T::one())
  }
}

/// Regular polygon with N sides, inscribed in a unit circle. Partially evaluated at compile-time.
///
/// `sdf(p) = max_i (p · n_i) - cos(π/N)` over the `N` edge normals — the field of
/// the intersection of half-planes: exact inside and beside the edges, an
/// *underestimate* beyond the vertices (distance to the edge line, not to the
/// vertex). A max of unit-gradient linear fields, hence 1-Lipschitz.
/// Degenerate for `N < 3`.
#[derive(Debug, Copy, Clone)]
pub struct NGonC<const N: usize>;

impl<T: Float, const N: usize> BoundingBox<T> for NGonC<N> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl<T: Float + FloatConst, const N: usize> SDF<T> for NGonC<N> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let p5 = T::one() / (T::one() + T::one());
    let n = T::from(N).unwrap();
    let angle = pixel.y.atan2(pixel.x) + T::FRAC_PI_2();
    let split = T::TAU() / n;
    let r = (T::PI() / n).cos();
    pixel.to_vector().length() * (split * (angle / split + p5).floor() - angle).cos() - r
  }
}

/// Regular polygon with N sides, inscribed in a unit circle. Evaluated at runtime.
///
/// Same field as [`NGonC`]: `max_i (p · n_i) - cos(π/n)` — exact inside,
/// underestimates beyond the vertices; 1-Lipschitz. Degenerate for `n < 3`.
#[derive(Debug, Copy, Clone)]
pub struct NGonR {
  pub n: u64
}

impl<T: Float> BoundingBox<T> for NGonR {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl<T: Float + FloatConst> SDF<T> for NGonR {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let p5 = T::one() / (T::one() + T::one());
    let n = T::from(self.n).unwrap();
    let angle = pixel.y.atan2(pixel.x) + T::FRAC_PI_2();
    let split = T::TAU() / n;
    let r = (T::PI() / n).cos();
    pixel.to_vector().length() * (split * (angle / split + p5).floor() - angle).cos() - r
  }
}

/// N-pointed regular star polygon, inscribed in a unit circle.
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

impl<T: Float> BoundingBox<T> for Star<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl<T: Float + FloatConst> SDF<T> for Star<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let module = |x: T, y: T| x - y * (x / y).floor();
    let n = T::from(self.n).unwrap();
    let an = T::PI() / n;
    let en = T::PI()  / self.m;
    let acs = V2::<_, WorldSpace>::new(an.cos(), an.sin());
    let ecs = V2::new(en.cos(), en.sin());

    let bn = module(pixel.x.atan2(pixel.y), (T::one() + T::one()) * an) - an;
    let mut p = V2::new(bn.cos(), bn.sin().abs())
      * pixel.to_vector().length()
      - acs;
    p += ecs * clamp(-p.dot(ecs), T::zero(), acs.y / ecs.y);
    p.length() * p.x.signum()
  }
}

/// Crescent moon; `phase` in `-1..=1`.
///
/// A unit circle minus a unit circle offset by `d = 2·phase`. Near the cusps
/// the field is the distance to the corner point; elsewhere
/// `sdf(p) = max(|p| - 1, 1 - |p - (d, 0)|)` — together the exact signed
/// distance, hence 1-Lipschitz. At `phase = 0` the crescent degenerates to its
/// boundary circle (`sdf = ||p| - 1|`, an empty-interior shape).
#[derive(Debug, Copy, Clone)]
pub struct Moon<T> {
  pub phase: T
}

impl<T: Float> BoundingBox<T> for Moon<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl<T: Float> SDF<T> for Moon<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let two = T::one() + T::one();
    let pixel = V2::<_, WorldSpace>::new(pixel.x, pixel.y.abs());
    let d = self.phase * two;
    // algebraically d²/2d; written directly to avoid 0/0 at phase = 0
    let a = d / two;
    let b = (T::one() - a * a).max(T::zero()).sqrt();

    if d * (pixel.x * b - pixel.y * a) > d * d * (b - pixel.y).max(T::zero()) {
      (pixel - V2::new(a, b)).length()
    } else {
      (pixel.length() - T::one()).max(
        -((pixel - V2::new(d, T::zero())).length() - T::one())
      )
    }
  }
}

/// A shard: rhombus with half-diagonals `(width, 1)`.
///
/// `p` is folded into the first quadrant, then measured against the single
/// edge segment, signed by the side of the edge line:
/// `sdf(p) = ±|q - proj(q)|` — the exact signed distance, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Kakera<T> {
  pub width: T
}

impl<T: Float> BoundingBox<T> for Kakera<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::new(-self.width, -T::one()),
      Point2D::new(self.width, T::one())
    )}}

impl<T: Float + Signed> SDF<T> for Kakera<T> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let two = T::one() + T::one();
    let ndot = |a: V2<_, _,>, b: V2<_, _,>| a.x*b.x - a.y*b.y;
    let b = V2::new(self.width, T::one());
    let q = pixel.to_vector().abs();
    let mut h = (-two * ndot(q, b) + ndot(b, b)) / b.dot(b);
    h = clamp(h, -T::one(), T::one());
    let d = (q - V2::new(T::one() - h, T::one() + h).component_mul(b) / two).length();
    d * (q.x * b.y + q.y * b.x - b.x * b.y).signum()
  }
}

/// Axis-aligned cross: arm half-length 1, arm half-width `thickness`.
///
/// `p` is folded into the octant `y ≤ x` of the first quadrant, reducing the
/// shape to one L-corner measured exactly (outside: distance to the corner
/// box edge; inside: negative distance to the nearest arm side):
/// exact signed distance, hence 1-Lipschitz. Assumes `thickness ≤ 1`.
#[derive(Debug, Copy, Clone)]
pub struct Cross<T> {
  pub thickness: T
}

impl<T: Float> BoundingBox<T> for Cross<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl<T: Float + Signed> SDF<T> for Cross<T>  {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let mut pixel = pixel.to_vector().abs();
    pixel = if pixel.y > pixel.x { pixel.yx() } else { pixel };
    let q = pixel - V2::new(T::one(), self.thickness);
    let k = q.x.max(q.y);
    let w = if k > T::zero() { q } else { V2::new(self.thickness - pixel.x, -k) };
    k.signum() * w.max(V2::splat(T::zero())).length()
  }
}

/// Annulus: unit circle with a concentric hole of radius `inner_r`.
///
/// `sdf(p) = max(|p| - 1, inner_r - |p|)` — the boolean subtraction of two
/// concentric circles, which for an annulus happens to be the exact signed
/// distance everywhere; 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Ring<T> {
  pub inner_r: T
}

impl<T: Float> BoundingBox<T> for Ring<T> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::splat(-T::one()),
      Point2D::splat(T::one())
    )}}

impl<T: Float> SDF<T> for Ring<T>  {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    Shape::<T>::subtraction(Circle,Circle.scale(self.inner_r))
      .sdf(pixel)
  }
}

/// Arbitrary simple polygon given by its vertices.
///
/// `sdf(p) = s · min_i dist(p, e_i)` — the minimum of exact distances to the
/// edge segments, signed by winding-crossing parity: the exact signed distance
/// for simple polygons, hence 1-Lipschitz. An empty vertex list yields the
/// constant "no shape" field (`L = 0`).
#[derive(Debug, Copy, Clone)]
pub struct Polygon<T> {
  pub vertices: T
}

impl<T, U> BoundingBox<T> for Polygon<U>
  where T: Float,
        U: AsRef<[Point2D<T, WorldSpace>]> {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::from_points(self.vertices.as_ref())
  }}

impl<T, U> SDF<T> for Polygon<U>
  where T: Float,
        U: AsRef<[Point2D<T, WorldSpace>]> {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let v = self.vertices.as_ref();
    let mut d = match v.get(0) {
      Some(&v) => (pixel - v).dot(pixel - v),
      None => return T::max_value() / (T::one() + T::one())
    };
    let mut s = T::one();
    let n = v.len();
    (0..n).zip(std::iter::once(n - 1).chain(0..n - 1))
      .for_each(|(i, j)| {
        let e = v[j] - v[i];
        let w = pixel - v[i];
        let b = w - e * clamp(w.dot(e) / e.dot(e), T::zero(), T::one());
        d = d.min(b.dot(b));
        let c = euclid::BoolVector3D {
          x: pixel.y >= v[i].y,
          y: pixel.y < v[j].y,
          z: e.x * w.y > e.y * w.x
        };
        if c.all() || c.none() {
          s = s.neg();
        }
      });
    s * d.sqrt()
  }
}

/// `= Rect { size: [2.0, 2.0] }` — exact SDF, 1-Lipschitz (see [`Rect`]).
#[derive(Debug, Copy, Clone)]
pub struct Square;

impl<T: Float> BoundingBox<T> for Square {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let two = T::one() + T::one();
    Rect { size: Point2D::splat(two) }.bounding_box()}}

impl<T> SDF<T> for Square
  where T: Float + Signed {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let two = T::one() + T::one();
    Rect { size: Point2D::splat(two) }.sdf(pixel)
  }}

/// `= Star { n: 5, m: 10.0 / 3.0 }` — exact SDF, 1-Lipschitz (see [`Star`]).
#[derive(Debug, Copy, Clone)]
pub struct Pentagram;

impl<T: Float> BoundingBox<T> for Pentagram {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let two = T::one() + T::one();
    let three = two + T::one();
    let ten = three * three + T::one();
    Star { n: 5, m: ten / three }.bounding_box()}}

impl<T> SDF<T> for Pentagram
  where T: Float + FloatConst {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let two = T::one() + T::one();
    let three = two + T::one();
    let ten = three * three + T::one();
    Star { n: 5, m: ten / three }.sdf(pixel)
  }}

/// `= Star { n: 6, m: 3.0 }` — exact SDF, 1-Lipschitz (see [`Star`]).
#[derive(Debug, Copy, Clone)]
pub struct Hexagram;

impl<T: Float> BoundingBox<T> for Hexagram {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    let three = T::one() + T::one() + T::one();
    Star { n: 6, m: three }.bounding_box()}}

impl<T> SDF<T> for Hexagram
  where T: Float + FloatConst {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
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

/// Latin cross: the union of two rectangles.
///
/// `min` of two exact box fields — exact in free space, underestimates the
/// interior depth where the rectangles overlap; 1-Lipschitz (see
/// [`Shape::union`]).
pub static HolyCross: Union <
  Rect<f64, WorldSpace> ,
  Translation < Rect<f64, WorldSpace>, f64 >
> = Union {
  s1: Rect { size: Point2D { x: 0.4, y: 2.0, _unit: PhantomData::<WorldSpace> } },
  s2: Translation {
    shape: Rect { size: Point2D {  x: 1.432, y: 0.4, _unit: PhantomData::<WorldSpace> } },
    offset: V2 { x: 0.0, y: -0.3, _unit: PhantomData::<WorldSpace> }
  }
};

// Every shape above is either an exact SDF or a 1-Lipschitz underestimate
// assembled from unit-gradient pieces (see each shape's doc), so the honest
// bound is `1`. Combinators propagate these automatically; see
// [`crate::sdf::Lipschitz`].
impl<T: Float> Lipschitz<T> for Circle { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Rect<T, WorldSpace> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Line<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const N: usize> Lipschitz<T> for NGonC<N> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for NGonR { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Star<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Moon<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Kakera<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Cross<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Ring<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U> Lipschitz<T> for Polygon<U> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Square { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Pentagram { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Hexagram { fn lipschitz(&self) -> T { T::one() } }

#[cfg(test)]
mod tests {
  use {
    super::*,
    crate::sdf,
    euclid::Angle,
    rand::prelude::*,
  };

  /// Numerically verify `|f(p) − f(q)| ≤ L·|p − q|` over random global and
  /// tightly-spaced point pairs (the latter stress the local gradient), plus
  /// finiteness everywhere. Every stored primitive relies on an honest
  /// Lipschitz bound: the redundancy test certifies with it (`sdf_geq_everywhere`),
  /// and the D*-pruned insertion walk skips subtrees with it.
  fn check_lipschitz(name: &str, l: f64, f: impl Fn(Point2D<f64, WorldSpace>) -> f64) {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let span = 2.5;
    for i in 0..20000 {
      let p = Point2D::<f64, WorldSpace>::new(
        rng.gen_range(-span..span), rng.gen_range(-span..span));
      let q = if i % 2 == 0 {
        Point2D::new(rng.gen_range(-span..span), rng.gen_range(-span..span))
      } else {
        // short-range pair: within 1e-4..1e-2 of p
        let angle = rng.gen_range(0.0..std::f64::consts::TAU);
        let r = 10f64.powf(rng.gen_range(-4.0..-2.0));
        p + V2::new(angle.cos(), angle.sin()) * r
      };
      let (fp, fq) = (f(p), f(q));
      assert!(fp.is_finite() && fq.is_finite(), "{name}: non-finite field at {p:?} / {q:?}");
      let dist = (p - q).length();
      assert!(
        (fp - fq).abs() <= l * dist * (1.0 + 1e-9) + 1e-12,
        "{name}: Lipschitz bound {l} violated: |f({p:?}) − f({q:?})| = {} > {l}·{dist}",
        (fp - fq).abs()
      );
    }
  }

  #[test] fn lipschitz_shapes() {
    check_lipschitz("Circle", 1.0, |p| Circle.sdf(p));
    check_lipschitz("Rect", 1.0, |p| Rect { size: Point2D::new(1.5, 0.8) }.sdf(p));
    check_lipschitz("Square", 1.0, |p| Square.sdf(p));
    check_lipschitz("Line", 1.0, |p| Line
      { a: Point2D::new(-0.8, -0.3), b: Point2D::new(0.6, 0.9), thickness: 0.2 }.sdf(p));
    check_lipschitz("Triangle", 1.0, |p| NGonC::<3>.sdf(p));
    check_lipschitz("Pentagon", 1.0, |p| NGonC::<5>.sdf(p));
    check_lipschitz("Heptagon", 1.0, |p| NGonC::<7>.sdf(p));
    check_lipschitz("NGonR(5)", 1.0, |p| NGonR { n: 5 }.sdf(p));
    check_lipschitz("Star(5, 10/3)", 1.0, |p| Star { n: 5, m: 10.0 / 3.0 }.sdf(p));
    check_lipschitz("Star(7, 3)", 1.0, |p| Star { n: 7, m: 3.0 }.sdf(p));
    check_lipschitz("Pentagram", 1.0, |p| Pentagram.sdf(p));
    check_lipschitz("Hexagram", 1.0, |p| Hexagram.sdf(p));
    for phase in [-1.0, -0.7, 0.0, 0.3, 1.0] {
      check_lipschitz(&format!("Moon({phase})"), 1.0, |p| Moon { phase }.sdf(p));
    }
    check_lipschitz("Kakera", 1.0, |p| Kakera { width: 0.5 }.sdf(p));
    check_lipschitz("Cross", 1.0, |p| Cross { thickness: 0.3 }.sdf(p));
    check_lipschitz("Ring", 1.0, |p| Ring { inner_r: 0.5 }.sdf(p));
    check_lipschitz("Polygon", 1.0, |p| Polygon { vertices: [
      Point2D::new(-0.9, -0.5), Point2D::new(0.8, -0.7),
      Point2D::new(0.5, 0.9), Point2D::new(-0.3, 0.4),
    ]}.sdf(p));
    check_lipschitz("HolyCross", 1.0, |p| HolyCross.sdf(p));
    check_lipschitz("boundary_rect", 1.0, sdf::boundary_rect);
  }

  #[test] fn lipschitz_combinators() {
    let star = || Star { n: 5, m: 10.0 / 3.0 };
    // translate / rotate / scale: precomposition with an isometry (or a
    // similarity whose value re-scale cancels the coordinate re-scale)
    // preserves the constant exactly
    check_lipschitz("translate", 1.0, |p| star().translate(V2::new(0.3, -0.2)).sdf(p));
    check_lipschitz("rotate", 1.0, |p| star().rotate(Angle::degrees(37.0)).sdf(p));
    check_lipschitz("scale(0.35)", 1.0, |p| star().scale(0.35).sdf(p));
    check_lipschitz("scale(2.5)", 1.0, |p| star().scale(2.5).sdf(p));
    // boolean ops: min/max of L-Lipschitz fields is max(L₁, L₂)-Lipschitz
    check_lipschitz("union", 1.0, |p| Circle.translate(V2::new(0.4, 0.0))
      .union(Square.scale(0.6)).sdf(p));
    check_lipschitz("subtraction", 1.0, |p| Shape::<f64>::subtraction(
      Square, Circle.scale(0.7).translate(V2::new(0.5, 0.5))).sdf(p));
    check_lipschitz("intersection", 1.0, |p| Shape::<f64>::intersection(
      Circle, Square.rotate(Angle::degrees(20.0))).sdf(p));
    // smooth_min: ∇ = w·∇f + (1−w)·∇g with w ∈ (0, 1) — a convex combination
    check_lipschitz("smooth_min", 1.0, |p| Circle.translate(V2::new(-0.4, 0.1))
      .smooth_min(Square.scale(0.5).translate(V2::new(0.5, 0.0)), 32.0).sdf(p));
  }

  // the trait-derived constants agree with the numerically-verified bounds
  #[test] fn lipschitz_trait() {
    assert_eq!(Lipschitz::<f64>::lipschitz(&Circle), 1.0);
    assert_eq!(Lipschitz::<f64>::lipschitz(&HolyCross), 1.0);
    let chain = Star { n: 5, m: 10.0 / 3.0 }
      .scale(0.35)
      .rotate(Angle::degrees(37.0))
      .translate(V2::new(0.3, -0.2));
    assert_eq!(chain.lipschitz(), 1.0);
    let boolean = Shape::<f64>::union(Circle, Shape::<f64>::subtraction(Square, Ring { inner_r: 0.5 }));
    assert_eq!(boolean.lipschitz(), 1.0);
  }
}