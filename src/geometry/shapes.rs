#![allow(non_upper_case_globals)]
use {
  super::{BoundingBox, Translation, Aabb, Point, Vector, P2, V2, Real, VectorExt},
  crate::sdf::{SDF, Lipschitz, Union},
  num_traits::{Float, FloatConst},
};
#[cfg(test)] use super::Shape;

fn clamp<T: Float>(mut x: T, min: T, max: T) -> T {
  if x < min { x = min; }
  if x > max { x = max; }
  x
}

/// Unit sphere (a circle in 2D — the default; `Circle::<3>` for a ball, …).
///
/// The dimension lives on the type so that combinator chains starting from a
/// bare `Circle` stay inferable; it defaults to 2 in type positions.
///
/// `sdf(p) = |p| - 1` — the exact signed distance in any dimension, hence
/// 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Circle<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Circle<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl <T: Real, const D: usize> SDF<T, D> for Circle<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    pixel.coords.length() - T::one()
  }
}

/// Axis-aligned box with center at the origin (a rectangle in 2D).
///
/// With `q = |p| - size/2` (componentwise): `sdf(p) = |max(q, 0)| + min(max_a q_a, 0)`
/// — the exact signed distance to the box in any dimension, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Rect<T, const D: usize> {
  pub size: Vector<T, D>
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Rect<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let two = T::one() + T::one();
    Aabb {
      min: Point::from(self.size / -two),
      max: Point::from(self.size / two),
    }}}

impl<T: Real, const D: usize> SDF<T, D> for Rect<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    let dist = pixel.coords.abs() - self.size / two;
    let outside_dist = dist
      .map(|x| x.max(T::zero()))
      .length();
    let inside_dist = dist.iter()
      .fold(T::neg_infinity(), |a, &b| a.max(b))
      .min(T::zero());
    outside_dist + inside_dist
  }}

/// Line segment from `a` to `b` with round caps (a capsule), any dimension.
///
/// `sdf(p) = dist(p, [a, b]) - thickness/2`, where `dist` projects `p` onto the
/// segment with a clamped parameter — the exact signed distance, hence
/// 1-Lipschitz. Degenerate for `a = b` (0/0).
#[derive(Debug, Copy, Clone)]
pub struct Line<T: nalgebra::Scalar, const D: usize> {
  pub a: Point<T, D>,
  pub b: Point<T, D>,
  pub thickness: T,
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Line<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let two = T::one() + T::one();
    let ret = Aabb::from_points([self.a, self.b]);
    let t = Vector::repeat(self.thickness / two);
    Aabb::new(ret.min - t, ret.max + t)
  }}

impl<T: Real, const D: usize> SDF<T, D> for Line<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let ba = self.b - self.a;
    let pa = pixel - self.a;
    let h = clamp(pa.dot(&ba) / ba.dot(&ba), T::zero(), T::one());
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

/// Regular polygon with N sides, inscribed in a unit circle. Evaluated at runtime.
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

impl<T: Real> BoundingBox<T, 2> for Moon<T> {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real> SDF<T, 2> for Moon<T> {
  fn sdf(&self, pixel: P2<T>) -> T {
    let two = T::one() + T::one();
    let pixel = V2::new(pixel.x, pixel.y.abs());
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

impl<T: Real> BoundingBox<T, 2> for Kakera<T> {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::new(
      Point::from([-self.width, -T::one()]),
      Point::from([self.width, T::one()])
    )}}

impl<T: Real> SDF<T, 2> for Kakera<T> {
  fn sdf(&self, pixel: P2<T>) -> T {
    let two = T::one() + T::one();
    let ndot = |a: V2<T>, b: V2<T>| a.x * b.x - a.y * b.y;
    let b = V2::new(self.width, T::one());
    let q = pixel.coords.abs();
    let mut h = (-two * ndot(q, b) + ndot(b, b)) / b.dot(&b);
    h = clamp(h, -T::one(), T::one());
    let d = (q - V2::new(T::one() - h, T::one() + h).component_mul(&b) / two).length();
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

impl<T: Real> BoundingBox<T, 2> for Cross<T> {
  fn bounding_box(&self) -> Aabb<T, 2> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real> SDF<T, 2> for Cross<T> {
  fn sdf(&self, pixel: P2<T>) -> T {
    let mut pixel = pixel.coords.abs();
    pixel = if pixel.y > pixel.x { V2::new(pixel.y, pixel.x) } else { pixel };
    let q = pixel - V2::new(T::one(), self.thickness);
    let k = q.x.max(q.y);
    let w = if k > T::zero() { q } else { V2::new(self.thickness - pixel.x, -k) };
    k.signum() * w.map(|x| x.max(T::zero())).length()
  }
}

/// Annulus: unit sphere with a concentric hole of radius `inner_r` (any
/// dimension; defaults to 2).
///
/// `sdf(p) = max(|p| - 1, inner_r - |p|)` — the boolean subtraction of two
/// concentric spheres, which for an annulus happens to be the exact signed
/// distance everywhere; 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Ring<T, const D: usize = 2> {
  pub inner_r: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Ring<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Ring<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let outer = pixel.coords.length() - T::one();
    let inner = pixel.coords.length() - self.inner_r;
    outer.max(-inner)
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

/// `= Rect { size: [2.0; D] }` — exact SDF, 1-Lipschitz (see [`Rect`]).
/// Any dimension; defaults to 2.
#[derive(Debug, Copy, Clone)]
pub struct Square<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Square<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Square<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    Rect { size: Vector::repeat(two) }.sdf(pixel)
  }}

/// `= Star { n: 5, m: 10.0 / 3.0 }` — exact SDF, 1-Lipschitz (see [`Star`]).
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

/// `= Star { n: 6, m: 3.0 }` — exact SDF, 1-Lipschitz (see [`Star`]).
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

/// Latin cross: the union of two rectangles.
///
/// `min` of two exact box fields — exact in free space, underestimates the
/// interior depth where the rectangles overlap; 1-Lipschitz (see
/// [`Shape::union`](super::Shape::union)).
pub static HolyCross: Union <
  Rect<f64, 2>,
  Translation < Rect<f64, 2>, f64, 2 >
> = Union {
  s1: Rect { size: nalgebra::vector![0.4, 2.0] },
  s2: Translation {
    shape: Rect { size: nalgebra::vector![1.432, 0.4] },
    offset: nalgebra::vector![0.0, -0.3]
  }
};

// Every shape above is either an exact SDF or a 1-Lipschitz underestimate
// assembled from unit-gradient pieces (see each shape's doc), so the honest
// bound is `1`. Combinators propagate these automatically; see
// [`crate::sdf::Lipschitz`].
impl<T: Float, const D: usize> Lipschitz<T> for Circle<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Rect<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float + nalgebra::Scalar, const D: usize> Lipschitz<T> for Line<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const N: usize> Lipschitz<T> for NGonC<N> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for NGonR { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Star<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Moon<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Kakera<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Cross<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Ring<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U> Lipschitz<T> for Polygon<U> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Square<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Pentagram { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Hexagram { fn lipschitz(&self) -> T { T::one() } }

#[cfg(test)]
mod tests {
  use {
    super::*,
    crate::sdf,
    nalgebra::Rotation2,
    rand::prelude::*,
  };

  /// Numerically verify `|f(p) − f(q)| ≤ L·|p − q|` over random global and
  /// tightly-spaced point pairs (the latter stress the local gradient), plus
  /// finiteness everywhere. Every stored primitive relies on an honest
  /// Lipschitz bound: the redundancy test certifies with it (`sdf_geq_everywhere`),
  /// and the D*-pruned insertion walk skips subtrees with it.
  fn check_lipschitz(name: &str, l: f64, f: impl Fn(P2<f64>) -> f64) {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let span = 2.5;
    for i in 0..20000 {
      let p = P2::new(
        rng.random_range(-span..span), rng.random_range(-span..span));
      let q = if i % 2 == 0 {
        P2::new(rng.random_range(-span..span), rng.random_range(-span..span))
      } else {
        // short-range pair: within 1e-4..1e-2 of p
        let angle = rng.random_range(0.0..std::f64::consts::TAU);
        let r = 10f64.powf(rng.random_range(-4.0..-2.0));
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
    check_lipschitz("Rect", 1.0, |p| Rect { size: V2::new(1.5, 0.8) }.sdf(p));
    check_lipschitz("Square", 1.0, |p| Square.sdf(p));
    check_lipschitz("Line", 1.0, |p| Line
      { a: P2::new(-0.8, -0.3), b: P2::new(0.6, 0.9), thickness: 0.2 }.sdf(p));
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
      P2::new(-0.9, -0.5), P2::new(0.8, -0.7),
      P2::new(0.5, 0.9), P2::new(-0.3, 0.4),
    ]}.sdf(p));
    check_lipschitz("HolyCross", 1.0, |p| HolyCross.sdf(p));
    check_lipschitz("boundary_rect", 1.0, sdf::boundary_rect::<f64, 2>);
  }

  #[test] fn lipschitz_combinators() {
    let star = || Star { n: 5, m: 10.0 / 3.0 };
    // translate / rotate / scale: precomposition with an isometry (or a
    // similarity whose value re-scale cancels the coordinate re-scale)
    // preserves the constant exactly
    check_lipschitz("translate", 1.0, |p| star().translate(V2::new(0.3, -0.2)).sdf(p));
    check_lipschitz("rotate", 1.0, |p| star().rotate(Rotation2::new(37f64.to_radians())).sdf(p));
    check_lipschitz("scale(0.35)", 1.0, |p| star().scale(0.35).sdf(p));
    check_lipschitz("scale(2.5)", 1.0, |p| star().scale(2.5).sdf(p));
    // boolean ops: min/max of L-Lipschitz fields is max(L₁, L₂)-Lipschitz
    check_lipschitz("union", 1.0, |p| Circle.translate(V2::new(0.4, 0.0))
      .union(Square.scale(0.6)).sdf(p));
    check_lipschitz("subtraction", 1.0, |p| Shape::<f64, 2>::subtraction(
      Square, Circle.scale(0.7).translate(V2::new(0.5, 0.5))).sdf(p));
    check_lipschitz("intersection", 1.0, |p| Shape::<f64, 2>::intersection(
      Circle, Square.rotate(Rotation2::new(20f64.to_radians()))).sdf(p));
    // smooth_min: ∇ = w·∇f + (1−w)·∇g with w ∈ (0, 1) — a convex combination
    check_lipschitz("smooth_min", 1.0, |p| Circle.translate(V2::new(-0.4, 0.1))
      .smooth_min(Square.scale(0.5).translate(V2::new(0.5, 0.0)), 32.0).sdf(p));
  }

  // the trait-derived constants agree with the numerically-verified bounds
  #[test] fn lipschitz_trait() {
    // a bare `Lipschitz` call is dimension-independent, so `D` must be named
    assert_eq!(Lipschitz::<f64>::lipschitz(&Circle::<2>), 1.0);
    assert_eq!(Lipschitz::<f64>::lipschitz(&HolyCross), 1.0);
    let chain = Star { n: 5, m: 10.0 / 3.0 }
      .scale(0.35)
      .rotate(Rotation2::new(37f64.to_radians()))
      .translate(V2::new(0.3, -0.2));
    assert_eq!(chain.lipschitz(), 1.0);
    let boolean = Shape::<f64, 2>::union(Circle, Shape::<f64, 2>::subtraction(Square, Ring::<f64, 2> { inner_r: 0.5 }));
    assert_eq!(boolean.lipschitz(), 1.0);
  }
}
