#![allow(non_upper_case_globals)]
use {
  super::{Shape, BoundingBox, WorldSpace, Translation},
  crate::sdf::{SDF, Union},
  euclid::{Box2D, Point2D, Vector2D as V2},
  num_traits::{Float, Signed, FloatConst},
  std::marker::PhantomData
};

fn clamp<T: Float>(mut x: T, min: T, max: T) -> T {
  if x < min { x = min; }
  if x > max { x = max; }
  x
}

/// Unit circle
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

/// Rectangle with center at `[0, 0]`
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

/// N-pointed regular star polygon, inscibed in a unit circle.
/// `m` is density, must be between `2..=n`
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

/// `phase` in `-1..=1`.
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
    let a = (d * d) / (two * d);
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

/// `= Rect { size: [2.0, 2.0] }`
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

/// `= Star { n: 5, m: 10.0 / 3.0 }`
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

/// `= Star { n: 6, m: 3.0 }`
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