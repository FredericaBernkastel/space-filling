#![allow(non_upper_case_globals)]
use {
  super::{BoundingBox, WorldSpace, Translation, Shape},
  crate::sdf::{SDF, Union},
  euclid::{Box2D, Point2D, Vector2D as V2},
  std::marker::PhantomData
};

/// Unit circle
#[derive(Debug, Copy, Clone)]
pub struct Circle;

impl<S> BoundingBox<f32, S> for Circle {
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

/// Rectangle with center at `[0, 0]`
#[derive(Debug, Copy, Clone)]
pub struct Rect<T, S> {
  pub size: Point2D<T, S>
}

impl<S> BoundingBox<f32, S> for Rect<f32, S> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      -self.size / 2.0,
      self.size / 2.0
    )}}

impl SDF<f32> for Rect<f32, WorldSpace> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let dist = pixel.to_vector().abs() - (self.size.to_vector() / 2.0);
    let outside_dist = dist
      .max(V2::splat(0.0))
      .length();
    let inside_dist = dist.x
      .max(dist.y)
      .min(0.0);
    outside_dist + inside_dist
  }}

#[derive(Debug, Copy, Clone)]
pub struct Line<T, S> {
  pub a: Point2D<T, S>,
  pub b: Point2D<T, S>,
  pub thickness: T,
}

impl<S> BoundingBox<f32, S> for Line<f32, S> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    let ret = Box2D::from_points([self.a, self.b]);
    let t = V2::splat(self.thickness / 2.0);
    Box2D::new(ret.min - t, ret.max + t)
  }}

impl SDF<f32> for Line<f32, WorldSpace> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let ba = self.b - self.a;
    let pa = pixel - self.a;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - self.thickness / 2.0
  }
}

/// Regular polygon with N sides, inscribed in a unit circle. Partially evaluated at compile-time.
#[derive(Debug, Copy, Clone)]
pub struct NGonC<const N: usize>;

impl<S, const N: usize> BoundingBox<f32, S> for NGonC<N> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl<const N: usize> SDF<f32> for NGonC<N> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    use std::f32::consts::*;
    let angle = pixel.y.atan2(pixel.x) + FRAC_PI_2;
    let split = TAU / N as f32;
    let r = (PI / N as f32).cos();
    pixel.to_vector().length() * (split * (angle / split + 0.5).floor() - angle).cos() - r
  }
}

/// Regular polygon with N sides, inscribed in a unit circle. Evaluated at runtime.
#[derive(Debug, Copy, Clone)]
pub struct NGonR {
  pub n: u64
}

impl<S> BoundingBox<f32, S> for NGonR {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl SDF<f32> for NGonR {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    use std::f32::consts::*;
    let angle = pixel.y.atan2(pixel.x) + FRAC_PI_2;
    let split = TAU / self.n as f32;
    let r = (PI / self.n as f32).cos();
    pixel.to_vector().length() * (split * (angle / split + 0.5).floor() - angle).cos() - r
  }
}

/// N-pointed regular star polygon, inscibed in a unit circle.
/// `m` is density, must be between `2..=n`
#[derive(Debug, Copy, Clone)]
pub struct Star {
  pub n: u64,
  pub m: f32
}

impl<S> BoundingBox<f32, S> for Star {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl SDF<f32> for Star {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    use std::f32::consts::*;
    let module = |x: f32, y: f32| x - y * (x / y).floor();
    let an = PI / self.n as f32;
    let en = PI / self.m;
    let acs = V2::<_, WorldSpace>::new(an.cos(), an.sin());
    let ecs = V2::new(en.cos(), en.sin());

    let bn = module(pixel.x.atan2(pixel.y), 2.0 * an) - an;
    let mut p = V2::new(bn.cos(), bn.sin().abs())
      * pixel.to_vector().length()
      - acs;
    p += ecs * (-p.dot(ecs)).clamp(0.0, acs.y / ecs.y);
    p.length() * p.x.signum()
  }
}

/// `phase` in `-1..=1`.
#[derive(Debug, Copy, Clone)]
pub struct Moon {
  pub phase: f32
}

impl<S> BoundingBox<f32, S> for Moon {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl SDF<f32> for Moon {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let pixel = V2::<_, WorldSpace>::new(pixel.x, pixel.y.abs());
    let d = self.phase * 2.0;
    let a = (d * d) / (2.0 * d);
    let b = (1.0 - a * a).max(0.0).sqrt();

    if d * (pixel.x * b - pixel.y * a) > d * d * (b - pixel.y).max(0.0) {
      (pixel - V2::new(a, b)).length()
    } else {
      (pixel.length() - 1.0).max(
        -((pixel - V2::new(d, 0.0)).length() - 1.0)
      )
    }
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Kakera<T> {
  pub b: T
}

impl<S> BoundingBox<f32, S> for Kakera<f32> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::new(-self.b, -1.0),
      Point2D::new(self.b, 1.0)
    )}}

impl SDF<f32> for Kakera<f32> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let ndot = |a: V2<_, _,>, b: V2<_, _,>| a.x*b.x - a.y*b.y;
    let b = V2::new(self.b, 1.0);
    let q = pixel.to_vector().abs();
    let h = ((-2.0 * ndot(q, b) + ndot(b, b)) / b.dot(b))
      .clamp(-1.0, 1.0);
    let d = (q - V2::new(1.0 - h, 1.0 + h).component_mul(b) * 0.5).length();
    d * (q.x * b.y + q.y * b.x - b.x * b.y).signum()
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Cross<T> {
  pub thickness: T
}

impl<S> BoundingBox<f32, S> for Cross<f32> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl SDF<f32> for Cross<f32>  {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let mut pixel = pixel.to_vector().abs();
    pixel = if pixel.y > pixel.x { pixel.yx() } else { pixel };
    let q = pixel - V2::new(1.0, self.thickness);
    let k = q.x.max(q.y);
    let w = if k > 0.0 { q } else { V2::new(self.thickness - pixel.x, -k) };
    k.signum() * w.max(V2::splat(0.0)).length()
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Ring<T> {
  pub inner_r: T
}

impl<S> BoundingBox<f32, S> for Ring<f32> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::new(
      Point2D::splat(-1.0),
      Point2D::splat(1.0)
    )}}

impl SDF<f32> for Ring<f32>  {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    Circle.subtraction(Circle.scale(V2::splat(self.inner_r))).sdf(pixel)
  }
}

#[derive(Debug, Copy, Clone)]
pub struct Polygon<T> {
  pub vertices: T
}

impl<T, S> BoundingBox<f32, S> for Polygon<T>
  where T: AsRef<[Point2D<f32, S>]> {
  fn bounding_box(&self) -> Box2D<f32, S> {
    Box2D::from_points(self.vertices.as_ref())
  }}

impl<T> SDF<f32> for Polygon<T>
  where T: AsRef<[Point2D<f32, WorldSpace>]> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 {
    let v = self.vertices.as_ref();
    let mut d = match v.get(0) {
      Some(&v) => (pixel - v).dot(pixel - v),
      None => return f32::MAX / 2.0
    };
    let mut s = 1.0;
    let n = v.len();
    (0..n).zip(std::iter::once(n - 1).chain(0..n - 1))
      .for_each(|(i, j)| {
        let e = v[j] - v[i];
        let w = pixel - v[i];
        let b = w - e * (w.dot(e) / e.dot(e)).clamp(0.0, 1.0);
        d = d.min(b.dot(b));
        let c = euclid::BoolVector3D {
          x: pixel.y >= v[i].y,
          y: pixel.y < v[j].y,
          z: e.x * w.y > e.y * w.x
        };
        if c.all() || c.none() {
          s *= -1.0;
        }
      });
    s * d.sqrt()
  }
}

/// `= NGonC::<3>`
pub type Triangle = NGonC::<3>;

/// `= NGonC::<5>`
pub type Pentagon = NGonC::<5>;

/// `= NGonC::<6>`
pub type Hexagon = NGonC::<6>;

/// `= NGonC::<7>`
pub type Heptagon = NGonC::<7>;

/// `= NGonC::<8>`
pub type Octagon = NGonC::<8>;

/// `= Rect { size: [2.0, 2.0] }`
pub static Square: Rect<f32, WorldSpace> = Rect {
  size: Point2D { x: 2.0, y: 2.0, _unit: PhantomData::<WorldSpace> }
};

/// `= Star { n: 5, m: 10.0 / 3.0 }`
pub static Pentagram: Star = Star { n: 5, m: 10.0 / 3.0 };

/// `= Star { n: 6, m: 3.0 }`
pub static Hexagram: Star = Star { n: 6, m: 3.0 };

pub static HolyCross: Union <
  Rect<f32, WorldSpace> ,
  Translation < Rect<f32, WorldSpace>, f32 >
> = Union {
  s1: Rect { size: Point2D { x: 0.4, y: 2.0, _unit: PhantomData::<WorldSpace> } },
  s2: Translation {
    shape: Rect { size: Point2D {  x: 1.432, y: 0.4, _unit: PhantomData::<WorldSpace> } },
    offset: V2 { x: 0.0, y: -0.3, _unit: PhantomData::<WorldSpace> }
  }
};