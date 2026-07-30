//! Shapes defined for **any** dimension count `D`.
//!
//! Their fields are built from vector algebra alone — norms, per-axis
//! `abs`/`max`, dot products — so nothing in here is tied to the plane. Two of
//! them ([`Moon`], [`Kakera`]) are *solids of revolution*: the point is first
//! reduced to the `(axial, radial)` half-plane about axis 0, then measured
//! against the same 2D generator, which is exact (see [`revolve`]).

use {
  super::clamp,
  crate::{
    geometry::{Aabb, BoundingBox, Point, Real, Vector, VectorExt, V2},
    sdf::{Lipschitz, SDF},
  },
  num_traits::Float,
};

/// Unit sphere (a circle in 2D — the default; `Hypersphere::<3>` for a ball, …).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/hypersphere.webp" alt="a disc" width="256">
///
/// The dimension lives on the type so that combinator chains starting from a
/// bare `Hypersphere` stay inferable; it defaults to 2 in type positions.
///
/// `sdf(p) = |p| - 1` — the exact signed distance in any dimension, hence
/// 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Hypersphere<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Hypersphere<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl <T: Real, const D: usize> SDF<T, D> for Hypersphere<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    pixel.coords.length() - T::one()
  }
}

/// Axis-aligned box with center at the origin (a rectangle in 2D, a cuboid in
/// 3D).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/hyperrect.webp" alt="a 1.5 x 0.9 rectangle" width="256">
///
/// With `q = |p| - size/2` (componentwise): `sdf(p) = |max(q, 0)| + min(max_a q_a, 0)`
/// — the exact signed distance to the box in any dimension, hence 1-Lipschitz.
#[derive(Debug, Copy, Clone)]
pub struct Hyperrect<T, const D: usize> {
  pub size: Vector<T, D>
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Hyperrect<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let two = T::one() + T::one();
    Aabb {
      min: Point::from(self.size / -two),
      max: Point::from(self.size / two),
    }}}

impl<T: Real, const D: usize> SDF<T, D> for Hyperrect<T, D> {
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

/// `= Hyperrect { size: [2.0; D] }` — exact SDF, 1-Lipschitz (see
/// [`Hyperrect`]). Any dimension; defaults to 2.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/hypersquare.webp" alt="a square" width="256">
#[derive(Debug, Copy, Clone)]
pub struct Hypersquare<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Hypersquare<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Hypersquare<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    Hyperrect { size: Vector::repeat(two) }.sdf(pixel)
  }}

/// Line segment from `a` to `b` with round caps (a capsule), any dimension.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/line.webp" alt="a thick segment with round caps" width="256">
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

/// Annulus: unit sphere with a concentric hole of radius `inner_r` (a spherical
/// shell in 3D; any dimension, defaults to 2).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/ring.webp" alt="an annulus of inner radius 0.55" width="256">
///
/// `sdf(p) = max(|p| - 1, inner_r - |p|)` — the boolean subtraction of two
/// concentric spheres, which for a shell happens to be the exact signed
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

/// Reduce `p` to the `(axial, radial)` half-plane of a solid of revolution
/// about axis 0: `axial = p₀`, `radial = |p₁..p_D|`.
///
/// A solid of revolution's signed distance *equals* the 2D signed distance of
/// its generator, measured in this half-plane: for any candidate `q` in the
/// solid, `|p − q| ≥ dist₂(reduce(p), reduce(q))`, with equality when `q`'s
/// perpendicular component is aligned with `p`'s — so the reduction loses
/// nothing, and the reduced 2D field is exact whenever the generator's is.
/// The map is 1-Lipschitz, so it preserves the generator's constant as well.
/// At `D = 2` it degenerates to `(p.x, |p.y|)`, hence these shapes are
/// unchanged in the plane.
#[inline]
fn revolve<T: Real, const D: usize>(p: Point<T, D>) -> V2<T> {
  let radial = (1..D)
    .fold(T::zero(), |acc, a| acc + p[a] * p[a])
    .sqrt();
  V2::new(p[0], radial)
}

/// Crescent moon, revolved about axis 0 in higher dimensions; `phase` in
/// `-1..=1`.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/moon.webp" alt="a crescent at phase 0.5" width="256">
///
/// A unit sphere minus a unit sphere offset by `d = 2·phase` along axis 0.
/// Near the cusps the field is the distance to the cusp ridge; elsewhere
/// `sdf(p) = max(|p| - 1, 1 - |p - d·e₀|)` — together the exact signed
/// distance, hence 1-Lipschitz. At `phase = 0` the crescent degenerates to its
/// boundary sphere (`sdf = ||p| - 1|`, an empty-interior shape).
///
/// In 3D this is the crescent of revolution — a ball with a spherical bite
/// taken out along the x-axis; see [`revolve`].
#[derive(Debug, Copy, Clone)]
pub struct Moon<T> {
  pub phase: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Moon<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Moon<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    // (axial, radial) — at D = 2 this is exactly `(p.x, |p.y|)`
    let pixel = revolve(pixel);
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

/// A shard: rhombus with half-diagonals `(width, 1)`, revolved about axis 0 in
/// higher dimensions — a bicone (spindle) in 3D with apexes at `±width·e₀` and
/// equatorial radius 1.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/kakera.webp" alt="a rhombus of width 0.55" width="256">
///
/// `p` is reduced to the first quadrant of the `(axial, radial)` half-plane,
/// then measured against the single edge segment, signed by the side of the
/// edge line: `sdf(p) = ±|q - proj(q)|` — the exact signed distance, hence
/// 1-Lipschitz. See [`revolve`].
#[derive(Debug, Copy, Clone)]
pub struct Kakera<T> {
  pub width: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Kakera<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let mut half = Vector::<T, D>::repeat(T::one());
    half[0] = self.width;
    Aabb::new(Point::from(half.map(|x| -x)), Point::from(half))
  }}

impl<T: Real, const D: usize> SDF<T, D> for Kakera<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    let ndot = |a: V2<T>, b: V2<T>| a.x * b.x - a.y * b.y;
    let b = V2::new(self.width, T::one());
    // |axial| paired with the (already non-negative) radial coordinate; at
    // D = 2 this is exactly `p.coords.abs()`
    let q = revolve(pixel);
    let q = V2::new(q.x.abs(), q.y);
    let mut h = (-two * ndot(q, b) + ndot(b, b)) / b.dot(&b);
    h = clamp(h, -T::one(), T::one());
    let d = (q - V2::new(T::one() - h, T::one() + h).component_mul(&b) / two).length();
    d * (q.x * b.y + q.y * b.x - b.x * b.y).signum()
  }
}

/// Axis-aligned cross: `D` orthogonal arms of half-length 1 and half-width
/// `thickness` (a plus sign in 2D, a 3-armed jack in 3D). Assumes
/// `thickness ≤ 1`.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/cross.webp" alt="a plus sign of thickness 0.3" width="256">
///
/// A point is inside iff its largest `|pₐ|` is `≤ 1` **and** its second
/// largest is `≤ thickness` — that is, at most one axis may leave the central
/// cube. Writing `q₁ ≥ q₂ ≥ …` for `|pₐ|` sorted descending (a permutation,
/// hence an isometry, so distances survive the sort):
///
/// ```text
/// outside:  √( max(q₁−1, 0)² + Σ_{a≥2} max(qₐ−t, 0)² )
/// inside:  −min( 1 − q₁,  √( max(t−q₁, 0)² + max(t−q₂, 0)² ) )
/// ```
///
/// The two interior terms are the two ways out: through an arm's end cap, or
/// past a reflex corner by lifting the two largest coordinates above `t`. Both
/// branches are the exact signed distance, hence 1-Lipschitz — a plain `min`
/// of `D` boxes would instead underestimate the interior depth.
///
/// Exact for every `thickness ≤ 1`, thick arms included. The classic 2D fold
/// (Quílez's `sdCross`, which this replaces) takes the second interior term
/// unconditionally and so mis-measures the depth once `thickness > 1/2`, where
/// a point in the central cube can be nearer to an arm cap than to a reflex
/// corner; see `shapes::tests::generalized_shapes_unchanged_in_2d`.
#[derive(Debug, Copy, Clone)]
pub struct Cross<T> {
  pub thickness: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Cross<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Cross<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let t = self.thickness;
    // the two largest |pₐ|, and Σ max(|pₐ| − t, 0)² over every axis
    let (mut q1, mut q2) = (T::neg_infinity(), T::neg_infinity());
    let mut excess = T::zero();
    for a in 0..D {
      let v = pixel[a].abs();
      if v > q1 { q2 = q1; q1 = v; } else if v > q2 { q2 = v; }
      let e = (v - t).max(T::zero());
      excess = excess + e * e;
    }

    if q1 > T::one() || q2 > t {
      let end = (q1 - T::one()).max(T::zero());
      // drop the largest axis' own term: it is allowed to exceed `t`
      let own = (q1 - t).max(T::zero());
      let side = (excess - own * own).max(T::zero());
      (end * end + side).sqrt()
    } else {
      let cap = T::one() - q1;
      let (a, b) = ((t - q1).max(T::zero()), (t - q2).max(T::zero()));
      let reflex = (a * a + b * b).sqrt();
      -cap.min(reflex)
    }
  }
}

/// A closed half-space `{ v : normal·v ≤ offset }`, the building block of
/// [`Polytope`]. Prefer [`Self::new`], which normalizes.
#[derive(Debug, Copy, Clone)]
pub struct HalfSpace<T, const D: usize> {
  /// Outward normal; must be a **unit** vector for the field to be 1-Lipschitz.
  pub normal: Vector<T, D>,
  pub offset: T,
}

impl<T: Real, const D: usize> HalfSpace<T, D> {
  /// Scales `normal` to unit length (and `offset` with it, which leaves the
  /// half-space itself unchanged), so the resulting field is 1-Lipschitz
  /// whatever magnitude was passed in.
  pub fn new(normal: Vector<T, D>, offset: T) -> Self {
    let len = normal.length();
    if len > T::zero() {
      Self { normal: normal / len, offset: offset / len }
    } else {
      Self { normal, offset }
    }
  }
}

/// Convex polytope: the intersection of half-spaces — the N-dimensional
/// analogue of [`Polygon`](super::d2::Polygon), and the arbitrary-dimension
/// generalization of [`NGonC`](super::d2::NGonC)'s construction.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/polytope.webp" alt="a pentagon from five half-spaces" width="256">
///
/// `sdf(p) = max_i (nᵢ·p - dᵢ)` over the bounding half-spaces. Exact in the
/// interior (where the nearest boundary point always lies on a face plane) and
/// beside each face; an *underestimate* outside past an edge or vertex, where
/// it reports the distance to the nearest face *plane* rather than to the
/// polytope — the same trade-off [`NGonC`](super::d2::NGonC) makes, and it errs
/// only toward reporting shapes as nearer than they are. A max of unit-gradient
/// linear fields, hence 1-Lipschitz provided the normals are unit (use
/// [`HalfSpace::new`]). An empty half-space list yields the constant "no shape"
/// field.
///
/// Convex only — a non-convex region is an intersection of no half-space set;
/// build one with [`Combinator::subtraction`](crate::geometry::Combinator::subtraction)
/// or [`union`](crate::geometry::Combinator::union) instead.
///
/// [`Self::bounding_box`] reports the unit box, following this module's
/// unit-shape convention: define the polytope inside the unit sphere (offsets
/// `≤ 1`, as `NGonC` does with `cos(π/n)`) and position it with
/// [`scale`](crate::geometry::Combinator::scale) / etc. A polytope that reaches
/// beyond `[-1, 1]^D` still has the correct field, but under-reports its
/// bounds — which affects only drawing and the transform pivots, never the
/// solvers.
#[derive(Debug, Copy, Clone)]
pub struct Polytope<U> {
  pub half_spaces: U
}

impl<T, U, const D: usize> BoundingBox<T, D> for Polytope<U>
  where T: Real,
        U: AsRef<[HalfSpace<T, D>]> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T, U, const D: usize> SDF<T, D> for Polytope<U>
  where T: Real,
        U: AsRef<[HalfSpace<T, D>]> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let hs = self.half_spaces.as_ref();
    if hs.is_empty() {
      return T::max_value() / (T::one() + T::one());
    }
    hs.iter()
      .map(|h| h.normal.dot(&pixel.coords) - h.offset)
      .fold(T::neg_infinity(), T::max)
  }
}

// Every shape above is either an exact SDF or a 1-Lipschitz underestimate
// assembled from unit-gradient pieces (see each shape's doc), so the honest
// bound is `1`. Combinators propagate these automatically; see
// [`crate::sdf::Lipschitz`].
impl<T: Float, const D: usize> Lipschitz<T> for Hypersphere<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Hyperrect<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Hypersquare<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float + nalgebra::Scalar, const D: usize> Lipschitz<T> for Line<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, const D: usize> Lipschitz<T> for Ring<T, D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Moon<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Kakera<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Cross<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U> Lipschitz<T> for Polytope<U> { fn lipschitz(&self) -> T { T::one() } }

/// The convex hull of `vertices`, expressed as the intersection of the
/// half-spaces with the given facet `normals` — an H-representation built from
/// a V-representation.
///
/// Each offset is the *support function* `h(n) = max_v n·v`, which is by
/// definition the hull's supporting plane in direction `n`; supply every facet
/// normal (typically the vertex directions of the dual polytope) and the result
/// is exactly the hull, while too few normals yield a superset. Normals are
/// normalized for you, so the field is 1-Lipschitz.
pub fn convex_hull<T: Real, const D: usize>(
  normals: impl IntoIterator<Item = Vector<T, D>>,
  vertices: &[Vector<T, D>],
) -> Polytope<Vec<HalfSpace<T, D>>> {
  let half_spaces = normals.into_iter()
    .map(|n| {
      let normal = n.robust_normalize();
      let offset = vertices.iter()
        .fold(T::neg_infinity(), |acc, v| acc.max(normal.dot(v)));
      HalfSpace { normal, offset }
    })
    .collect();
  Polytope { half_spaces }
}

/// Scale `vertices` so the farthest lands on the unit sphere, following this
/// module's unit-shape convention.
pub(crate) fn unit_circumradius<T: Real, const D: usize>(vertices: &mut [Vector<T, D>]) {
  let far = vertices.iter().fold(T::zero(), |acc, v| acc.max(v.length()));
  if far > T::zero() {
    for v in vertices.iter_mut() { *v /= far; }
  }
}

/// Visit every permutation of `slots`, reporting whether it is *even*.
///
/// A recursive selection walk with explicit swap counting — the vertex orbits of
/// the exceptional 4-polytopes are defined over even permutations only, so
/// unambiguous parity matters more here than speed (`n ≤ 6`).
pub(crate) fn permutations<T: Copy>(slots: &[T], visit: &mut impl FnMut(&[T], bool)) {
  fn walk<T: Copy>(
    buf: &mut [T],
    pos: usize,
    swaps: usize,
    visit: &mut impl FnMut(&[T], bool),
  ) {
    if pos + 1 >= buf.len() {
      visit(buf, swaps % 2 == 0);
      return;
    }
    for j in pos..buf.len() {
      buf.swap(pos, j);
      walk(buf, pos + 1, swaps + (j != pos) as usize, visit);
      buf.swap(pos, j);
    }
  }
  let mut buf = slots.to_vec();
  walk(&mut buf, 0, 0, visit);
}

/// Every sign assignment of the non-zero entries of `pattern`, appended to
/// `out` — the other half of an orbit generator.
pub(crate) fn sign_orbit<T: Real, const D: usize>(
  pattern: &[T],
  out: &mut Vec<Vector<T, D>>,
) {
  let flippable: Vec<usize> = (0..D).filter(|&a| pattern[a] != T::zero()).collect();
  for bits in 0..1u32 << flippable.len() {
    let mut v = Vector::<T, D>::from_fn(|a, _| pattern[a]);
    for (i, &a) in flippable.iter().enumerate() {
      if bits >> i & 1 == 1 { v[a] = -v[a]; }
    }
    out.push(v);
  }
}

/// All permutations (or only the even ones) of `pattern`, with every sign
/// assignment — the standard way regular-polytope vertex orbits are tabulated.
pub(crate) fn orbit<T: Real, const D: usize>(
  pattern: [T; D],
  even_only: bool,
  out: &mut Vec<Vector<T, D>>,
) {
  permutations(&pattern, &mut |perm, even| {
    if even_only && !even { return; }
    sign_orbit(perm, out);
  });
  // permutations of a pattern with repeated entries collide, and so do sign
  // flips of a zero; both leave duplicates, which are harmless for a support
  // function but wasteful as half-spaces
  dedup_directions(out);
}

/// Drop duplicate vectors (within a tight tolerance), preserving order.
pub(crate) fn dedup_directions<T: Real, const D: usize>(v: &mut Vec<Vector<T, D>>) {
  let eps = T::from(1e-9).unwrap();
  let mut kept: Vec<Vector<T, D>> = Vec::with_capacity(v.len());
  for &candidate in v.iter() {
    if !kept.iter().any(|k| (k - candidate).length() < eps) {
      kept.push(candidate);
    }
  }
  *v = kept;
}

/// The `D+1` unit vertex directions of a regular simplex: pairwise dot `-1/D`,
/// summing to zero.
///
/// Built by projecting the standard simplex `{e₀ … e_D} ⊂ ℝ^(D+1)` onto the
/// hyperplane orthogonal to `(1,…,1)` through the Helmert basis, whose `k`-th
/// row is `(1,…,1,-(k+1),0,…,0)/√((k+1)(k+2))`. Those rows are orthonormal, so
/// the projection is an isometry and regularity survives it.
pub fn simplex_vertices<T: Real, const D: usize>() -> Vec<Vector<T, D>> {
  (0..=D)
    .map(|i| Vector::<T, D>::from_fn(|k, _| {
      let k1 = T::from(k + 1).unwrap();
      let denom = (k1 * (k1 + T::one())).sqrt();
      if i <= k { T::one() / denom }
      else if i == k + 1 { -k1 / denom }
      else { T::zero() }
    }).robust_normalize())
    .collect()
}

/// Regular simplex inscribed in the unit sphere — triangle, tetrahedron,
/// 5-cell and onward — as a [`Polytope`] of `D+1` half-spaces.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/simplex.webp" alt="a triangle, the 2D simplex" width="256">
///
/// The facet opposite each vertex has that vertex's direction as its *inward*
/// normal and, at unit circumradius, sits `1/D` from the centre. Exact inside,
/// conservative outside, 1-Lipschitz (see [`Polytope`]); round it with
/// [`offset`](crate::geometry::Combinator::offset) to be exact everywhere.
pub fn simplex<T: Real, const D: usize>() -> Polytope<Vec<HalfSpace<T, D>>> {
  let verts = simplex_vertices::<T, D>();
  convex_hull(verts.iter().map(|v| v.map(|x| -x)), &verts)
}

/// Permutohedron: the Voronoi cell of the `A_D*` lattice, and a polytope that
/// **tiles space by translation** in every dimension — a hexagon in 2D, the
/// [`truncated_octahedron`](super::d3::truncated_octahedron) in 3D.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/permutohedron.webp" alt="a hexagon, the 2D permutohedron" width="256">
///
/// It is the convex hull of all permutations of `(0, 1, …, D)`, which lives in
/// the hyperplane `Σx = const` of `ℝ^(D+1)`; this returns its isometric image in
/// `ℝ^D` (the same Helmert projection as [`simplex_vertices`]), scaled to unit
/// circumradius. Facets are indexed by the proper non-empty subsets of the
/// `D+1` coordinates, so there are `2^(D+1) - 2` of them: 6 in 2D, 14 in 3D,
/// 30 in 4D.
///
/// Exact inside, conservative outside, 1-Lipschitz. Enumerates `(D+1)!`
/// vertices, so keep `D` modest.
pub fn permutohedron<T: Real, const D: usize>() -> Polytope<Vec<HalfSpace<T, D>>> {
  // Helmert row `k` of the projection ℝ^(D+1) → ℝ^D; also centres its argument,
  // since the component along (1,…,1) is annihilated
  let project = |w: &[T]| Vector::<T, D>::from_fn(|k, _| {
    let k1 = T::from(k + 1).unwrap();
    let denom = (k1 * (k1 + T::one())).sqrt();
    let head = (0..=k).fold(T::zero(), |acc, i| acc + w[i]);
    (head - k1 * w[k + 1]) / denom
  });

  let ladder: Vec<T> = (0..=D).map(|i| T::from(i).unwrap()).collect();
  let mut vertices = vec![];
  permutations(&ladder, &mut |perm, _| vertices.push(project(perm)));
  unit_circumradius(&mut vertices);

  let normals = (1..(1u32 << (D + 1)) - 1).map(|mask| {
    let indicator: Vec<T> = (0..=D)
      .map(|i| if mask >> i & 1 == 1 { T::one() } else { T::zero() })
      .collect();
    project(&indicator)
  });
  convex_hull(normals, &vertices)
}

/// Cross-polytope (the `ℓ¹` ball): the hypercube's dual — a rhombus in 2D, an
/// octahedron in 3D, the 16-cell in 4D. Any dimension; defaults to 2.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/orthoplex.webp" alt="a square rotated 45 degrees, the 2D orthoplex" width="256">
///
/// `sdf(p) = (Σ|pₐ| - 1)/√D`, the closed form of the maximum over all `2^D`
/// facet planes `(±1,…,±1)/√D`, so this costs `O(D)` where the equivalent
/// [`Polytope`] would cost `O(2^D)`. Exact in the interior and beside each
/// facet, an underestimate past an edge or vertex; 1-Lipschitz, its gradient
/// being the unit vector `(±1,…,±1)/√D`.
#[derive(Debug, Copy, Clone)]
pub struct Orthoplex<const D: usize = 2>;

impl<T: Real, const D: usize> BoundingBox<T, D> for Orthoplex<D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Orthoplex<D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let l1 = pixel.coords.iter().fold(T::zero(), |acc, &x| acc + x.abs());
    (l1 - T::one()) / T::from(D).unwrap().sqrt()
  }
}

/// Unit `ℓᵖ` ball — the superellipsoid / Lamé family `{ p : ‖p‖_p ≤ 1 }`.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/lp_ball.webp" alt="the unit ball of the l4 norm, a squarish disc" width="256">
///
/// Interpolates a whole family of rounded boxes: `p = 1` is the [`Orthoplex`],
/// `p = 2` the [`Hypersphere`], and `p → ∞` approaches the [`Hypersquare`];
/// `p` between 4 and 8 gives the "squircle" look that reads especially well in
/// volumetric renders.
///
/// `sdf(p) = ‖p‖_p - 1`. Not the exact Euclidean distance, but an honest
/// *underestimate*, since any `L`-Lipschitz function vanishing on the boundary
/// satisfies `|f| ≤ L·dist`. The constant follows from the triangle inequality
/// and the comparison of `ℓᵖ` norms,
///
/// ```text
/// |f(x) - f(y)| ≤ ‖x - y‖_p ≤ D^max(0, 1/p - 1/2) · ‖x - y‖₂,
/// ```
///
/// so the field is **1-Lipschitz for `p ≥ 2`**, and `D^(1/p - 1/2)`-Lipschitz
/// for `1 ≤ p < 2`; the [`Lipschitz`] impl reports whichever applies.
///
/// Requires `p ≥ 1`. Below 1 the "ball" turns star-shaped and tempting, but
/// `‖·‖_p` is no longer a norm and its gradient blows up like `|xₐ|^(p-1)` at
/// the axes — an unbounded gradient admits no honest Lipschitz constant at all,
/// which would make the ADF's pruning unsound rather than merely imprecise.
#[derive(Debug, Copy, Clone)]
pub struct LpBall<T, const D: usize = 2> {
  pub p: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for LpBall<T, D> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for LpBall<T, D> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let sum = pixel.coords.iter()
      .fold(T::zero(), |acc, &x| acc + x.abs().powf(self.p));
    sum.powf(T::one() / self.p) - T::one()
  }
}

/// Chain of round-capped segments through `vertices` — one capsule per
/// consecutive pair, in any dimension.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/polyline.webp" alt="a thick four-vertex zigzag" width="256">
///
/// `sdf(p) = minᵢ dist(p, [vᵢ, vᵢ₊₁]) - thickness/2`: exact outside, since the
/// distance to a union is the minimum of the distances, and an underestimate of
/// the interior depth where consecutive capsules overlap — exactly as
/// [`Combinator::union`](crate::geometry::Combinator::union). 1-Lipschitz. Fewer than two
/// vertices gives the constant "no shape" field.
///
/// Useful for skeletal and filamentary structures, and for curves with no
/// closed-form distance, which can be sampled into a polyline instead — see
/// [`torus_knot`](super::d3::torus_knot).
#[derive(Debug, Copy, Clone)]
pub struct Polyline<U, T> {
  pub vertices: U,
  pub thickness: T
}

impl<T, U, const D: usize> BoundingBox<T, D> for Polyline<U, T>
  where T: Real,
        U: AsRef<[Point<T, D>]> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let two = T::one() + T::one();
    Aabb::from_points(self.vertices.as_ref().iter().copied())
      .inflate(self.thickness / two)
  }}

impl<T, U, const D: usize> SDF<T, D> for Polyline<U, T>
  where T: Real,
        U: AsRef<[Point<T, D>]> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let v = self.vertices.as_ref();
    let two = T::one() + T::one();
    if v.len() < 2 {
      return T::max_value() / two;
    }
    let mut best = T::max_value();
    for w in v.windows(2) {
      let ba = w[1] - w[0];
      let pa = pixel - w[0];
      let h = clamp(pa.dot(&ba) / ba.dot(&ba), T::zero(), T::one());
      best = best.min((pa - ba * h).length());
    }
    best - self.thickness / two
  }
}

/// Cartesian product of balls over a partition of the axes: `spec` lists
/// `(block length, radius)` for consecutive runs, which should sum to `D`.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/product_ball.webp" alt="z-slices of a cylinder" width="256">
///
/// One type covers a family. In 3D, `[(2, r), (1, h)]` is a cylinder and
/// `[(1, a), (1, b), (1, c)]` a box, so [`Hyperrect`] is the all-ones case; in
/// 4D, `[(2, r₁), (2, r₂)]` is the **duocylinder** and `[(3, r), (1, h)]` the
/// spherinder.
///
/// The field is the box construction applied to the per-block radial excesses
/// `qᵢ = ‖p_Bᵢ‖ - rᵢ`, namely `min(max qᵢ, 0) + ‖max(q, 0)‖` — the **exact**
/// signed distance, because the nearest point is reached by clamping each
/// block's radius independently, and 1-Lipschitz since the blocks act on
/// disjoint axes.
///
/// Blocks are contiguous; [`rotate`](crate::geometry::Combinator::rotate) the shape
/// if you need interleaved axes.
#[derive(Debug, Copy, Clone)]
pub struct ProductBall<U> {
  pub spec: U
}

impl<T, U, const D: usize> BoundingBox<T, D> for ProductBall<U>
  where T: Real,
        U: AsRef<[(usize, T)]> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let mut half = Vector::<T, D>::repeat(T::zero());
    let mut axis = 0;
    for &(len, radius) in self.spec.as_ref() {
      for _ in 0..len {
        if axis < D { half[axis] = radius; }
        axis += 1;
      }
    }
    Aabb::new(Point::from(half.map(|x| -x)), Point::from(half))
  }}

impl<T, U, const D: usize> SDF<T, D> for ProductBall<U>
  where T: Real,
        U: AsRef<[(usize, T)]> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let mut outside = T::zero();        // Σ max(qᵢ, 0)²
    let mut inside = T::neg_infinity(); // max qᵢ
    let mut axis = 0;
    for &(len, radius) in self.spec.as_ref() {
      let mut sq = T::zero();
      for _ in 0..len {
        if axis < D { sq = sq + pixel[axis] * pixel[axis]; }
        axis += 1;
      }
      let q = sq.sqrt() - radius;
      inside = inside.max(q);
      let e = q.max(T::zero());
      outside = outside + e * e;
    }
    outside.sqrt() + inside.min(T::zero())
  }
}

/// Solid torus: a disc of radius `minor` swept around axis 0 at distance
/// `major` — defined in any dimension, and equal to
/// `Hypersphere.scale(minor).revolve(major)`.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/torus.webp" alt="slices of a torus along its axis of revolution" width="256">
///
/// `sdf(p) = ‖(|p₁..p_D| - major, p₀)‖ - minor`, the exact signed distance while
/// `major ≥ minor` keeps the swept disc clear of the axis. A smaller `major`
/// makes the torus pass through itself and the field becomes a conservative
/// underestimate near the axis; 1-Lipschitz either way.
///
/// Degenerates gracefully: in 2D the "torus" is the pair of discs at
/// `(0, ±major)`, which is what sweeping a one-dimensional radial coordinate
/// gives.
#[derive(Debug, Copy, Clone)]
pub struct Torus<T> {
  pub major: T,
  pub minor: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Torus<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    let mut half = Vector::<T, D>::repeat(self.major + self.minor);
    half[0] = self.minor;
    Aabb::new(Point::from(half.map(|x| -x)), Point::from(half))
  }}

impl<T: Real, const D: usize> SDF<T, D> for Torus<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let q = revolve(pixel);
    V2::new(q.y - self.major, q.x).length() - self.minor
  }
}

/// Gyroid: the triply-periodic minimal surface `Σ sin(k·xₐ)·cos(k·xₐ₊₁)`, and
/// in any dimension its cyclic analogue.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/gyroid.webp" alt="z-slices of a gyroid surface" width="256">
///
/// The shape is one of the two interpenetrating labyrinths, `{f ≤ 0}` — a
/// genuine solid filling half of space. [`shell`](crate::geometry::Combinator::shell)
/// turns it into the thickened *surface* instead: a single connected sheet
/// winding through space, which makes a far more interesting ADF workload than
/// a bag of balls.
///
/// ```
/// # use space_filling::{geometry::*, sdf::SDF};
/// let labyrinth = Gyroid { frequency: 8.0 }
///   .shell(0.03f64)
///   .intersection(Hypersquare::<3>);
/// assert!(labyrinth.sdf(Point::<f64, 3>::from([0.4, 0.1, -0.2])).is_finite());
/// ```
///
/// This is a level-set function rather than a distance function — but it is
/// divided by its own gradient bound (`|∂f/∂xₐ| ≤ 2k`, hence `|∇f| ≤ 2k√D`),
/// which makes the stored field **1-Lipschitz** and therefore an honest
/// conservative underestimate of the true distance: a 1-Lipschitz function
/// vanishing on the boundary can never exceed the distance to it. Pruning stays
/// sound, merely less aggressive than for an exact field.
///
/// The gyroid is unbounded and periodic, so — as with [`Polytope`] —
/// [`Self::bounding_box`] reports the unit box, on the assumption that you
/// intersect it with a container.
#[derive(Debug, Copy, Clone)]
pub struct Gyroid<T> {
  /// Spatial frequency: one cell per `2π/frequency` of space.
  pub frequency: T
}

impl<T: Real, const D: usize> BoundingBox<T, D> for Gyroid<T> {
  fn bounding_box(&self) -> Aabb<T, D> {
    Aabb::symmetric(T::one())
  }}

impl<T: Real, const D: usize> SDF<T, D> for Gyroid<T> {
  fn sdf(&self, pixel: Point<T, D>) -> T {
    let two = T::one() + T::one();
    let k = self.frequency;
    let f = (0..D).fold(T::zero(), |acc, a|
      acc + (k * pixel[a]).sin() * (k * pixel[(a + 1) % D]).cos());
    // normalized by the gradient bound, so the reported field is 1-Lipschitz
    f / (two * k * T::from(D).unwrap().sqrt())
  }
}

// Exact fields and conservative unit-gradient underestimates alike: the honest
// bound is 1. `LpBall` is the one shape whose constant depends on its
// parameters, and it implements the trait itself.
impl<T: Float, const D: usize> Lipschitz<T> for Orthoplex<D> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U, V> Lipschitz<T> for Polyline<U, V> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float, U> Lipschitz<T> for ProductBall<U> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Torus<T> { fn lipschitz(&self) -> T { T::one() } }
impl<T: Float> Lipschitz<T> for Gyroid<T> { fn lipschitz(&self) -> T { T::one() } }

impl<T: Real, const D: usize> Lipschitz<T> for LpBall<T, D> {
  fn lipschitz(&self) -> T {
    let two = T::one() + T::one();
    if self.p >= two {
      T::one()
    } else {
      // D^(1/p − 1/2), the norm-comparison constant
      T::from(D).unwrap().powf(T::one() / self.p - T::one() / two)
    }
  }
}
