//! Shapes defined for **four dimensions only** (`D = 4`).
//!
//! Four dimensions is the richest case. Three of the six convex regular
//! 4-polytopes are members of families that run through every dimension, and so
//! live in [`dn`](super::dn) — the 5-cell is [`simplex`](super::dn::simplex),
//! the tesseract is [`Hypersquare`](super::dn::Hypersquare), the 16-cell is
//! [`Orthoplex`](super::dn::Orthoplex). The other three are exceptional to
//! 4-space and are here: the [`cell_24`], [`cell_120`] and [`cell_600`].
//!
//! The [`cell_24`] is the standout. It exists in no other dimension, it is the
//! only self-dual regular polytope besides the simplices, its 24 vertices *are*
//! the 24 roots of `D₄`, and it is that lattice's Voronoi cell — so, like the
//! [`truncated_octahedron`](super::d3::truncated_octahedron) in 3-space, it
//! **tiles 4-space by translation**.
//!
//! Also here are the two curved shapes that need a 2 + 2 split of the axes: the
//! [`duocylinder`] and the [`CliffordTorus`].

use {
  super::dn::{
    convex_hull, orbit, unit_circumradius, HalfSpace, Polytope, ProductBall
  },
  crate::{
    geometry::{Aabb, BoundingBox, Point, Real, Vector, VectorExt, V2},
    sdf::{Lipschitz, SDF},
  },
  num_traits::Float,
};

/// The golden ratio, `(1 + √5)/2`.
fn phi<T: Real>() -> T {
  let five = T::from(5.0).unwrap();
  (T::one() + five.sqrt()) / (T::one() + T::one())
}

/// The 24 vertex directions of the 24-cell: all permutations of `(±1, ±1, 0, 0)`.
///
/// These are the roots of the `D₄` lattice — the shortest vectors of the
/// checkerboard lattice `{x ∈ ℤ⁴ : Σx even}`.
pub fn cell_24_vertices<T: Real>() -> Vec<Vector<T, 4>> {
  let (zero, one) = (T::zero(), T::one());
  let mut out = vec![];
  orbit([zero, zero, one, one], false, &mut out);
  unit_circumradius(&mut out);
  out
}

/// The 120 vertex directions of the 600-cell: the 8 axis directions, the 16
/// half-diagonals `(±½, ±½, ±½, ±½)`, and the 96 *even* permutations of
/// `(±φ/2, ±½, ±1/(2φ), 0)`.
///
/// All three orbits already sit at radius 1, since
/// `φ²/4 + 1/4 + 1/(4φ²) = 1`. These double as the 120 facet normals of the
/// [`cell_120`], the two being dual.
pub fn cell_600_vertices<T: Real>() -> Vec<Vector<T, 4>> {
  let (zero, one, two) = (T::zero(), T::one(), T::one() + T::one());
  let half = one / two;
  let f = phi::<T>();
  let mut out = vec![];
  orbit([zero, zero, zero, one], false, &mut out);
  super::dn::sign_orbit(&[half, half, half, half], &mut out);
  orbit([f / two, half, one / (two * f), zero], true, &mut out);
  super::dn::dedup_directions(&mut out);
  unit_circumradius(&mut out);
  out
}

/// The 600 vertex directions of the 120-cell — equivalently, the 600 facet
/// directions of the [`cell_600`], since the two are dual.
///
/// Derived from [`cell_600_vertices`] rather than tabulated: the 600-cell's
/// cells are the 4-cliques of its edge graph (two vertices are adjacent exactly
/// when they sit at the minimal pairwise distance), and each cell's normalized
/// centroid is a facet direction — hence a vertex of the dual.
///
/// Deriving instead of tabulating is what keeps the pair **aligned**. A dual
/// pair is only dual in one relative orientation, and the classical coordinate
/// listings for these two polytopes — both perfectly regular — do not share it;
/// pairing them directly inscribes a wrong polytope whose facets miss most of
/// its vertices. Constructing one from the other cannot get that wrong.
pub fn cell_120_vertices<T: Real>() -> Vec<Vector<T, 4>> {
  let v = cell_600_vertices::<T>();
  let n = v.len();
  let dist = |i: usize, j: usize| (v[i] - v[j]).length();
  let mut edge = T::max_value();
  for i in 0..n {
    for j in i + 1..n { edge = edge.min(dist(i, j)); }
  }
  let eps = T::from(1e-6).unwrap();
  let adjacent = |i: usize, j: usize| (dist(i, j) - edge).abs() < eps;
  let neighbours: Vec<Vec<usize>> = (0..n)
    .map(|i| (0..n).filter(|&j| j != i && adjacent(i, j)).collect())
    .collect();

  let mut out = vec![];
  for i in 0..n {
    for &j in neighbours[i].iter().filter(|&&j| j > i) {
      for &k in neighbours[i].iter().filter(|&&k| k > j && adjacent(j, k)) {
        for &l in neighbours[i].iter()
          .filter(|&&l| l > k && adjacent(j, l) && adjacent(k, l)) {
          out.push((v[i] + v[j] + v[k] + v[l]).robust_normalize());
        }
      }
    }
  }
  super::dn::dedup_directions(&mut out);
  unit_circumradius(&mut out);
  out
}

/// The **24-cell** inscribed in the unit sphere — 24 octahedral facets, and the
/// Voronoi cell of the `D₄` lattice, so it tiles 4-space.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/cell_24.webp" alt="slices of a 24-cell along its fourth axis" width="256">
///
/// Self-dual: its 24 facet normals are the 8 axis directions and the 16
/// half-diagonals — which is the same set of 24 directions as its vertices, up
/// to the rotation that carries the polytope to its dual.
///
/// Exact inside, conservative outside, 1-Lipschitz (see
/// [`Polytope`]).
pub fn cell_24<T: Real>() -> Polytope<Vec<HalfSpace<T, 4>>> {
  let (zero, one, two) = (T::zero(), T::one(), T::one() + T::one());
  let half = one / two;
  let mut normals = vec![];
  orbit([zero, zero, zero, one], false, &mut normals);
  super::dn::sign_orbit(&[half, half, half, half], &mut normals);
  super::dn::dedup_directions(&mut normals);
  convex_hull(normals, &cell_24_vertices::<T>())
}

/// The **120-cell** inscribed in the unit sphere — 120 dodecahedral facets,
/// whose normals are the [`cell_600_vertices`] (the dual).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/cell_120.webp" alt="slices of a 120-cell along its fourth axis" width="256">
pub fn cell_120<T: Real>() -> Polytope<Vec<HalfSpace<T, 4>>> {
  convex_hull(cell_600_vertices::<T>(), &cell_120_vertices::<T>())
}

/// The **600-cell** inscribed in the unit sphere — 600 tetrahedral facets,
/// whose normals are the [`cell_120_vertices`] (the dual).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/cell_600.webp" alt="slices of a 600-cell along its fourth axis" width="256">
///
/// The most expensive shape in the crate to evaluate: 600 half-spaces per
/// sample. Consider baking it if you need it in a hot loop.
pub fn cell_600<T: Real>() -> Polytope<Vec<HalfSpace<T, 4>>> {
  convex_hull(cell_120_vertices::<T>(), &cell_600_vertices::<T>())
}

/// Duocylinder: the product of two discs, `‖p₀₁‖ ≤ r1` and `‖p₂₃‖ ≤ r2` — a
/// shape that needs a 2 + 2 split of the axes and so exists only in 4D.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/duocylinder.webp" alt="slices of a duocylinder, morphing from a disc to a rectangle as the slicing plane tilts from one factor into the other" width="256">
///
/// A thin wrapper over [`ProductBall`], hence the exact signed distance and
/// 1-Lipschitz. Its boundary is two 3-manifolds meeting along the
/// [`CliffordTorus`].
pub fn duocylinder<T: Real>(r1: T, r2: T) -> ProductBall<[(usize, T); 2]> {
  ProductBall { spec: [(2, r1), (2, r2)] }
}

/// Clifford torus: the flat 2-torus `‖p₀₁‖ = r1`, `‖p₂₃‖ = r2`, given
/// `thickness` — the ridge where the [`duocylinder`]'s two boundary pieces meet.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/clifford_torus.webp" alt="slices of a Clifford torus, morphing from an annulus into four blobs as the slicing plane tilts from one factor into the other" width="256">
///
/// `sdf(p) = ‖(‖p₀₁‖ - r1, ‖p₂₃‖ - r2)‖ - thickness/2`, the **exact** signed
/// distance: the nearest point is found by moving each 2-block radially to its
/// own radius, independently, so the two costs combine in quadrature.
/// 1-Lipschitz.
///
/// With `r1 = r2 = 1/√2` it lies on the unit 3-sphere, splitting it into two
/// congruent halves — the standard picture of the Hopf fibration's base.
#[derive(Debug, Copy, Clone)]
pub struct CliffordTorus<T> {
  pub r1: T,
  pub r2: T,
  pub thickness: T
}

impl<T: Real> BoundingBox<T, 4> for CliffordTorus<T> {
  fn bounding_box(&self) -> Aabb<T, 4> {
    let two = T::one() + T::one();
    let h = self.thickness / two;
    let half = Vector::<T, 4>::from([
      self.r1 + h, self.r1 + h, self.r2 + h, self.r2 + h]);
    Aabb::new(Point::from(half.map(|x| -x)), Point::from(half))
  }}

impl<T: Real> SDF<T, 4> for CliffordTorus<T> {
  fn sdf(&self, pixel: Point<T, 4>) -> T {
    let two = T::one() + T::one();
    let block = |a: usize, b: usize|
      (pixel[a] * pixel[a] + pixel[b] * pixel[b]).sqrt();
    V2::new(block(0, 1) - self.r1, block(2, 3) - self.r2).length()
      - self.thickness / two
  }
}

impl<T: Float> Lipschitz<T> for CliffordTorus<T> {
  fn lipschitz(&self) -> T { T::one() }
}
