//! Shapes defined for **three dimensions only** (`D = 3`).
//!
//! These are the shapes with no analogue in other dimensions. Of the five
//! Platonic solids, three *do* generalize and live in [`dn`](super::dn) instead
//! — the tetrahedron is [`simplex`](super::dn::simplex), the cube is
//! [`Hypersquare`](super::dn::Hypersquare), the octahedron is
//! [`Orthoplex`](super::dn::Orthoplex) — while the **dodecahedron** and
//! **icosahedron** are exceptional to 3-space, existing only because the
//! icosahedral symmetry group does.
//!
//! The other two residents are the space-*filling* polyhedra: the
//! [`truncated_octahedron`] and [`rhombic_dodecahedron`] tile 3-space by
//! translation all by themselves, being the Voronoi cells of the BCC and FCC
//! lattices — the perfect, degenerate case of the problem this crate solves.
//! (The truncated octahedron is also the 3D
//! [`permutohedron`](super::dn::permutohedron), so that one does generalize;
//! it is repeated here for its name and its lattice.)
//!
//! Every polyhedron here is a [`Polytope`] built by [`convex_hull`] from a
//! vertex orbit and the dual's vertex orbit as facet normals: exact inside,
//! conservative outside, 1-Lipschitz. Round one with
//! [`offset`](crate::geometry::Combinator::offset) to make it exact
//! everywhere.

use {
  super::dn::{convex_hull, orbit, unit_circumradius, HalfSpace, Polyline, Polytope},
  crate::geometry::{Point, Real, Vector},
};

/// The golden ratio, `(1 + √5)/2` — the source of icosahedral symmetry.
fn phi<T: Real>() -> T {
  let five = T::from(5.0).unwrap();
  (T::one() + five.sqrt()) / (T::one() + T::one())
}

/// The 12 vertex directions of a regular icosahedron: the cyclic permutations
/// of `(0, ±1, ±φ)`.
pub fn icosahedron_vertices<T: Real>() -> Vec<Vector<T, 3>> {
  let (zero, one, f) = (T::zero(), T::one(), phi::<T>());
  let mut out = vec![];
  // cyclic, not all, permutations — the icosahedron's 12 vertices form three
  // orthogonal golden rectangles
  for rot in 0..3 {
    let pattern = [zero, one, f];
    let rotated: [T; 3] = std::array::from_fn(|a| pattern[(a + rot) % 3]);
    let mut orbit_out = vec![];
    super::dn::sign_orbit(&rotated, &mut orbit_out);
    out.extend(orbit_out);
  }
  super::dn::dedup_directions(&mut out);
  unit_circumradius(&mut out);
  out
}

/// The 20 vertex directions of a regular dodecahedron: `(±1, ±1, ±1)` together
/// with the cyclic permutations of `(0, ±φ, ±1/φ)`.
///
/// Note the order within that second orbit. A dual pair is only dual in one
/// relative orientation, and `(0, ±1/φ, ±φ)` — the listing one might reach for
/// first — is the mirror image: its dodecahedron is perfectly regular, but the
/// [`icosahedron_vertices`] then point at its *vertices* rather than its faces,
/// so using them as facet normals would inscribe a larger, wrong polytope.
pub fn dodecahedron_vertices<T: Real>() -> Vec<Vector<T, 3>> {
  let (zero, one, f) = (T::zero(), T::one(), phi::<T>());
  let mut out = vec![];
  super::dn::sign_orbit(&[one, one, one], &mut out);
  for rot in 0..3 {
    let pattern = [zero, f, one / f];
    let rotated: [T; 3] = std::array::from_fn(|a| pattern[(a + rot) % 3]);
    super::dn::sign_orbit(&rotated, &mut out);
  }
  super::dn::dedup_directions(&mut out);
  unit_circumradius(&mut out);
  out
}

/// Regular icosahedron inscribed in the unit sphere — 20 triangular facets,
/// whose normals are the [`dodecahedron_vertices`] (the dual).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/icosahedron.webp" alt="z-slices of an icosahedron" width="256">
pub fn icosahedron<T: Real>() -> Polytope<Vec<HalfSpace<T, 3>>> {
  convex_hull(dodecahedron_vertices::<T>(), &icosahedron_vertices::<T>())
}

/// Regular dodecahedron inscribed in the unit sphere — 12 pentagonal facets,
/// whose normals are the [`icosahedron_vertices`] (the dual).
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/dodecahedron.webp" alt="z-slices of a dodecahedron" width="256">
pub fn dodecahedron<T: Real>() -> Polytope<Vec<HalfSpace<T, 3>>> {
  convex_hull(icosahedron_vertices::<T>(), &dodecahedron_vertices::<T>())
}

/// Truncated octahedron — 14 facets (6 square, 8 hexagonal), the Voronoi cell
/// of the BCC lattice, and one of the few polyhedra that **tiles 3-space
/// alone**.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/truncated_octahedron.webp" alt="z-slices of a truncated octahedron" width="256">
///
/// Vertices are all permutations of `(0, ±1, ±2)`; facet normals are the 6 axis
/// directions and the 8 cube diagonals. Identical up to scale to the 3D
/// [`permutohedron`](super::dn::permutohedron), which is the same construction
/// carried to every dimension.
pub fn truncated_octahedron<T: Real>() -> Polytope<Vec<HalfSpace<T, 3>>> {
  let (zero, one, two) = (T::zero(), T::one(), T::one() + T::one());
  let mut vertices = vec![];
  orbit([zero, one, two], false, &mut vertices);
  unit_circumradius(&mut vertices);

  let mut normals = vec![];
  orbit([zero, zero, one], false, &mut normals); // ±eₐ
  super::dn::sign_orbit(&[one, one, one], &mut normals); // cube diagonals
  super::dn::dedup_directions(&mut normals);
  convex_hull(normals, &vertices)
}

/// Rhombic dodecahedron — 12 rhombic facets, the Voronoi cell of the FCC
/// lattice, and the other polyhedron that **tiles 3-space alone**.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/rhombic_dodecahedron.webp" alt="z-slices of a rhombic dodecahedron" width="256">
///
/// Vertices are `(±1, ±1, ±1)` together with `(±2, 0, 0)` and its permutations
/// — two orbits at different radii, so only the six axial vertices reach the
/// unit sphere. Facet normals are the 12 permutations of `(±1, ±1, 0)`.
pub fn rhombic_dodecahedron<T: Real>() -> Polytope<Vec<HalfSpace<T, 3>>> {
  let (zero, one, two) = (T::zero(), T::one(), T::one() + T::one());
  let mut vertices = vec![];
  super::dn::sign_orbit(&[one, one, one], &mut vertices);
  orbit([zero, zero, two], false, &mut vertices);
  super::dn::dedup_directions(&mut vertices);
  unit_circumradius(&mut vertices);

  let mut normals = vec![];
  orbit([zero, one, one], false, &mut normals);
  convex_hull(normals, &vertices)
}

/// A `(p, q)` torus knot, sampled into a closed [`Polyline`] of `segments`
/// capsules.
///
/// <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/torus_knot.webp" alt="z-slices of a (2,3) torus knot" width="256">
///
/// The knot winds `p` times around the torus's axis while looping `q` times
/// through its hole; `(2, 3)` is the trefoil. Curves like this have no
/// closed-form distance — the nearest-point problem is a root find — so the
/// honest approach is to sample the curve densely and let the exact capsule
/// field do the rest: the result is the exact distance to the *inscribed
/// polygon*, converging to the knot as `segments` grows, and 1-Lipschitz at any
/// resolution.
///
/// Scaled to unit circumradius, and closed by repeating the first sample.
///
/// ```
/// # use space_filling::geometry::*;
/// let trefoil = torus_knot::<f64>(2, 3, 256, 0.08);
/// ```
pub fn torus_knot<T: Real>(
  p: u32,
  q: u32,
  segments: usize,
  thickness: T,
) -> Polyline<Vec<Point<T, 3>>, T> {
  let tau = T::from(std::f64::consts::TAU).unwrap();
  let two = T::one() + T::one();
  let (pf, qf) = (T::from(p).unwrap(), T::from(q).unwrap());
  let n = T::from(segments.max(3)).unwrap();
  // max |point| is (2 + 1) along the tube's outer equator
  let scale = T::one() / (two + T::one());

  let mut vertices: Vec<Point<T, 3>> = (0..segments.max(3))
    .map(|i| {
      let t = tau * T::from(i).unwrap() / n;
      let r = two + (qf * t).cos();
      Point::from([
        r * (pf * t).cos() * scale,
        r * (pf * t).sin() * scale,
        -(qf * t).sin() * scale,
      ])
    })
    .collect();
  if let Some(&first) = vertices.first() {
    vertices.push(first); // close the loop
  }
  Polyline { vertices, thickness }
}
