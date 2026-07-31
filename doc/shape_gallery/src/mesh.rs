//! Turn a 3D signed distance field into a quad mesh, and write it as Wavefront
//! OBJ for Blender — the same conventions as `doc/video2/src/format/obj.rs`
//! (one `o` group, `v` + `vn` per vertex, `f a//a` faces), so the same import
//! settings work for both.
//!
//! The algorithm is *dual contouring*. Sample the field on a grid; every cell
//! whose corners disagree in sign owns one vertex, and every grid edge that
//! changes sign becomes a quad joining the four cells around it. That much is
//! shared with marching cubes, but needs no 256-entry case table and yields
//! well-shaped quads rather than slivers.
//!
//! What the cell's vertex *is* decides whether sharp edges survive. Taking the
//! mean of the crossing points — plain surface nets — cannot represent a crease:
//! the vertex is pulled to whichever face has more crossings in that cell, which
//! alternates from cell to cell, and a polytope edge comes out as a half-cell
//! zigzag that reads as a dotted line at any resolution fine enough to matter.
//! So instead each crossing contributes its tangent *plane* — the point plus the
//! field's gradient there — and the vertex is the least-squares intersection of
//! all of them:
//!
//! ```text
//! x = argmin Σ (nᵢ · (x - pᵢ))²    ⟹   (Σ nᵢnᵢᵀ) x = Σ nᵢ(nᵢ · pᵢ)
//! ```
//!
//! One plane leaves the vertex free to slide within it, two pin it to their line
//! of intersection, three to a corner — which is exactly the behaviour wanted:
//! flat where the surface is flat, and snapped onto edges and corners where they
//! exist.
//!
//! Getting that right at a hard edge turns out to be four separate things, and
//! each of them shows up as its own artefact when it is wrong:
//!
//!   - **The singular directions.** `Σ nᵢnᵢᵀ` is rank 1 on a flat patch and rank 2
//!     along an edge, so it cannot simply be inverted. Adding `λI` to make it
//!     invertible resolves the free directions toward wherever the constant term
//!     leans, which is an arbitrary point along the face or the edge; vertices end
//!     up scattered. Solved instead for the displacement from the crossings'
//!     centroid, with only the eigen-directions above a relative cutoff inverted —
//!     the minimum-norm answer, and the vertex stays put along whatever the planes
//!     leave undetermined.
//!   - **Where the vertex may go.** Strictly its own cell. The surface passes
//!     through that cell, so the feature the fit aims at passes through it too, and
//!     even a quarter-cell of slack lets two neighbouring vertices swap places,
//!     which inverts the quad between them.
//!   - **How the quad is cut.** Its four vertices come from four cells and are not
//!     coplanar; where the surface turns a right angle they straddle two faces and
//!     the quad is twisted, so one diagonal folds and the other does not. Cutting
//!     on shape alone gets that wrong half the time ([`Mesh::split`]).
//!   - **Which normal a face uses.** One normal per vertex cannot serve both sides
//!     of a crease, and the one it gets is the mean of whichever crossings its own
//!     cell happened to hold — a mix that changes from cell to cell, which shades a
//!     hard edge like static. Faces keep the analytic vertex normal only where the
//!     two agree, and use their own plane where they do not.
//!
//! Normals are otherwise the analytic gradient, never the polygons: on a smooth
//! patch a sphere shades perfectly round independent of the grid, and a facet
//! perfectly flat.
//!
//! What none of that reaches is the grid itself. A crease is resolved to about one
//! cell and wobbles by that much, and where a wedge is thinner than a cell the
//! corner signs simply do not describe the surface — no vertex placement recovers
//! what was never sampled. That is a resolution problem, and the caller is expected
//! to spend resolution on it where [`Mesh::crease_scale`] says it will help.
//!
//! Each vertex also carries a *crease intensity* ([`crease_field`]), written into
//! the OBJ's vertex colour. The renderer makes the shape semi-transparent and
//! lights the creases from that, so the glow is a property of the geometry — of
//! where the surface stops being smooth — and not of where the viewer happens to
//! be standing.

use {
  nalgebra::Matrix3,
  rayon::prelude::*,
  space_filling::geometry::{Point, Vector, VectorExt},
  std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
  },
};

type P3 = Point<f64, 3>;
type V3 = Vector<f64, 3>;

/// Corner `c` of a cell sits at offset `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`.
fn corner(c: usize) -> V3 {
  V3::new((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64)
}

/// Above this agreement between a face's own normal and a vertex's analytic one,
/// the face uses the vertex normal and shades smoothly; below it, the face uses
/// its own. 25 degrees.
const SMOOTH_COS: f64 = 0.906;

/// The 12 cell edges: corner pairs differing in exactly one bit, each once.
const EDGES: [(usize, usize); 12] = [
  (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
  (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
];

/// One cell's vertex, as solved.
struct Solved {
  pos: P3,
  /// Mean of the cell's crossing points — inside the cell by construction, and
  /// monotone in the cell index, which is what makes it a safe fallback.
  centroid: P3,
  normal: V3,
  /// whether the least-squares position had to be clamped into the cell
  escaped: bool,
}

/// What went wrong, if anything. Every field here is a defect that would
/// otherwise reach the render unnoticed.
pub struct Diagnostics {
  /// Whether the surface stayed clear of the meshing box. A shape that reaches
  /// the boundary is being clipped.
  pub contained: bool,
  /// Vertices whose least-squares position fell outside their own cell and had to
  /// be clamped. A handful is normal at a sharp corner; a large share means the
  /// fits are degenerate and the surface is under-resolved.
  pub clamped: usize,
  /// Faces wound against their own corner normals before repair.
  pub inverted_before: usize,
  /// Vertices retreated to their cell centroid to undo those inversions.
  pub repaired: usize,
  /// Faces still wound inside-out after repair. Should be zero.
  pub inverted: usize,
}

pub struct Mesh {
  pub verts: Vec<P3>,
  /// Outward unit normal per vertex, index-parallel to `verts`.
  pub normals: Vec<V3>,
  pub quads: Vec<[u32; 4]>,
  /// Per-quad outward direction, as `±(axis + 1)`: which grid axis the quad's
  /// dual edge runs along, signed by which way the field increases across it.
  ///
  /// This is what makes a face's correct orientation decidable. Comparing the
  /// geometry against the mean of the corner normals does not work where it
  /// matters: at a wedge thinner than a cell, neighbouring cells sample opposite
  /// faces of the sheet, their normals very nearly cancel, and the comparison
  /// reports an inversion whichever way the face is wound. The sign field has no
  /// such ambiguity — the surface normal points the way the field increases.
  pub axis: Vec<i8>,
  /// Per-vertex crease intensity in `0..1`, index-parallel to `verts` — see
  /// [`crease_field`]. The renderer drives emission and opacity from it, so the
  /// glow follows the shape's own geometry rather than the viewing angle.
  pub edge: Vec<f64>,
  /// The same measure before the ramp: raw normal deviation per cell, in radians
  /// give or take. Kept because its *scaling* is what tells a crease from a tight
  /// curve — see [`Mesh::crease_scale`].
  pub dev: Vec<f64>,
}

/// How sharp the surface is at each vertex, in `0..1`.
///
/// The quantity is the root-mean-square deviation of the unit normal over the
/// vertex's one-ring, `√(mean_j ‖n_j - n_i‖²)`, which is the Dirichlet energy
/// density `|∇n|²` of the normal field in discrete form. It is dimensionless, and
/// it separates the two cases cleanly: across a smooth patch neighbouring normals
/// differ by the turning over one cell, `≈ h·κ`, whereas across a crease they
/// differ by the dihedral angle itself no matter how fine the grid — at these
/// resolutions a factor of roughly seven.
///
/// Note that the *Laplacian* of the normal field is the wrong statistic here even
/// though a crease is exactly where it blows up: taking `mean_j(n_j) - n_i` at a
/// symmetric crease cancels, the two faces' normals averaging back to the
/// bisector the vertex already has. The deviation, being a sum of squares, cannot
/// cancel.
///
/// The result is then dilated along the mesh, since a crease is one cell wide and
/// would otherwise render as a hairline rather than something that reads as a
/// glow.
fn crease_field(
  verts: usize,
  normals: &[V3],
  quads: &[[u32; 4]],
  step: f64,
) -> (Vec<f64>, Vec<f64>) {
  // Where the ramp sits, in units of normal deviation per cell. A flat facet gives
  // 0 and a gently curved patch almost as little, so both stay dark; a tube of
  // radius 0.1 gives about 0.07 and glows faintly; a right-angled crease gives
  // 0.54 and saturates long before. So "high" and "discontinuous" both light up,
  // in that order of intensity.
  //
  // `HI` is deliberately well below what a sharp crease produces, because the
  // interesting polytopes are not sharp: cross-sections of the 600-cell meet at
  // shallow dihedrals, a few degrees in places, and a ramp scaled to right angles
  // leaves their edge graph as a dotted suggestion. Bringing the top down to 0.22
  // — around 13° — saturates those too, while a torus at 0.02 stays under `LO`.
  const LO: f64 = 0.035;
  const HI: f64 = 0.22;
  // How far the glow bleeds along the surface, and by what factor per step.
  //
  // In world units, not cells: the resolution is chosen per shape now, so counting
  // cells would draw a thinner line on whichever shapes earned a finer grid — the
  // opposite of what a consistent set of pictures needs. About six pixels at the
  // published size, which is enough to read as a line and no more, now that the
  // faces meeting a crease are genuinely flat and the crease is not being widened
  // to cover for a jittering normal.
  const GLOW_WIDTH: f64 = 0.013;
  const DECAY: f64 = 0.55;
  let dilate = (GLOW_WIDTH / step).round().max(1.0) as usize;

  // Sum of squared normal differences per vertex, accumulated over the quads'
  // own edges — no adjacency structure needed, and an interior edge being
  // counted once per incident quad is uniform, so the mean is unaffected.
  let mut acc = vec![0f64; verts];
  let mut count = vec![0f64; verts];
  let mut edges: Vec<(u32, u32)> = Vec::with_capacity(quads.len() * 4);
  for q in quads {
    for k in 0..4 {
      let (u, v) = (q[k], q[(k + 1) % 4]);
      let d = (normals[v as usize] - normals[u as usize]).dot(
        &(normals[v as usize] - normals[u as usize]),
      );
      for i in [u, v] {
        acc[i as usize] += d;
        count[i as usize] += 1.0;
      }
      edges.push((u, v));
    }
  }

  let smoothstep = |x: f64| {
    let t = ((x - LO) / (HI - LO)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
  };
  let dev: Vec<f64> = (0..verts)
    .map(|i| if count[i] == 0.0 { 0.0 } else { (acc[i] / count[i]).sqrt() })
    .collect();
  let mut e: Vec<f64> = dev.iter().copied().map(smoothstep).collect();

  let mut next = e.clone();
  for _ in 0..dilate {
    next.copy_from_slice(&e);
    for &(u, v) in &edges {
      let (u, v) = (u as usize, v as usize);
      next[u] = next[u].max(DECAY * e[v]);
      next[v] = next[v].max(DECAY * e[u]);
    }
    std::mem::swap(&mut e, &mut next);
  }

  // Then soften it. Dilating by a maximum advances the band's edge one whole cell
  // at a time, so its boundary is a staircase — and since the renderer turns this
  // value into light, that staircase shows up as a serrated fringe along every
  // glowing line, which looks exactly like a meshing fault and is not one. A couple
  // of averaging passes make the field continuous across the surface; the core
  // barely moves, being a plateau several cells wide.
  const SOFTEN: usize = 2;
  let mut sum = vec![0f64; verts];
  let mut deg = vec![0f64; verts];
  for _ in 0..SOFTEN {
    sum.iter_mut().for_each(|x| *x = 0.0);
    deg.iter_mut().for_each(|x| *x = 0.0);
    for &(u, v) in &edges {
      let (u, v) = (u as usize, v as usize);
      sum[u] += e[v];
      sum[v] += e[u];
      deg[u] += 1.0;
      deg[v] += 1.0;
    }
    for i in 0..verts {
      if deg[i] > 0.0 {
        e[i] = 0.5 * (e[i] + sum[i] / deg[i]);
      }
    }
  }
  (e, dev)
}

impl Mesh {
  pub fn is_empty(&self) -> bool {
    self.quads.is_empty()
  }

  /// How sharp the sharpest part of this surface is, at this resolution: the
  /// 99th percentile of the raw per-cell normal deviation.
  ///
  /// On its own the number means little, because at a coarse spacing a tight
  /// smooth curve and a crease look alike — both turn the normal a lot from one
  /// cell to the next. What separates them is how that changes when the grid is
  /// refined. Across a smooth patch the deviation is the turning over one cell,
  /// `h·κ`, so it halves when `h` does; across a crease it is the dihedral angle,
  /// which is a property of the shape and does not move. Comparing this figure at
  /// two resolutions therefore answers the question a single measurement cannot.
  ///
  /// A percentile rather than a mean, because a shape is usually a mixture — a
  /// cylinder is smooth walls plus two rim creases — and the mean would be swamped
  /// by whichever part has more area, when what matters is whether there is any
  /// crease at all.
  pub fn crease_scale(&self) -> f64 {
    if self.dev.is_empty() {
      return 0.0;
    }
    let mut d = self.dev.clone();
    let k = ((d.len() as f64 * 0.99) as usize).min(d.len() - 1);
    let (_, nth, _) = d.select_nth_unstable_by(k, f64::total_cmp);
    *nth
  }

  /// Radius of the bounding sphere about the bounding-box centre — what the
  /// camera needs in order to frame the shape from any direction.
  pub fn radius(&self) -> f64 {
    if self.verts.is_empty() {
      return 0.0;
    }
    let (mut lo, mut hi) = (self.verts[0], self.verts[0]);
    for v in &self.verts {
      for a in 0..3 {
        lo[a] = lo[a].min(v[a]);
        hi[a] = hi[a].max(v[a]);
      }
    }
    let c = (lo.coords + hi.coords) * 0.5;
    self.verts.iter().map(|v| (v.coords - c).length()).fold(0.0, f64::max)
  }

  /// Whether triangle `t` is wound against the direction the field increases.
  ///
  /// A quad is dual to a grid edge the surface crosses, so the surface there is
  /// roughly perpendicular to that edge and its normal has a solid component along
  /// it. The field increases outward, which fixes the sign that component must
  /// have — and unlike the corner normals, the field cannot be ambiguous about it.
  /// The comparison is relative to the triangle's own area, and strict, so that a
  /// *degenerate* face is not mistaken for a backwards one. They arise for real:
  /// where the surface has a flat part square to a grid axis — the cap of the
  /// duocylinder's cross-section, say — the crossings land exactly on cell
  /// boundaries and the face collapses to zero area in projection. Such a face
  /// draws nothing and is nobody's problem, whereas counting it as an inversion
  /// sends the repair pass chasing something it cannot fix.
  fn tri_wrong_way(&self, t: &[u32; 3], ax: i8) -> bool {
    let p = t.map(|i| self.verts[i as usize].coords);
    let geo = (p[1] - p[0]).cross(&(p[2] - p[0]));
    let along = geo[(ax.unsigned_abs() - 1) as usize] * f64::from(ax.signum());
    along < -1e-6 * geo.norm()
  }

  /// Whether quad `k` is inside-out however it is cut.
  fn is_inverted(&self, k: usize) -> bool {
    self.split(k).is_none()
  }

  /// Faces that render inside-out.
  pub fn inverted_faces(&self) -> usize {
    (0..self.quads.len()).filter(|&k| self.is_inverted(k)).count()
  }

  /// Quad `k` cut into two correctly-wound triangles, if either diagonal manages
  /// it; the shorter diagonal when both do.
  ///
  /// A dual-contoured quad's four vertices come from four different cells and need
  /// not be coplanar. Where the surface turns a right angle — the rim of a
  /// cylinder, say, where the wall meets the cap — two of them sit on one face and
  /// two on the other, and the quad is not merely non-planar but twisted. One of
  /// its two triangulations then folds back on itself while the other does not, so
  /// which diagonal is chosen decides whether the mesh is sound there. Choosing on
  /// shape alone, as is usual, gets it wrong about half the time.
  fn split(&self, k: usize) -> Option<[[u32; 3]; 2]> {
    let q = self.quads[k];
    let ax = self.axis[k];
    let p = q.map(|i| self.verts[i as usize].coords);
    let candidates = [
      [[q[0], q[1], q[2]], [q[0], q[2], q[3]]],
      [[q[0], q[1], q[3]], [q[1], q[2], q[3]]],
    ];
    // index 0 cuts along 0-2, index 1 along 1-3
    let shorter =
      usize::from((p[3] - p[1]).norm_squared() < (p[2] - p[0]).norm_squared());
    [shorter, 1 - shorter]
      .into_iter()
      .map(|i| candidates[i])
      .find(|pair| !pair.iter().any(|t| self.tri_wrong_way(t, ax)))
  }

  /// The vertices of every inverted face.
  fn inverted_indices(&self) -> Vec<usize> {
    (0..self.quads.len())
      .filter(|&k| self.is_inverted(k))
      .flat_map(|k| self.quads[k].map(|i| i as usize))
      .collect()
  }

  /// Every quad as two triangles, cut so as not to fold ([`Mesh::split`]).
  ///
  /// Triangles rather than quads because a renderer handed a non-planar quad picks
  /// a diagonal of its own — and not the same one in the importer as in the
  /// raytracer.
  pub fn triangles(&self) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(self.quads.len() * 2);
    for k in 0..self.quads.len() {
      let q = self.quads[k];
      let p = q.map(|i| self.verts[i as usize].coords);
      let fallback = if (p[2] - p[0]).norm_squared() <= (p[3] - p[1]).norm_squared() {
        [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
      } else {
        [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
      };
      out.extend(self.split(k).unwrap_or(fallback));
    }
    out
  }

  /// Mean agreement between each quad's geometric normal and its vertices'
  /// analytic ones. Near +1 when the winding is outward; a self-check on the
  /// quad orientation rather than something the renderer needs.
  pub fn winding_agreement(&self) -> f64 {
    if self.quads.is_empty() {
      return 1.0;
    }
    let sum: f64 = self
      .quads
      .iter()
      .map(|q| {
        let p = q.map(|i| self.verts[i as usize].coords);
        let geo = (p[1] - p[0]).cross(&(p[2] - p[0])).robust_normalize();
        let ana = q
          .iter()
          .map(|&i| self.normals[i as usize])
          .fold(V3::zeros(), |a, b| a + b)
          .robust_normalize();
        geo.dot(&ana)
      })
      .sum();
    sum / self.quads.len() as f64
  }
}

/// Mesh the zero level set of `f` inside the cube `[-half, half]³`, on a
/// `res`-cell grid.
///
/// Returns the mesh and a [`Diagnostics`] the caller is expected to look at: the
/// ways this can go wrong are all silent in the output, so they are counted.
pub fn dual_contour(
  f: &(dyn Fn(P3) -> f64 + Send + Sync),
  half: f64,
  res: usize,
) -> (Mesh, Diagnostics) {
  let n = res + 1; // samples per axis
  let step = 2.0 * half / res as f64;
  let at_pos = |i: usize| -half + step * i as f64;

  // The samples are held as `f32`. They are only ever used to decide a sign and to
  // interpolate a crossing along one cell edge, both of which f32 resolves to
  // within a ten-thousandth of a cell; every gradient and every Newton-style
  // refinement calls the field directly in f64. Halving this array is what makes
  // the finer grids below fit — at 512 it would otherwise be a gigabyte on its own.
  let mut g = vec![0f32; n * n * n];
  g.par_chunks_mut(n * n).enumerate().for_each(|(k, slab)| {
    let z = at_pos(k);
    for j in 0..n {
      let y = at_pos(j);
      for i in 0..n {
        slab[j * n + i] = f(P3::new(at_pos(i), y, z)) as f32;
      }
    }
  });
  let at = |i: usize, j: usize, k: usize| f64::from(g[i + j * n + k * n * n]);

  // the surface is clear of the box iff no boundary sample is inside
  let mut contained = true;
  for k in 0..n {
    for j in 0..n {
      for i in 0..n {
        if (i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1)
          && at(i, j, k) <= 0.0
        {
          contained = false;
        }
      }
    }
  }

  // Which cells own a vertex. Split from the solve below so the expensive part —
  // a gradient per edge crossing, and a field like the 600-cell costs 600
  // half-space tests per evaluation — runs in parallel.
  // Keyed sparsely rather than as a dense grid: the surface touches well under a
  // percent of the cells, and a dense `u32` per cell would cost more than the
  // samples themselves at the resolutions this runs at.
  let cell_key = |x: usize, y: usize, z: usize| (x + (y + z * res) * res) as u32;
  let mut cell_vert: HashMap<u32, u32> = HashMap::new();
  let mut active: Vec<[usize; 3]> = vec![];
  for z in 0..res {
    for y in 0..res {
      for x in 0..res {
        let inside = (0..8)
          .filter(|c| at(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1)) < 0.0)
          .count();
        if inside == 0 || inside == 8 {
          continue;
        }
        cell_vert.insert(cell_key(x, y, z), active.len() as u32);
        active.push([x, y, z]);
      }
    }
  }

  // Central-difference half-width. An eighth of a cell, not a half: the width is
  // how far a crease can reach across the difference and corrupt the plane fed to
  // the QEF, and only crossings within `h` of one are affected. f64 has plenty of
  // headroom for the smaller denominator.
  let h = step * 0.125;
  let solved: Vec<Solved> = active
    .par_iter()
    .map(|&[x, y, z]| {
      let origin = V3::new(at_pos(x), at_pos(y), at_pos(z));
      let d: [f64; 8] = std::array::from_fn(|c| {
        at(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1))
      });

      // the crossing points, and the tangent plane of the surface at each
      let mut ata = Matrix3::zeros();
      let mut atb = V3::zeros();
      let mut centroid = V3::zeros();
      let mut normal = V3::zeros();
      let mut hits = 0.0;
      for (a, b) in EDGES {
        if (d[a] < 0.0) == (d[b] < 0.0) {
          continue;
        }
        let t = d[a] / (d[a] - d[b]);
        let (ca, cb) = (corner(a), corner(b));
        let p = origin + (ca + (cb - ca) * t) * step;
        let grad = V3::from_fn(|i, _| {
          let mut e = V3::zeros();
          e[i] = h;
          (f(P3::from(p + e)) - f(P3::from(p - e))) / (2.0 * h)
        })
        .robust_normalize();
        ata += grad * grad.transpose();
        atb += grad * grad.dot(&p);
        centroid += p;
        normal += grad;
        hits += 1.0;
      }
      centroid /= hits;

      // Solve for the displacement from the centroid, not for the point itself,
      // and invert only the directions the planes actually constrain.
      //
      // `ata` is rank-deficient wherever the surface is not a corner: rank 1 on a
      // flat patch, rank 2 along an edge. Adding `λI` to make it invertible — the
      // obvious fix — resolves those free directions toward whatever the constant
      // term happens to prefer, which is an arbitrary point somewhere along the
      // face or the edge. Truncating instead gives the *minimum-norm* displacement:
      // the vertex slides only along constrained directions and otherwise stays at
      // the centroid. The threshold is relative to the largest eigenvalue, since
      // the eigenvalues scale with how many crossings this cell happened to have.
      let eig = ata.symmetric_eigen();
      let residual = atb - ata * centroid;
      let cutoff = 0.05 * eig.eigenvalues.iter().fold(0.0f64, |a, &b| a.max(b));
      let mut delta = V3::zeros();
      for i in 0..3 {
        let sigma = eig.eigenvalues[i];
        if sigma > cutoff {
          let u = eig.eigenvectors.column(i);
          delta += u * (u.dot(&residual) / sigma);
        }
      }

      // Strictly inside the cell. The surface passes through this cell, so the
      // feature the QEF is aiming at — a face, an edge line, a corner — passes
      // through it too, and a solution outside means the fit was degenerate.
      // Allowing even a quarter-cell of slack lets two neighbouring vertices swap
      // places, which inverts the quad between them: that is where flipped and
      // self-intersecting faces come from.
      let want = centroid + delta;
      let pos = V3::from_fn(|i, _| want[i].clamp(origin[i], origin[i] + step));
      Solved {
        pos: P3::from(pos),
        centroid: P3::from(centroid),
        normal: normal.robust_normalize(),
        escaped: (pos - want).norm() > 1e-12,
      }
    })
    .collect();

  // a quad around every sign-changing grid edge, over the 4 cells sharing it
  let mut quads = vec![];
  let mut axes: Vec<i8> = vec![];
  for z in 0..n {
    for y in 0..n {
      for x in 0..n {
        let p = [x, y, z];
        for a in 0..3 {
          let (b, c) = ((a + 1) % 3, (a + 2) % 3);
          // the edge needs its far sample, and both neighbour axes a cell on
          // each side; the surface never reaches the boundary, so nothing is
          // lost by skipping the rim
          if p[a] + 1 >= n || p[b] == 0 || p[c] == 0 || p[b] >= res || p[c] >= res {
            continue;
          }
          let mut q = p;
          q[a] += 1;
          let (d0, d1) = (at(p[0], p[1], p[2]), at(q[0], q[1], q[2]));
          if (d0 < 0.0) == (d1 < 0.0) {
            continue;
          }
          let mut idx = [u32::MAX; 4];
          for (m, (db, dc)) in [(0, 0), (1, 0), (1, 1), (0, 1)].into_iter().enumerate() {
            let mut cm = p;
            cm[b] -= db;
            cm[c] -= dc;
            if let Some(&v) = cell_vert.get(&cell_key(cm[0], cm[1], cm[2])) {
              idx[m] = v;
            }
          }
          if idx.contains(&u32::MAX) {
            continue;
          }
          // The (db, dc) cycle runs counter-clockwise seen from -a, so it faces
          // outward already when the far sample is the interior one.
          if d1 < 0.0 {
            idx.reverse();
          }
          quads.push(idx);
          axes.push((a as i8 + 1) * if d1 > 0.0 { 1 } else { -1 });
        }
      }
    }
  }

  let mut verts: Vec<P3> = solved.iter().map(|s| s.pos).collect();
  let normals: Vec<V3> = solved.iter().map(|s| s.normal).collect();

  // Repair inversions by retreating to the centroid.
  //
  // What is left after the strict clamp is not a placement mistake: it is a wedge
  // thinner than a cell — the gyroid sheet leaving its clipping ball at a glancing
  // angle, the sweep of a star's point — where the corner signs genuinely do not
  // describe the surface, and the four cells around an edge do not come out in a
  // convex cycle. No vertex position fixes that, and no attainable resolution
  // removes it entirely at a tangency.
  //
  // But the centroid of a cell's crossings is monotone in the cell index, so a
  // quad whose four vertices all sit at their centroids cannot invert. Retreating
  // just the offending vertices trades a little sharpness at those few places for
  // topology that is right everywhere, which is the correct way round: a sharp
  // crease that is inside-out reads far worse than a slightly soft one. Repeated,
  // because moving a vertex can invert a neighbouring quad.
  let inverted_before = {
    let probe = Mesh {
      verts: verts.clone(),
      normals: normals.clone(),
      quads: quads.clone(),
      axis: axes.clone(),
      edge: vec![],
      dev: vec![],
    };
    probe.inverted_faces()
  };
  let mut repaired = 0usize;
  for _ in 0..4 {
    let probe = Mesh {
      verts: verts.clone(),
      normals: normals.clone(),
      quads: quads.clone(),
      axis: axes.clone(),
      edge: vec![],
      dev: vec![],
    };
    let bad = probe.inverted_indices();
    if bad.is_empty() {
      break;
    }
    for i in bad {
      if verts[i] != solved[i].centroid {
        verts[i] = solved[i].centroid;
        repaired += 1;
      }
    }
  }

  let (edge, dev) = crease_field(verts.len(), &normals, &quads, step);
  let mesh = Mesh { verts, normals, quads, axis: axes, edge, dev };
  let stats = Diagnostics {
    contained,
    clamped: solved.iter().filter(|s| s.escaped).count(),
    inverted_before,
    repaired,
    inverted: mesh.inverted_faces(),
  };
  (mesh, stats)
}

/// sRGB encode, so a value survives the OBJ vertex-colour round trip.
///
/// Blender's importer treats the three optional floats on a `v` line as an sRGB
/// colour and decodes them to linear on the way in. Pre-encoding here cancels
/// that exactly, so the attribute the shader samples is the number written.
fn linear_to_srgb(x: f64) -> f64 {
  let x = x.clamp(0.0, 1.0);
  if x <= 0.003_130_8 {
    x * 12.92
  } else {
    1.055 * x.powf(1.0 / 2.4) - 0.055
  }
}

/// Write `mesh` as a single-object OBJ, with [`Mesh::edge`] in the vertex colour.
pub fn write_obj(path: &Path, name: &str, mesh: &Mesh) -> anyhow::Result<()> {
  let mut w = BufWriter::new(File::create(path)?);
  writeln!(w, "# space-filling shape gallery: {name}")?;
  writeln!(w, "# vertex colour carries the crease intensity, sRGB-encoded")?;
  writeln!(w, "o {name}")?;
  for (v, &e) in mesh.verts.iter().zip(&mesh.edge) {
    let c = linear_to_srgb(e);
    writeln!(w, "v {:.5} {:.5} {:.5} {c:.4} {c:.4} {c:.4}", v.x, v.y, v.z)?;
  }
  // Split normals, by the usual smoothing-angle rule but measured against the
  // *analytic* normal rather than an averaged one.
  //
  // A crease vertex is shared by the faces on both sides of the crease, and one
  // normal cannot serve both. Worse, the one it gets is the mean of whichever
  // crossings its own cell happened to contain, and that mix changes from cell to
  // cell along the crease — which is what makes a hard edge shade like static.
  // So a face keeps the vertex normal only where the two agree, and falls back to
  // its own plane where they do not: facets stay flat right up to the crease, and
  // the strip of faces straddling it reads as one clean narrow bevel.
  let tris = mesh.triangles();
  let mut face_normals: Vec<V3> = vec![];
  let mut corners: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
  for t in &tris {
    let p = t.map(|i| mesh.verts[i as usize].coords);
    let face = (p[1] - p[0]).cross(&(p[2] - p[0])).robust_normalize();
    let mut own = None;
    corners.push(std::array::from_fn(|k| {
      let v = t[k] as usize;
      if mesh.normals[v].dot(&face) >= SMOOTH_COS {
        v
      } else {
        *own.get_or_insert_with(|| {
          face_normals.push(face);
          mesh.verts.len() + face_normals.len() - 1
        })
      }
    }));
  }
  for n in mesh.normals.iter().chain(&face_normals) {
    writeln!(w, "vn {:.4} {:.4} {:.4}", n.x, n.y, n.z)?;
  }
  for (t, c) in tris.iter().zip(&corners) {
    // OBJ indices are 1-based, and normals are indexed independently of vertices
    let (v, n) = (t.map(|i| i + 1), c.map(|i| i + 1));
    writeln!(
      w,
      "f {}//{} {}//{} {}//{}",
      v[0], n[0], v[1], n[1], v[2], n[2]
    )?;
  }
  w.flush()?;
  Ok(())
}
