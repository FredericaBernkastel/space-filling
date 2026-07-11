//! Adaptively Sampled Distance Field, backed by a [`quadtree`] arena, in any
//! compile-time dimension count.
//!
//! Each node (bucket) stores a handful of [`Primitive`]s — a field closure
//! together with its declared Lipschitz bound — and represents their pointwise
//! `min`. `ADF` itself implements [`SDF`], so a field composed
//! of millions of primitives is sampled in logarithmic time by descending to
//! the leaf covering the query point, rather than evaluated at quadratic cost.

#![allow(clippy::mut_from_ref)]
use {
  crate::{
    geometry::{Point, Aabb, Real, VectorExt, DistPoint},
    sdf::{SDF, Lipschitz},
  },
  quadtree::{
    Quadtree, Node, Refine, Dim, Branching, Children, child_rect, child_rects
  },
  std::{
    sync::Arc,
    fmt::{Debug, Formatter}
  },
  nalgebra::Scalar,
};

#[cfg(test)] mod tests;
pub(crate) mod quadtree;

/// An SDF primitive stored in the tree: the field function together with its
/// declared Lipschitz constant.
///
/// `lipschitz = 1` is exact for true signed-distance functions and lets the
/// redundancy test prune soundly. For a primitive whose gradient exceeds 1 —
/// e.g. an approximate or fractal distance estimator — declare a larger bound:
/// the test stays conservative for that primitive (it only ever certifies on a
/// real proof, so nothing contributing is dropped or skipped; pruning merely
/// becomes less effective the larger the bound).
#[derive(Clone)]
pub struct Primitive<Float: Scalar, const D: usize> {
  pub f: Arc<dyn Fn(Point<Float, D>) -> Float + Send + Sync>,
  pub lipschitz: Float,
}

impl<_Float: Real, const D: usize> Primitive<_Float, D> {
  /// A primitive assumed to be a true SDF (`lipschitz = 1`). For a shape type,
  /// prefer [`Self::from_shape`], which derives the bound automatically.
  pub fn new(f: impl Fn(Point<_Float, D>) -> _Float + Send + Sync + 'static) -> Self {
    Self { f: Arc::new(f), lipschitz: _Float::one() }
  }
  /// Declare the Lipschitz constant of this primitive's field.
  pub fn with_lipschitz(mut self, lipschitz: _Float) -> Self {
    self.lipschitz = lipschitz;
    self
  }
  /// Wrap a shape, deriving both the field and its Lipschitz bound from the
  /// [`SDF`] and [`Lipschitz`] impls — no manual constant. Combinator chains
  /// (translate/rotate/scale/booleans) propagate the bound of their operands,
  /// so a custom estimator declares its constant once, on the type.
  pub fn from_shape<S>(shape: S) -> Self
  where
    S: SDF<_Float, D> + Lipschitz<_Float> + Send + Sync + 'static,
  {
    let lipschitz = shape.lipschitz();
    Self { f: Arc::new(move |p| shape.sdf(p)), lipschitz }
  }
}

#[derive(Clone)]
pub struct ADF<Float: Scalar, const D: usize> {
  pub tree: Quadtree<Vec<Primitive<Float, D>>, Float, D>,
  /// Max subdivisions the redundancy test ([`sdf_geq_everywhere`])
  /// may use to prove/refute `f >= g` over a node. Higher = finer proofs (a
  /// primitive is pruned only when provably redundant to within ~`node/2^n`),
  /// at more work in near-tangent regions.
  prune_subdiv: u32,
  /// Largest Lipschitz constant ever declared by an inserted primitive; the
  /// whole field is `lipschitz_max`-Lipschitz (monotone over-approximation:
  /// primitives pruned later do not lower it).
  lipschitz_max: Float,
}

impl <_Float: Real, const D: usize> SDF<_Float, D> for &[Primitive<_Float, D>] {
  fn sdf(&self, pixel: Point<_Float, D>) -> _Float {
    self.iter()
      .map(|p| (p.f)(pixel))
      .reduce(|a, b| if a <= b { a } else { b })
      .unwrap_or(_Float::max_value() / (_Float::one() + _Float::one()))
  }
}

/// The Lipschitz constant of `min` over a bucket: `min` of `L_i`-Lipschitz
/// functions is `max(L_i)`-Lipschitz.
fn bucket_lipschitz<_Float: Real, const D: usize>(bucket: &[Primitive<_Float, D>]) -> _Float {
  bucket.iter().map(|p| p.lipschitz).fold(_Float::one(), _Float::max)
}

/// Returns `true` only when `f(v) >= g(v)` is *provable* for every `v` in
/// `domain`. Sound **provided `f` is `l_f`-Lipschitz and `g` is `l_g`-Lipschitz**
/// (true SDFs are 1-Lipschitz): then `f - g` is `(l_f + l_g)`-Lipschitz, so over
/// a box of half-diagonal `h` centred at `c`,
/// `f - g >= (f - g)(c) - (l_f + l_g)·h`. That bound *proves a sub-box
/// clean* (`(f-g)(c) - (l_f+l_g)·h >= 0`) or discards it toward a witness
/// (`(f-g)(c) < 0`); undecided sub-boxes are refined up to `max_subdiv`
/// levels, beyond which it conservatively answers `false`.
///
/// Cost is adaptive: well-separated fields settle at the root and a real witness
/// is reached by descent — no fixed grid or GD schedule. Larger constants are
/// conservative: certification just needs deeper refinement, and an overly large
/// bound degrades into "only a real witness ever decides", never unsoundness.
fn sdf_geq_everywhere<_Float, F, G, const D: usize>(
  f: F,
  g: G,
  domain: Aabb<_Float, D>,
  l_f: _Float,
  l_g: _Float,
  max_subdiv: u32,
) -> bool
where
  _Float: Real,
  F: Fn(Point<_Float, D>) -> _Float,
  G: Fn(Point<_Float, D>) -> _Float,
  Dim<D>: Branching,
{
  let two = _Float::one() + _Float::one();
  let l_sum = l_f + l_g;
  let mut stack = vec![(domain, 0u32)];
  while let Some((rect, depth)) = stack.pop() {
    let diff = f(rect.center()) - g(rect.center());
    if diff < _Float::zero() {
      return false; // witness: f < g here, so `f >= g everywhere` is false
    }
    let half_diag = rect.size().length() / two;
    if diff >= l_sum * half_diag {
      continue; // `f - g >= 0` proved over the whole box
    }
    if depth >= max_subdiv {
      return false; // undecided within budget → conservatively assume a witness
    }
    for sub in child_rects(rect) {
      stack.push((sub, depth + 1));
    }
  }
  true
}

impl <_Float: Real + Send + Sync, const D: usize> ADF<_Float, D>
where
  Dim<D>: Branching,
  // trivially satisfied — the GAT is always an array — but the compiler
  // cannot see through the projection
  Children<Vec<Primitive<_Float, D>>, D>: Send,
{
  /// Create a new ADF instance. `max_depth` specifies maximum number of tree subdivisions;
  /// `init` specifies initial sdf primitives.
  pub fn new(max_depth: u8, init: Vec<Primitive<_Float, D>>) -> Self {
    let lipschitz_max = bucket_lipschitz(&init);
    Self {
      tree: Quadtree::new(max_depth, init),
      prune_subdiv: 8,
      lipschitz_max,
    }
  }
  /// Controls precision of primitive pruning in a bucket: the redundancy test may
  /// refine a node up to `subdiv` times to prove `f >= g` (see [`sdf_geq_everywhere`]).
  pub fn with_prune_subdiv(mut self, subdiv: u32) -> Self {
    self.prune_subdiv = subdiv;
    self
  }
  /// Add a new sdf primitive function, assumed to be a true SDF (`lipschitz = 1`).
  /// See [`Self::insert_primitive_domain`] for approximate fields.
  pub fn insert_sdf_domain(
    &mut self,
    domain: Aabb<_Float, D>,
    f: Arc<dyn Fn(Point<_Float, D>) -> _Float + Send + Sync>
  ) -> bool {
    self.insert_primitive_domain(domain, Primitive { f, lipschitz: _Float::one() })
  }

  /// Add a new sdf primitive with an explicit Lipschitz bound (see [`Primitive`]).
  pub fn insert_primitive_domain(
    &mut self,
    domain: Aabb<_Float, D>,
    prim: Primitive<_Float, D>
  ) -> bool {
    self.insert_where(move |node| node.rect.intersects(&domain), prim)
  }

  /// Insert a primitive placed at a **maximum** `p` of the field (any placement
  /// with `S ⊆ B̄(x₀, d)`), without an explicit domain: the walk itself visits
  /// exactly the subtrees that can meet the update region
  ///
  /// ```text
  /// D* = { v : g(v) > |v − x₀| − d }.
  /// ```
  ///
  /// A subtree `R` is skipped once `ĝ(c_R) + L_B·h(R) ≤ dist(R, x₀) − d`, where
  /// `ĝ` is the node's own bucket field and `L_B` that bucket's Lipschitz
  /// constant: exact at leaves; at internal nodes the bucket is its
  /// pre-subdivision snapshot, which is a valid *upper* bound of `g` since
  /// insertions only ever lower the field. This is the sound replacement for
  /// the `4√2·d` heuristic rectangle, which no constant can make correct
  /// (see `solver::adf::tests::insertion_domain`).
  pub fn insert_at_maximum(
    &mut self,
    p: DistPoint<_Float, _Float, D>,
    prim: Primitive<_Float, D>
  ) -> bool {
    self.insert_within(p.point, p.distance, prim)
  }

  /// Like [`Self::insert_at_maximum`], with an explicit containment radius:
  /// the primitive must satisfy `S ⊆ B̄(center, radius)`. A caller placing a
  /// shape much smaller than the free ball (e.g. scaled to `d/4`) can pass its
  /// actual reach, shrinking the visited region `D*` accordingly.
  pub fn insert_within(
    &mut self,
    center: Point<_Float, D>,
    radius: _Float,
    prim: Primitive<_Float, D>
  ) -> bool {
    let two = _Float::one() + _Float::one();
    self.insert_where(move |node| {
      let dist = (center - node.rect.clamp_point(&center)).length();
      let half_diag = node.rect.size().length() / two;
      node.data.as_slice().sdf(node.rect.center())
        + bucket_lipschitz(&node.data) * half_diag > dist - radius
    }, prim)
  }

  fn insert_where(
    &mut self,
    keep: impl Fn(&Node<Vec<Primitive<_Float, D>>, _Float, D>) -> bool + Sync,
    prim: Primitive<_Float, D>
  ) -> bool {
    const BUCKET_SIZE: usize = 3;
    self.lipschitz_max = self.lipschitz_max.max(prim.lipschitz);
    // Copied out so the parallel `decide` closure captures plain values instead
    // of borrowing `self` (which `refine_leaves` already borrows via `tree`).
    let subdiv = self.prune_subdiv;
    let max_depth = self.tree.max_depth;

    // Only leaves admitted by `keep` are visited; each yields an independent
    // decision, evaluated in parallel and applied afterwards. Previously divided
    // nodes' fresh children are not revisited within a single call — same as the
    // old `Skip`-after-`subdivide` behaviour.
    self.tree.refine_leaves(keep, |node| {
      let f = &prim.f;
      let l_bucket = bucket_lipschitz(&node.data);

      // f(v) >= g(v) forall v e D — the new primitive never lowers the field here.
      if sdf_geq_everywhere(
        f.as_ref(), |p| node.data.as_slice().sdf(p),
        node.rect, prim.lipschitz, l_bucket, subdiv
      ) {
        return Refine::None;
      }

      // g(v) >= f(v) forall v e D — f dominates the whole node, replace it.
      if sdf_geq_everywhere(
        |p| node.data.as_slice().sdf(p), f.as_ref(),
        node.rect, l_bucket, prim.lipschitz, subdiv
      ) {
        return Refine::SetData(vec![prim.clone()]);
      }

      // remove SDF primitives that do not affect the field within `rect`
      let prune = |data: &[Primitive<_Float, D>], rect| {
        let mut g = vec![];
        for (i, p_i) in data.iter().enumerate() {
          let sdf_old = |p|
            data.iter().enumerate()
              .filter_map(|(j, p_j)| if i != j { Some((p_j.f)(p)) } else { None })
              .fold(_Float::max_value() / (_Float::one() + _Float::one()), |a, b| a.min(b));
          let l_old = data.iter().enumerate()
            .filter_map(|(j, p_j)| (i != j).then_some(p_j.lipschitz))
            .fold(_Float::one(), _Float::max);
          // keep `p_i` unless it is provably redundant (>= the rest) within `rect`
          if !sdf_geq_everywhere((p_i.f).as_ref(), sdf_old, rect, p_i.lipschitz, l_old, subdiv) {
            g.push(p_i.clone())
          }
        }
        g
      };

      if node.depth == max_depth || node.data.len() < BUCKET_SIZE {
        // Max depth reached (cannot subdivide) or the bucket still has room:
        // append. Re-pruning the whole bucket on every append would cost O(n^2)
        // per insert and the audit shows ~94% of a crowded bucket genuinely
        // contributes, so it is not worth it.
        let mut data = node.data.clone();
        data.push(prim.clone());
        Refine::SetData(data)
      } else {
        // Max bucket size reached: subdivide, pruning the combined set per child.
        let mut combined = node.data.clone();
        combined.push(prim.clone());
        Refine::Subdivide(<Dim<D> as Branching>::children_from_fn(
          |i| prune(combined.as_slice(), child_rect(node.rect, i))))
      }
    })
  }

  /// The insertion domain for a primitive placed at a **local maximum** `p` of
  /// the field — a sound replacement for [`crate::util::domain_empirical`].
  ///
  /// Any primitive `S ⊆ B̄(x₀, d)` (which all `offset = d − r` style placements
  /// satisfy) obeys `f(v) ≥ |v − x₀| − d`, so it can only lower the field inside
  ///
  /// ```text
  /// D* = { v : g(v) > |v − x₀| − d },
  /// ```
  ///
  /// and `D*` is tight: every `v ∈ D*` is updated by *some* admissible `S`.
  /// `D*` is not bounded by any multiple of `d` (an escape ray between contact
  /// points can extend it arbitrarily), so no constant-sized box is
  /// correct in general — this method instead covers `D*` by tree leaves,
  /// discarding a subtree `R` once
  ///
  /// ```text
  /// g(c_R) + L·h(R)  ≤  dist(R, x₀) − d      ⟹      D* ∩ R = ∅
  /// ```
  ///
  /// (`g(v) ≤ g(c_R) + L·h(R)` by `L`-Lipschitz continuity of the whole field),
  /// and returns the bounding box of the surviving leaves.
  pub fn update_domain(&self, p: DistPoint<_Float, _Float, D>) -> Aabb<_Float, D> {
    let (x0, d) = (p.point, p.distance);
    let two = _Float::one() + _Float::one();
    let l = self.lipschitz_max;

    let mut bounds: Option<Aabb<_Float, D>> = None;
    self.tree.visit_leaves(
      |node| {
        let half_diag = node.rect.size().length() / two;
        // distance from x0 to the box
        let dist = (x0 - node.rect.clamp_point(&x0)).length();
        self.sdf(node.rect.center()) + l * half_diag <= dist - d
      },
      |leaf| {
        bounds = Some(match bounds {
          Some(b) => b.union(&leaf.rect),
          None => leaf.rect,
        });
      },
    );
    // The leaf containing x0 always qualifies, so `bounds` is non-empty.
    bounds.unwrap()
  }

  /// # Safety
  /// Nobody is safe
  // This intentionally launders `&self` into `&mut Self` — inherently UB per the
  // `invalid_reference_casting` deny lint, which is the whole point of the escape
  // hatch. Explicitly opt out rather than dodge the lint.
  #[allow(invalid_reference_casting)]
  pub unsafe fn as_mut(&self) -> &mut Self {
    &mut *(self as *const Self).cast_mut()
  }
}

impl <_Float: Real, const D: usize> SDF<_Float, D> for ADF<_Float, D>
where
  Dim<D>: Branching,
{
  fn sdf(&self, pixel: Point<_Float, D>) -> _Float {
    match self.tree.pt_to_node(pixel) {
      Some(node) => node.data.as_slice().sdf(pixel),
      None => self.tree.root().data.as_slice().sdf(pixel),
    }}}

impl <_Float: Real, const D: usize> crate::geometry::BoundingBox<_Float, D> for ADF<_Float, D> {
  fn bounding_box(&self) -> Aabb<_Float, D> {
    Aabb::unit()
  }}

impl <_Float: Real, const D: usize> Debug for ADF<_Float, D>
where
  Dim<D>: Branching,
{
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    use humansize::{FileSize, file_size_opts as options};

    let mut max_depth = 0u8;
    let mut bucket_slots = 0usize;
    self.tree.traverse(&mut |node| {
      max_depth = max_depth.max(node.depth);
      bucket_slots += node.data.len();
      Ok(())
    }).ok();
    // Actual data only — Vec spare capacity is not counted (it can always be
    // shrunk to fit): the arena's nodes, plus each bucket's `Primitive`s (fat
    // pointer + Lipschitz constant per slot). The closures behind the `Arc`s
    // are shared between buckets and are not attributed here.
    let total_size = std::mem::size_of::<Self>()
      + self.tree.node_count()
        * std::mem::size_of::<Node<Vec<Primitive<_Float, D>>, _Float, D>>()
      + bucket_slots * std::mem::size_of::<Primitive<_Float, D>>();
    f.debug_struct("ADF")
      .field("total_nodes", &self.tree.node_count())
      .field("max_depth", &max_depth)
      .field("size", &total_size.file_size(options::BINARY).unwrap())
      .finish()
  }
}
