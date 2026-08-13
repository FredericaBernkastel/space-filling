//! Adaptively Sampled Distance Field, backed by a [`tree`] arena, in any
//! compile-time dimension count and either subdivision layout.
//!
//! Each node (bucket) stores a handful of [`Primitive`]s — a field closure
//! together with its declared Lipschitz bound — and represents their pointwise
//! `min`. `ADF` itself implements [`SDF`], so a field composed
//! of millions of primitives is sampled in logarithmic time by descending to
//! the leaf covering the query point, rather than evaluated at quadratic cost.
//!
//! The backing tree's layout is a type parameter with **no default**, because it
//! is a real decision rather than a detail: `ADF<f64, 3, Orthant>` splits all
//! three axes at once, `ADF<f64, 6, Kd>` splits one axis per level. State it
//! through the turbofish, or one slot at a time with [`builder()`]:
//!
//! ```
//! # use adaptive_distance_field::adf;
//! let field = adf::builder().f64().dims::<3>().orthant().bounded(6);
//! ```
//!
//! The depth budget counts *full* subdivisions either way, so the two layouts
//! represent the same field at the same resolution and differ only in cost.

#![allow(clippy::mut_from_ref)]
use {
  crate::{
    geometry::{Point, Aabb, Real, Vector, VectorExt, DistPoint},
    sdf::{SDF, Lipschitz},
  },
  tree::{Tree, Node, Refine, Split, Dim, Branching, child_rects},
  std::{
    sync::Arc,
    fmt::{Debug, Formatter}
  },
  nalgebra::Scalar,
};

#[cfg(test)] mod tests;
pub mod builder;
pub mod manifold;
pub mod tree;
pub use builder::{builder, AdfBuilder, Dims, Unset};
pub use manifold::Manifold;
pub use tree::{CutPolicy, Cyclic, Kd, KdBy, Layout, Orthant, WeightedKd, Widest};
/// The tree module's previous name, so existing paths keep resolving.
pub use tree as quadtree;

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

/// What a node of the tree holds.
///
/// A leaf holds the primitives themselves, and answers queries with their `min`.
/// A node that has subdivided holds a single number instead: the only thing its
/// frozen bucket was ever consulted for afterwards, namely an upper bound of the
/// field over its cell. That bound stays valid however much is inserted later,
/// because insertions only ever *lower* the field — the same monotonicity the
/// insertion walk of [`ADF::insert_at_maximum`] already rests on.
///
/// Keeping the primitives there instead would cost `bucket × 24` bytes per
/// internal node for nothing: descent stops at leaves, and only leaves are ever
/// re-pruned or re-queried.
#[derive(Clone)]
pub enum Bucket<Float: Scalar, const D: usize> {
  /// A leaf's primitives; the field here is their pointwise `min`.
  Leaf(Vec<Primitive<Float, D>>),
  /// `ĝ(c_R) + L_B · h(R)`, frozen when this node subdivided.
  Bound(Float),
}

impl<_Float: Real, const D: usize> Bucket<_Float, D> {
  /// The primitives stored here — empty once the node has subdivided.
  #[inline]
  pub fn primitives(&self) -> &[Primitive<_Float, D>] {
    match self {
      Bucket::Leaf(v) => v,
      Bucket::Bound(_) => &[],
    }
  }

  /// Occupied primitive slots, for accounting.
  #[inline]
  pub fn len(&self) -> usize {
    self.primitives().len()
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// An upper bound of the field over `rect`: `ĝ(c) + L·h`, by Lipschitz
  /// continuity of `min` over the bucket. Read back verbatim once frozen.
  #[inline]
  pub fn upper_bound(&self, rect: &Aabb<_Float, D>) -> _Float {
    match self {
      Bucket::Leaf(v) => Self::bound_of(v, rect),
      Bucket::Bound(u) => *u,
    }
  }

  /// The bound a set of primitives certifies over `rect` — what [`Bucket::Bound`]
  /// caches at subdivision.
  fn bound_of(bucket: &[Primitive<_Float, D>], rect: &Aabb<_Float, D>) -> _Float {
    let two = _Float::one() + _Float::one();
    bucket.sdf(rect.center()) + bucket_lipschitz(bucket) * (rect.size().length() / two)
  }
}

#[derive(Clone)]
pub struct ADF<Float: Scalar, const D: usize, L> {
  pub tree: Tree<Bucket<Float, D>, Float, D, L>,
  /// Tree **levels** the redundancy test ([`sdf_geq_everywhere_levels`]) may use
  /// to prove/refute `f >= g` over a node. Higher = finer proofs (a primitive is
  /// pruned only when provably redundant to within ~`node/2^n`), at more work in
  /// near-tangent regions.
  ///
  /// Stored in levels rather than subdivisions because the two stop being
  /// interchangeable in high dimension: a binary layout spends `D` levels per
  /// subdivision, so the mildest useful setting of
  /// [`Self::with_prune_subdiv`] is a hundred levels at `D = 100`, and an
  /// undecided proof then walks a binary tree a hundred deep.
  prune_levels: u32,
  /// Largest Lipschitz constant ever declared by an inserted primitive; the
  /// whole field is `lipschitz_max`-Lipschitz (monotone over-approximation:
  /// primitives pruned later do not lower it).
  lipschitz_max: Float,
  /// Levels a single overflow may divide through — see
  /// [`Self::with_split_round`]. One by default.
  split_round: u8,
  /// Primitives a leaf may hold before it divides — see
  /// [`Self::with_bucket_size`].
  bucket_size: usize,
  /// Whether an overflowing leaf must see the cut prune something before it
  /// divides — see [`Self::with_cut_must_prune`]. Defaults to
  /// `D >= CUT_MUST_PRUNE_MIN_DIMS`.
  cut_must_prune: bool,
}

/// What an insertion can reach: the caller's guarantee that the primitive's
/// support lies inside this set.
///
/// The insertion walk skips a subtree once the field there cannot fall below the
/// distance to this set, so a tighter set is a shorter walk. A ball is the right
/// answer for a body placed at a maximum of the free space; a box is the right
/// answer for anything with per-axis radii, whose containing ball is `√D` times
/// too generous.
#[derive(Clone, Copy, Debug)]
pub enum Reach<Float: Scalar, const D: usize> {
  /// `S ⊆ B̄(centre, radius)`.
  Ball { centre: Point<Float, D>, radius: Float },
  /// `S ⊆` this box.
  Box(Aabb<Float, D>),
}

impl<_Float: Real, const D: usize> Reach<_Float, D> {
  /// Distance from `rect` to this set, negative or zero where they meet.
  ///
  /// Exact for both variants, and for two axis-aligned boxes it is the norm of
  /// the per-axis gaps — no square roots of a max, no bounding-ball slack.
  #[inline]
  pub fn distance_to(&self, rect: &Aabb<_Float, D>) -> _Float {
    match self {
      Reach::Ball { centre, radius } =>
        (*centre - rect.clamp_point(centre)).length() - *radius,
      Reach::Box(b) => {
        let mut sq = _Float::zero();
        for a in 0..D {
          // zero on any axis whose intervals overlap
          let gap = (b.min[a] - rect.max[a])
            .max(rect.min[a] - b.max[a])
            .max(_Float::zero());
          sq = sq + gap * gap;
        }
        sq.sqrt()
      }
    }
  }

  /// The box this reaches, for a caller that needs one — a ball's bounding box.
  pub fn bounds(&self) -> Aabb<_Float, D> {
    match self {
      Reach::Ball { centre, radius } => Aabb {
        min: Point::from(centre.coords.map(|x| x - *radius)),
        max: Point::from(centre.coords.map(|x| x + *radius)),
      },
      Reach::Box(b) => *b,
    }
  }
}

/// The dimension from which an overflowing leaf divides only if dividing prunes
/// something — see [`ADF::with_cut_must_prune`].
///
/// Four, where the covering overhead of a box against the ball its certificate
/// covers, `(√D/2)^D = (D/4)^(D/2)`, passes 1: 0.65 at `D = 3`, 1 at 4, 3.4 at 6,
/// 9.8e6 at 20. Measured crossover is the same integer — ×0.10 the circles placed
/// at `D = 3`, ×3.9 to ×75 from 4 up.
///
/// `D = 3` loses to a local-greedy trap: the first cut of the domain prunes nothing
/// whatever, so refusing it forecloses the depth at which cuts *would* bite. Above
/// the crossover there is no such depth.
///
/// Never fires under [`Orthant`], whose all-axes cuts do prune; the constant
/// guards a [`Kd`] trap.
pub const CUT_MUST_PRUNE_MIN_DIMS: usize = 4;

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

/// Default primitives a bucket may hold before the node divides;
/// [`ADF::with_bucket_size`] overrides it.
const BUCKET_SIZE: usize = 3;

/// Drop every primitive that provably does not affect the field within `rect`.
///
/// Errs toward keeping: a primitive survives unless `sdf_geq_everywhere_in`
/// *proves* it is dominated by the rest over the whole cell, so the stored field
/// never deviates from the `min` over everything inserted.
fn prune_bucket<_Float, const D: usize, L>(
  data: &[Primitive<_Float, D>],
  rect: Aabb<_Float, D>,
  levels: u32,
) -> Vec<Primitive<_Float, D>>
where
  _Float: Real,
  L: Layout<D>,
{
  let mut kept = vec![];
  for (i, p_i) in data.iter().enumerate() {
    let sdf_old = |p|
      data.iter().enumerate()
        .filter_map(|(j, p_j)| if i != j { Some((p_j.f)(p)) } else { None })
        .fold(_Float::max_value() / (_Float::one() + _Float::one()), |a, b| a.min(b));
    let l_old = data.iter().enumerate()
      .filter_map(|(j, p_j)| (i != j).then_some(p_j.lipschitz))
      .fold(_Float::one(), _Float::max);
    if !sdf_geq_everywhere_levels::<L, _, _, _, D>(
      (p_i.f).as_ref(), sdf_old, rect, p_i.lipschitz, l_old, levels) {
      kept.push(p_i.clone())
    }
  }
  kept
}

/// Divide `rect` down to `round_end`, pruning `bucket` against each cell on the
/// way, and return the subtree that results.
///
/// A branch stops as soon as its pruned bucket fits, so the round is adaptive
/// rather than a fixed `2^D` fan-out: under [`Orthant`], where
/// `LEVELS_PER_SPLIT` is 1, `round_end` is the caller's own depth plus one and
/// this produces exactly one level — the historical behaviour, unchanged.
///
/// Also returns the smallest bucket any cell of the round ended up with — free,
/// every `kept` being in hand — which [`ADF::with_cut_must_prune`] compares against
/// the bucket the round started from.
fn divide_round<_Float, const D: usize, L>(
  bucket: &[Primitive<_Float, D>],
  rect: Aabb<_Float, D>,
  depth: u16,
  round_end: u16,
  levels: u32,
  capacity: usize,
) -> (L::Children<Split<Bucket<_Float, D>, D, L>>, usize)
where
  _Float: Real,
  L: Layout<D>,
{
  let mut min_kept = usize::MAX;
  let children = L::children_from_fn(|i| {
    let cell = L::child_rect(&rect, depth, i);
    let kept = prune_bucket::<_Float, D, L>(bucket, cell, levels);
    min_kept = min_kept.min(kept.len());
    let child_depth = depth + 1;
    if child_depth >= round_end || kept.len() < capacity {
      Split::Leaf(Bucket::Leaf(kept))
    } else {
      let (grandchildren, deeper) = divide_round::<_Float, D, L>(
        &kept, cell, child_depth, round_end, levels, capacity);
      min_kept = min_kept.min(deeper);
      Split::Node {
        parent: Bucket::Bound(Bucket::bound_of(&kept, &cell)),
        children: Box::new(grandchildren),
      }
    }
  });
  (children, min_kept)
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
pub fn sdf_geq_everywhere<_Float, F, G, const D: usize>(
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

/// [`sdf_geq_everywhere`], refining boxes in the layout `L` rather than always by
/// orthants.
///
/// Identical claim, identical soundness — the inequality never mentions how a box
/// was cut. What changes is the shape of the search: [`Orthant`] pushes `2^D`
/// sub-boxes per level, [`Kd`] pushes 2 and needs `D` levels to reach
/// the same size, so `max_subdiv` is multiplied by
/// [`Layout::LEVELS_PER_SPLIT`] to keep the *resolution* of the proof fixed
/// across layouts. In high dimension that trades a `2^D`-way fan-out for a deeper
/// walk over the sub-boxes a witness actually lives in.
// `L` leads the parameter list so a caller can name just the layout —
// `sdf_geq_everywhere_in::<Kd>(..)` — and leave the rest to inference.
pub fn sdf_geq_everywhere_in<L, _Float, F, G, const D: usize>(
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
  L: Layout<D>,
{
  sdf_geq_everywhere_levels::<L, _, _, _, D>(
    f, g, domain, l_f, l_g, max_subdiv * L::LEVELS_PER_SPLIT as u32)
}

/// [`sdf_geq_everywhere_in`] with the budget given in **levels** rather than in
/// full subdivisions.
///
/// Which is the honest unit for a binary layout: `LEVELS_PER_SPLIT` is `D`, so
/// one "subdivision" of budget is 100 levels at `D = 100`, and an undecided box
/// then drives a binary branch-and-bound a hundred deep — `2^100` boxes in the
/// worst case, and in practice a hang. Callers that want a *cheap* answer rather
/// than a thorough one should say so here.
///
/// At zero levels this is exactly the inscribed-ball test: the root either
/// clears on `(f-g)(c) ≥ (L_f + L_g)·h`, or fails.
pub fn sdf_geq_everywhere_levels<L, _Float, F, G, const D: usize>(
  f: F,
  g: G,
  domain: Aabb<_Float, D>,
  l_f: _Float,
  l_g: _Float,
  levels: u32,
) -> bool
where
  _Float: Real,
  F: Fn(Point<_Float, D>) -> _Float,
  G: Fn(Point<_Float, D>) -> _Float,
  L: Layout<D>,
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
    if depth >= levels {
      return false; // undecided within budget → conservatively assume a witness
    }
    // `depth` doubles as the cut-axis counter for `Kd`, and is ignored by
    // `Orthant` — the same call serves both.
    for sub in L::children_from_fn(|i| L::child_rect(&rect, depth as u16, i)) {
      stack.push((sub, depth + 1));
    }
  }
  true
}

impl <_Float: Real + Send + Sync, const D: usize, L> ADF<_Float, D, L>
where
  L: Layout<D>,
  // trivially satisfied — the GAT is always an array — but the compiler
  // cannot see through the projection
  L::Children<Split<Bucket<_Float, D>, D, L>>: Send,
{
  /// Create an ADF in the layout `L`: `ADF::<f64, 6, Kd>::new(3, init)`, or
  /// `adf::builder().f64().dims::<6>().kd().build(3, init)` to name the
  /// parameters one at a time.
  ///
  /// `max_depth` specifies the maximum number of **full** subdivisions —
  /// halvings of every axis — so it means the same resolution in either layout;
  /// `init` specifies initial sdf primitives. The arena stores the budget in
  /// levels, which for [`Kd`] is `max_depth * D`.
  pub fn new(max_depth: u8, init: Vec<Primitive<_Float, D>>) -> Self {
    Self::new_in(Aabb::unit(), max_depth, init)
  }

  /// A field over an arbitrary domain box rather than the unit cube.
  ///
  /// The domain's extents are where per-axis weights live: a box of extent `γ`
  /// makes axis `i` matter in proportion to `γᵢ`, and
  /// [`Widest`] cuts in descending `γ` so that refinement cost
  /// follows the *effective* dimension rather than `D`. On a cube every axis
  /// weighs the same and `Widest` is exactly [`Cyclic`].
  ///
  /// `init` must bound the same box — [`sdf::boundary_box`](crate::sdf::boundary_box)
  /// over `domain`, not [`boundary_rect`](crate::sdf::boundary_rect), unless the
  /// domain *is* the unit cube.
  pub fn new_in(
    domain: Aabb<_Float, D>,
    max_depth: u8,
    init: Vec<Primitive<_Float, D>>,
  ) -> Self {
    let lipschitz_max = bucket_lipschitz(&init);
    let levels = (max_depth as usize * L::LEVELS_PER_SPLIT).min(u16::MAX as usize) as u16;
    Self {
      tree: Tree::new_in(domain, levels, Bucket::Leaf(init)),
      prune_levels: 8 * L::LEVELS_PER_SPLIT as u32,
      lipschitz_max,
      split_round: 1,
      bucket_size: BUCKET_SIZE,
      cut_must_prune: D >= CUT_MUST_PRUNE_MIN_DIMS,
    }
  }

  /// Bytes held by the field: the arena's nodes plus each bucket's occupied
  /// `Primitive` slots.
  ///
  /// `Vec` spare capacity is excluded (it can always be shrunk to fit), and so
  /// are the closures behind the `Arc`s, which are shared between buckets and
  /// cannot be attributed to one.
  pub fn memory_bytes(&self) -> usize {
    let mut slots = 0usize;
    self.tree.traverse(&mut |node| { slots += node.data.len(); Ok(()) }).ok();
    std::mem::size_of::<Self>()
      + self.tree.arena_bytes()
      + slots * std::mem::size_of::<Primitive<_Float, D>>()
  }

  /// The layout's short name — `"orthant"` or `"k-d"`.
  pub fn layout_name(&self) -> &'static str {
    L::NAME
  }
  /// Controls precision of primitive pruning in a bucket: the redundancy test may
  /// refine a node up to `subdiv` times to prove `f >= g` (see [`sdf_geq_everywhere`]).
  pub fn with_prune_subdiv(mut self, subdiv: u32) -> Self {
    self.prune_levels = subdiv * L::LEVELS_PER_SPLIT as u32;
    self
  }

  /// The same budget in tree **levels**, which above `D ≈ 12` is the only usable
  /// unit: one subdivision is `D` levels for a binary layout, so
  /// [`Self::with_prune_subdiv`] cannot express anything cheaper than `D`.
  /// Single digits are normal here.
  pub fn with_prune_levels(mut self, levels: u32) -> Self {
    self.prune_levels = levels;
    self
  }

  /// Primitives a leaf may hold before it divides. Three by default.
  ///
  /// A pure performance knob: the field is bit-identical at every capacity, since
  /// pruning and the redundancy proof never consult it. Larger buckets mean fewer,
  /// fatter leaves — a shallower tree and less memory, paid for by more primitives
  /// to evaluate per query, so there is an optimum rather than a direction.
  ///
  /// It also interacts with the proof: `sdf_geq_everywhere` resolves margins at
  /// scale `(L_f + L_g)·h(node)`, so a bucket of steepness `L` needs roughly `L`
  /// times deeper branch-and-bound per redundancy proof, while dividing the node
  /// halves `h` once and for all. Fields of mixed steepness therefore want
  /// capacity scaled as `⌊β / max(L_bucket, L_prim)⌋` rather than a constant;
  /// measured optima were 5 for 1-Lipschitz balls and 1 for a Mandelbrot
  /// estimator at `L = 4`.
  pub fn with_bucket_size(mut self, size: usize) -> Self {
    self.bucket_size = size.max(1);
    self
  }

  /// Whether an overflowing leaf may only divide if dividing prunes something.
  /// Defaults to `D >= `[`CUT_MUST_PRUNE_MIN_DIMS`].
  ///
  /// The trial division is computed either way, so the test is free. What it
  /// refuses is a division leaving every cell with the parent's entire bucket —
  /// which in high dimension is every division, a ball straddling a cut surviving
  /// on both sides, so the tree doubles per insertion and stores nothing new. One
  /// fat leaf answers queries with the same primitives the deep tree would have.
  ///
  /// A refused leaf retries once its bucket has doubled: `O(log n)` trials rather
  /// than an `O(n^2)` one per insertion.
  ///
  /// A performance knob, not a correctness one — pruning only ever drops primitives
  /// it has *proved* redundant, so the field is bit-identical either way. At
  /// `D = 6` over 150 balls, off against on: build ×5.0 slower, query ×2.96 faster,
  /// memory ×620 larger, so query-heavy work in `4..10` may want it off.
  pub fn with_cut_must_prune(mut self, require: bool) -> Self {
    self.cut_must_prune = require;
    self
  }

  /// Set the depth budget in tree **levels**, replacing the full-subdivision
  /// count [`Self::new`] derived.
  ///
  /// "Halve every axis once" stops being a useful unit in high dimension: a k-d
  /// tree spends `D` levels on it, so at `D = 100` a single subdivision is 100
  /// levels and two is already most of what a `u16` will hold. Worse, it is the
  /// wrong unit — on a weighted domain the tail axes are deliberately never cut,
  /// so a "full" subdivision is not something the tree ever wants to complete.
  /// Budget the levels directly instead:
  ///
  /// ```
  /// # use adaptive_distance_field::adf::{self, ADF, Kd, Primitive};
  /// # use adaptive_distance_field::sdf;
  /// let field = ADF::<f64, 100, Kd>::new(1, vec![Primitive::new(sdf::boundary_rect)])
  ///   .with_levels(24);   // 24 cuts, not 100
  /// assert_eq!(field.tree.max_depth, 24);
  /// ```
  pub fn with_levels(mut self, levels: u16) -> Self {
    self.tree.max_depth = levels;
    self
  }

  /// How many levels one overflowing leaf may divide through in a single
  /// insertion. One — a single level — by default.
  ///
  /// Setting it to [`Layout::LEVELS_PER_SPLIT`] makes an overflow complete a full
  /// round of axis cuts, so that a layout halving one axis per level takes its
  /// pruning decisions on the same cell sizes as one halving them all at once.
  /// That sounds like it should help `Kd` and, measured, it does not: it enlarges
  /// the tree by 3–10% for no gain in leaf occupancy or query time (see
  /// CHANGELOG.md). Kept because the negative result is worth being able to
  /// reproduce, and because the premise may hold for other primitive mixes.
  pub fn with_split_round(mut self, levels: u8) -> Self {
    self.split_round = levels.max(1);
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
    self.insert_within_reach(Reach::Ball { centre: center, radius }, prim)
  }

  /// Certify that the field is non-negative everywhere in `rect` — that the box
  /// is free space.
  ///
  /// [`sdf_geq_everywhere_in`] with the constant zero on the right: sound one
  /// way only, so `true` means proved and `false` means *undecided within the
  /// budget*, never "occupied".
  /// `levels` is a budget in tree levels, not subdivisions — see
  /// [`sdf_geq_everywhere_levels`]. Small is right here: at zero this is the
  /// inscribed-ball test, each level buys one halving of `h`, and an undecided
  /// box is not worth chasing when the caller can simply try a smaller one.
  pub fn box_is_free(&self, rect: Aabb<_Float, D>, levels: u32) -> bool {
    sdf_geq_everywhere_levels::<L, _, _, _, D>(
      |p| self.sdf(p),
      |_| _Float::zero(),
      rect,
      self.lipschitz_max,
      _Float::zero(),
      levels,
    )
  }

  /// The largest box at `p` of the given aspect that the field certifies free.
  ///
  /// A ball is the wrong body once the domain is not a cube: it is limited by
  /// the *thinnest* free direction and wastes every other one, which at
  /// `D = 100` is essentially all of them. This grows a box of fixed shape
  /// instead, so a body placed on a weighted manifold inherits the manifold's
  /// anisotropy — pass [`Manifold::aspect`](manifold::Manifold::aspect).
  ///
  /// `aspect` is normalised internally, so `t·aspect` has circumradius `t` and
  /// the search can start at `t = g(p)` with no proof at all: that box is inside
  /// the free ball. From there it doubles while the certificate holds, then
  /// bisects. `steps` bounds each half, so the cost is `2·steps`
  /// certifications and never a search over `D` axes separately.
  ///
  /// Sound by construction: every returned box has been proved free, or is the
  /// unproved-but-inscribed starting box. Returns an empty box at `p` when the
  /// field there is not positive.
  pub fn grow_box(
    &self,
    p: Point<_Float, D>,
    aspect: Vector<_Float, D>,
    steps: u32,
    levels: u32,
  ) -> Aabb<_Float, D> {
    let two = _Float::one() + _Float::one();
    let clearance = self.sdf(p);
    let norm = aspect.length();
    if !(clearance > _Float::zero()) || !(norm > _Float::zero()) {
      return Aabb { min: p, max: p };
    }
    let unit = aspect / norm;
    let at = |t: _Float| Aabb {
      min: Point::from(p.coords - unit * t),
      max: Point::from(p.coords + unit * t),
    };

    // `lo` is always free — the inscribed box to begin with, circumradius
    // `clearance` and so inside the free ball — and `hi` never proved so.
    let mut lo = clearance;
    let mut hi = clearance * two;
    for _ in 0..steps {
      if self.box_is_free(at(hi), levels) { lo = hi; hi = hi * two } else { break }
    }
    // bisect the gap even when the first doubling was refused, or everything
    // between the inscribed box and twice it goes unexamined
    for _ in 0..steps {
      let mid = (lo + hi) / two;
      if self.box_is_free(at(mid), levels) { lo = mid } else { hi = mid }
    }
    at(lo)
  }

  /// [`Self::insert_within`] with the containment set given as a [`Reach`]
  /// rather than always as a ball.
  ///
  /// The `D*` argument never needed a ball. For any `S ⊆ B`, every `v` satisfies
  /// `f(v) ≥ dist(v, B)`, so a subtree `R` cannot change and is skipped once
  /// `ĝ(c_R) + L_B·h(R) ≤ dist(R, B)` — the same inequality [`Reach::Ball`]
  /// spells out with `dist(R, x₀) − d`.
  ///
  /// Which matters for anisotropic bodies. A box of half-extents `ρ` is
  /// contained in a ball of radius `‖ρ‖`, its **circumradius**, so describing it
  /// as a ball inflates the visited region by the shape's anisotropy
  /// `κ = R_S/r_S` — `√D` for a box, hence `D^(D/2)` in volume. Passing
  /// [`Reach::Box`] keeps the walk as tight as the body actually is.
  pub fn insert_within_reach(
    &mut self,
    reach: Reach<_Float, D>,
    prim: Primitive<_Float, D>
  ) -> bool {
    self.insert_where(move |node| {
      // One scalar at an internal node, one bucket evaluation at a leaf — see
      // [`Bucket`].
      node.data.upper_bound(&node.rect) > reach.distance_to(&node.rect)
    }, prim)
  }

  fn insert_where(
    &mut self,
    keep: impl Fn(&Node<Bucket<_Float, D>, _Float, D>) -> bool + Sync,
    prim: Primitive<_Float, D>
  ) -> bool {
    self.lipschitz_max = self.lipschitz_max.max(prim.lipschitz);
    // Copied out so the parallel `decide` closure captures plain values instead
    // of borrowing `self` (which `refine_leaves` already borrows via `tree`).
    let levels = self.prune_levels;
    let max_depth = self.tree.max_depth;
    let split_round = self.split_round;
    let capacity = self.bucket_size;
    let cut_must_prune = self.cut_must_prune;

    // Only leaves admitted by `keep` are visited; each yields an independent
    // decision, evaluated in parallel and applied afterwards. Previously divided
    // nodes' fresh children are not revisited within a single call — same as the
    // old `Skip`-after-`subdivide` behaviour.
    self.tree.refine_leaves(keep, |node| {
      let f = &prim.f;
      let bucket = node.data.primitives();
      let l_bucket = bucket_lipschitz(bucket);

      // f(v) >= g(v) forall v e D — the new primitive never lowers the field here.
      if sdf_geq_everywhere_levels::<L, _, _, _, D>(
        f.as_ref(), |p| bucket.sdf(p),
        node.rect, prim.lipschitz, l_bucket, levels
      ) {
        return Refine::None;
      }

      // g(v) >= f(v) forall v e D — f dominates the whole node, replace it.
      if sdf_geq_everywhere_levels::<L, _, _, _, D>(
        |p| bucket.sdf(p), f.as_ref(),
        node.rect, l_bucket, prim.lipschitz, levels
      ) {
        return Refine::SetData(Bucket::Leaf(vec![prim.clone()]));
      }

      // A refused leaf sits over capacity, so retry only once its bucket has
      // doubled: `O(log n)` trials, with the bucket's length as the clock.
      let retry_now = !cut_must_prune
        || bucket.len() == capacity
        || bucket.len().is_power_of_two();

      if node.depth == max_depth || bucket.len() < capacity || !retry_now {
        // Max depth reached (cannot subdivide) or the bucket still has room:
        // append. Re-pruning the whole bucket on every append would cost O(n^2)
        // per insert and the audit shows ~94% of a crowded bucket genuinely
        // contributes, so it is not worth it.
        let mut data = bucket.to_vec();
        data.push(prim.clone());
        Refine::SetData(Bucket::Leaf(data))
      } else {
        // Max bucket size reached: divide, and keep dividing until every axis has
        // been halved once — `L::LEVELS_PER_SPLIT` levels. A layout that halves
        // one axis per level would otherwise take its pruning decisions on
        // half-cells, where far less is provably redundant than in the cells an
        // all-axes-at-once layout reaches immediately; completing the round puts
        // both on the same cell size. Adaptive: a branch stops early the moment
        // pruning empties its bucket below the threshold.
        let mut combined = bucket.to_vec();
        combined.push(prim.clone());
        let round_end = (node.depth as usize + split_round as usize)
          .min(max_depth as usize) as u16;

        let (children, min_kept) = divide_round::<_Float, D, L>(
          &combined, node.rect, node.depth, round_end, levels, capacity);

        // Below `CUT_MUST_PRUNE_MIN_DIMS` this refusal is a local-greedy trap: the
        // first cut of the domain prunes nothing, so the tree never reaches the
        // depth at which cuts do bite. At and above it there is no such depth —
        // every cell keeps the whole bucket, so a division buys two copies of it.
        if cut_must_prune && min_kept >= combined.len() {
          return Refine::SetData(Bucket::Leaf(combined));
        }

        Refine::Subdivide {
          // The root is the one node that keeps its primitives after dividing:
          // `sdf` falls back to it for points outside the domain, where there is
          // no leaf to descend to, and a bound cannot answer a query. One node's
          // worth of buckets, and the seed field stays readable from outside.
          parent: if node.depth == 0 {
            Bucket::Leaf(combined.clone())
          } else {
            Bucket::Bound(Bucket::bound_of(&combined, &node.rect))
          },
          children: Box::new(children),
        }
      }
    })
  }

  /// The insertion domain for a primitive placed at a **local maximum** `p` of
  /// the field — the sound replacement for the historical `4·√2·d` rectangle.
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
          Some(b) => b.merge(&leaf.rect),
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

impl <_Float: Real, const D: usize, L> SDF<_Float, D> for ADF<_Float, D, L>
where
  L: Layout<D>,
{
  fn sdf(&self, pixel: Point<_Float, D>) -> _Float {
    match self.tree.pt_to_node(pixel) {
      Some(node) => node.data.primitives().sdf(pixel),
      None => self.tree.root().data.primitives().sdf(pixel),
    }}}

impl <_Float: Real, const D: usize, L> crate::geometry::BoundingBox<_Float, D> for ADF<_Float, D, L>
where
  L: Layout<D>,
{
  /// The domain the field was built over — the root's cell, not necessarily the
  /// unit cube, since [`ADF::new_in`] admits any box.
  fn bounding_box(&self) -> Aabb<_Float, D> {
    self.tree.root().rect
  }}

impl <_Float: Real, const D: usize, L> Debug for ADF<_Float, D, L>
where
  L: Layout<D>,
{
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    use humansize::{FileSize, file_size_opts as options};

    let mut max_depth = 0u16;
    let mut bucket_slots = 0usize;
    self.tree.traverse(&mut |node| {
      max_depth = max_depth.max(node.depth);
      bucket_slots += node.data.len();
      Ok(())
    }).ok();
    // See `memory_bytes` for what is and is not attributed.
    let total_size = std::mem::size_of::<Self>()
      + self.tree.arena_bytes()
      + bucket_slots * std::mem::size_of::<Primitive<_Float, D>>();
    f.debug_struct("ADF")
      .field("layout", &L::NAME)
      .field("total_nodes", &self.tree.node_count())
      .field("leaves", &self.tree.leaf_count())
      // in tree levels: `L::LEVELS_PER_SPLIT` of them make one full subdivision
      .field("max_depth", &max_depth)
      .field("size", &total_size.file_size(options::BINARY).unwrap())
      .finish()
  }
}
