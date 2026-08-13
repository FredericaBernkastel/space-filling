//! Region tree over a box — the unit hypercube `[0, 1]^DIMS` by default, or any
//! [`Aabb`] through [`Tree::new_in`] — backed by a flat arena, in one of three
//! **compile-time** subdivision layouts.
//!
//! | layout | children per node | levels per full split | ceiling on `DIMS` |
//! |---|---|---|---|
//! | [`Orthant`] | `2^DIMS` | 1 | 6 |
//! | [`Kd`] | 2 | `DIMS` | none |
//! | [`WeightedKd`] | 2 | `DIMS` (an upper bound) | none |
//!
//! [`Orthant`] splits every axis at once — a quadtree at `DIMS = 2`, an octree
//! at `DIMS = 3` — and is the historical layout. [`Kd`] splits a single axis per
//! level, cycling `depth % DIMS`, so its branching factor is 2 in every
//! dimension. [`WeightedKd`] is [`Kd`] cutting the *widest* axis instead, which
//! on an anisotropic domain spends its cuts where the diameter is and leaves the
//! short axes alone; on a cube the two are the same tree. Both reach the same cell
//! sizes as [`Orthant`]; they differ in how many nodes and
//! how much per-node work it takes to get there, which is what makes the choice
//! matter as `DIMS` grows (see `tests/layout.rs`).
//!
//! Every node lives in a single [`Vec`]; a node references its children by the
//! arena index of the first one, and siblings are always stored contiguously.
//! Compared to a `Box`-linked tree this keeps siblings adjacent in memory
//! (cache-local traversal) and turns clone/drop of the whole tree into a single
//! allocation. It also removes the raw-pointer aliasing the previous
//! `Box`-per-subtree implementation relied on for parallel refinement.
//!
//! Dimension count and layout are both compile-time constants: branching factor,
//! child arrays and descent loops monomorphize per `(DIMS, L)`, so
//! `Tree<_, _, 2, Orthant>` compiles to the same code as the previous
//! quadtree-only implementation.

use {
  crate::geometry::{Aabb, Point, Real},
  anyhow::Result,
  nalgebra::Scalar,
  std::{marker::PhantomData, num::NonZeroU32},
};

/// Compile-time dimension marker; [`Branching`] ties it to its `2^DIMS`-way
/// child arrays.
pub struct Dim<const DIMS: usize>;

/// The `2^DIMS`-way branching of a [`Dim`]: supplies true `[T; 2^DIMS]` arrays
/// (stable Rust cannot express a generic-`DIMS`-dependent array length), so
/// child payloads live on the stack. Implemented for `DIMS = 1..=6`.
pub trait Branching {
  /// `2^DIMS`.
  const CHILDREN: usize;
  /// Exactly `[T; 2^DIMS]`.
  type Children<T>: IntoIterator<Item = T>;
  fn children_from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Children<T>;
}

macro_rules! impl_branching {($($dims:literal)*) => {$(
  impl Branching for Dim<$dims> {
    const CHILDREN: usize = 1 << $dims;
    type Children<T> = [T; 1 << $dims];
    #[inline]
    fn children_from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Children<T> {
      std::array::from_fn(f)
    }
  }
)*}}
impl_branching!(1 2 3 4 5 6);

/// `[T; 2^DIMS]`.
pub type Children<T, const DIMS: usize> = <Dim<DIMS> as Branching>::Children<T>;

/// How a node divides into children — the tree's subdivision strategy, chosen at
/// compile time.
///
/// Implementors are zero-sized markers: [`Orthant`] and [`Kd`]. A layout owns
/// three decisions, and nothing else in the tree depends on the geometry:
/// how many children a node has, where each child's cell lies, and which child
/// a query point descends into.
pub trait Layout<const DIMS: usize> {
  /// Children per subdivided node.
  const CHILDREN: usize;
  /// Levels equivalent to halving every axis once. `ADF` multiplies its depth
  /// budget by this, so a given budget means the same *resolution* in either
  /// layout rather than the same number of levels.
  const LEVELS_PER_SPLIT: usize;
  /// Short name, for diagnostics.
  const NAME: &'static str;
  /// Whether a descent should carry the cell down with it, halving a local copy,
  /// instead of reading each node's stored rect.
  ///
  /// Carrying costs one `child_rect` per level and saves a load of `2·DIMS`
  /// floats. That trade only pays when a layout takes many levels to get
  /// anywhere: measured, it is worth ~25% of query time to [`Kd`] at `DIMS = 3`
  /// and costs [`Orthant`] ~30%.
  const CARRY_CELL: bool;
  /// Exactly `[T; CHILDREN]`.
  type Children<T>: IntoIterator<Item = T>;
  fn children_from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Children<T>;
  /// The `i`-th sub-cell of a node with cell `rect` at `depth`. Child cells
  /// share their boundary coordinates bit-for-bit with the parent and each
  /// other (min/max representation), so the tiling is exact.
  fn child_rect<F: Real>(rect: Aabb<F, DIMS>, depth: u8, i: usize) -> Aabb<F, DIMS>;
  /// Which child of a node at `depth` contains `pt` — the descent step, and the
  /// exact inverse of [`Self::child_rect`] on the half-open cells.
  fn child_index<F: Real>(rect: &Aabb<F, DIMS>, depth: u8, pt: &Point<F, DIMS>) -> usize;
}

/// Splits every axis at once: `2^DIMS` children per node, one level per full
/// split. Limited to `DIMS <= 6` by [`Branching`], and to roughly `DIMS <= 10`
/// in practice by the fan-out itself — a thousand-way branch per subdivision.
#[derive(Clone, Copy, Debug, Default)]
pub struct Orthant;

/// Which axis a binary cut halves.
///
/// The certificate clears a cell when the centre margin covers
/// `(L_f + L_g)·h(R)`, and `h(R) = ½√(Σ sᵢ²)`, so halving side `sᵢ` buys a
/// reduction proportional to `sᵢ²`. Cutting the *widest* axis is therefore the
/// greedy move against the only quantity the proof cares about, and on a domain
/// whose extents encode weights it is the same rule as "cut in descending γ".
pub trait CutPolicy<const DIMS: usize> {
  const NAME: &'static str;
  /// The axis to halve, for the cell `rect` at `depth`. Takes the cell rather
  /// than its extent so that a policy ignoring geometry costs literally nothing:
  /// computing `rect.size()` for [`Cyclic`] to discard measured real time.
  fn axis<F: Real>(rect: &Aabb<F, DIMS>, depth: u8) -> usize;
}

/// Round-robin `depth % DIMS`, indifferent to extent.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cyclic;

/// Descending weight: the widest axis, ties to the lowest index.
///
/// On a cube this *is* [`Cyclic`] — sides start equal, so the argmax picks axis
/// 0, then 1 once axis 0 is short, and so on, returning to 0 exactly when every
/// axis has been halved once. The two produce identical trees there, which
/// `widest_reduces_to_cyclic_on_a_cube` pins.
///
/// On an anisotropic domain the tail axes are never reached: the proof succeeds
/// while they still contribute nothing to `h`, so levels-to-certify tracks the
/// number of *significant* axes rather than `DIMS`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Widest;

impl<const DIMS: usize> CutPolicy<DIMS> for Cyclic {
  const NAME: &'static str = "k-d";
  #[inline]
  fn axis<F: Real>(_rect: &Aabb<F, DIMS>, depth: u8) -> usize {
    cut_axis::<DIMS>(depth)
  }
}

impl<const DIMS: usize> CutPolicy<DIMS> for Widest {
  const NAME: &'static str = "k-d widest";
  #[inline]
  fn axis<F: Real>(rect: &Aabb<F, DIMS>, _depth: u8) -> usize {
    const { assert!(DIMS > 0, "Kd needs at least one axis to cut") }
    let (mut best, mut longest) = (0, rect.max[0] - rect.min[0]);
    // strictly greater, so equal extents resolve to the lowest index — the tie
    // rule that makes a cube reproduce `depth % DIMS`
    for a in 1..DIMS {
      let len = rect.max[a] - rect.min[a];
      if len > longest {
        best = a;
        longest = len;
      }
    }
    best
  }
}

/// A binary tree cutting one axis per level, choosing that axis by `P`.
///
/// Spell it [`Kd`] for round-robin cuts or [`WeightedKd`] for weight-ordered
/// ones; both are aliases of this.
#[derive(Clone, Copy, Debug, Default)]
pub struct KdBy<P>(PhantomData<P>);

/// [`KdBy`] cutting axes round-robin, `depth % DIMS`: 2 children per node,
/// `DIMS` levels per full split — the historical `Kd`.
///
/// Reaches the same cell sizes as [`Orthant`] while allocating only the cells a
/// descent actually enters, and carries no ceiling on `DIMS`, since it needs no
/// `2^DIMS`-sized array.
pub type Kd = KdBy<Cyclic>;

/// [`KdBy`] cutting the widest axis first, so cost follows the effective
/// dimension of an anisotropic domain rather than its ambient one.
pub type WeightedKd = KdBy<Widest>;

impl<const DIMS: usize> Layout<DIMS> for Orthant
where
  Dim<DIMS>: Branching,
{
  const CHILDREN: usize = <Dim<DIMS> as Branching>::CHILDREN;
  const LEVELS_PER_SPLIT: usize = 1;
  const NAME: &'static str = "orthant";
  const CARRY_CELL: bool = false;
  type Children<T> = <Dim<DIMS> as Branching>::Children<T>;

  #[inline]
  fn children_from_fn<T>(f: impl FnMut(usize) -> T) -> Self::Children<T> {
    <Dim<DIMS> as Branching>::children_from_fn(f)
  }

  /// Bit `a` of `i` selects the upper half along axis `a`. For `DIMS = 2` this
  /// is the quadrant order TL, TR, BL, BR.
  #[inline]
  fn child_rect<F: Real>(rect: Aabb<F, DIMS>, _depth: u8, i: usize) -> Aabb<F, DIMS> {
    child_rect(rect, i)
  }

  /// Per-axis comparison against the centre — `pt[a] >= center[a]` sets bit `a`.
  #[inline]
  fn child_index<F: Real>(rect: &Aabb<F, DIMS>, _depth: u8, pt: &Point<F, DIMS>) -> usize {
    let center = rect.center();
    let mut child = 0usize;
    for a in 0..DIMS {
      child |= ((pt[a] >= center[a]) as usize) << a;
    }
    child
  }
}

impl<P: CutPolicy<DIMS>, const DIMS: usize> Layout<DIMS> for KdBy<P> {
  const CHILDREN: usize = 2;
  /// Levels to halve every axis once. Under [`Widest`] on an anisotropic domain
  /// a round may spend several cuts on one long axis and never reach a short
  /// one, so this is a conservative *upper* bound there rather than an exact
  /// count — budgets derived from it stay at least as large as they are today.
  const LEVELS_PER_SPLIT: usize = DIMS;
  const NAME: &'static str = P::NAME;
  const CARRY_CELL: bool = true;
  type Children<T> = [T; 2];

  #[inline]
  fn children_from_fn<T>(mut f: impl FnMut(usize) -> T) -> [T; 2] {
    [f(0), f(1)]
  }

  /// Child 0 is the lower half along the cut axis, child 1 the upper.
  #[inline]
  fn child_rect<F: Real>(rect: Aabb<F, DIMS>, depth: u8, i: usize) -> Aabb<F, DIMS> {
    let a = P::axis(&rect, depth);
    let mid = rect.center()[a];
    let mut out = rect;
    if i == 0 { out.max[a] = mid } else { out.min[a] = mid }
    out
  }

  #[inline]
  fn child_index<F: Real>(rect: &Aabb<F, DIMS>, depth: u8, pt: &Point<F, DIMS>) -> usize {
    let a = P::axis(rect, depth);
    (pt[a] >= rect.center()[a]) as usize
  }
}

/// The axis a [`Kd`] node at `depth` cuts. Round-robin, so after `DIMS` levels
/// every axis has been halved exactly once.
///
/// `DIMS = 0` is rejected at compile time: unlike [`Orthant`], whose ceiling is
/// enforced by the missing [`Branching`] impl, [`Kd`] is implemented for every
/// `DIMS`, and a zero-dimensional tree would divide by zero here rather than fail
/// to typecheck.
#[inline]
pub fn cut_axis<const DIMS: usize>(depth: u8) -> usize {
  const { assert!(DIMS > 0, "Kd needs at least one axis to cut") }
  depth as usize % DIMS
}

/// The `i`-th sub-cell of `rect` under the [`Orthant`] tiling: bit `a` of `i`
/// selects the upper half along axis `a`.
///
/// Kept as a free function because the branch-and-bound of
/// [`sdf_geq_everywhere`](crate::adf::sdf_geq_everywhere) refines *boxes* rather
/// than tree nodes, and needs the same exact tiling without a node to ask.
#[inline]
pub fn child_rect<F: Real, const DIMS: usize>(
  rect: Aabb<F, DIMS>,
  i: usize,
) -> Aabb<F, DIMS> {
  let c = rect.center();
  let mut min = rect.min;
  let mut max = rect.max;
  for a in 0..DIMS {
    if i & (1 << a) != 0 { min[a] = c[a]; } else { max[a] = c[a]; }
  }
  Aabb { min, max }
}

/// All `2^DIMS` sub-cells of `rect`, in [`child_rect`] order.
#[inline]
pub fn child_rects<F: Real, const DIMS: usize>(
  rect: Aabb<F, DIMS>,
) -> Children<Aabb<F, DIMS>, DIMS>
where
  Dim<DIMS>: Branching,
{
  <Dim<DIMS> as Branching>::children_from_fn(|i| child_rect(rect, i))
}

/// A single tree node. `Data` is the user-defined payload.
///
/// `children` is the arena index of the first of `L::CHILDREN` contiguous
/// children, stored as `Option<NonZeroU32>`: children are always pushed after
/// the root (index 0), so a first-child index is never 0, which lets the niche
/// shrink the field to 4 bytes. For `Data = Vec<_>` and `DIMS = 2` this keeps
/// the whole node at 64 bytes — one cache line, and a power-of-two stride so
/// arena indexing is a shift. The node is layout-agnostic: only the number of
/// children behind `children` differs.
#[derive(Clone)]
pub struct Node<Data, Float: Scalar, const DIMS: usize> {
  pub rect: Aabb<Float, DIMS>,
  pub depth: u8,
  pub data: Data,
  /// Arena index of the first child, or `None` for a leaf.
  children: Option<NonZeroU32>,
}

impl<Data, Float: Scalar, const DIMS: usize> Node<Data, Float, DIMS> {
  #[inline]
  pub fn is_leaf(&self) -> bool {
    self.children.is_none()
  }
}

/// A region tree stored as a flat arena, in the layout `L`. `nodes[0]` is always
/// the root.
#[derive(Clone)]
pub struct Tree<Data, Float: Scalar, const DIMS: usize, L> {
  nodes: Vec<Node<Data, Float, DIMS>>,
  /// `nodes[i].children`, mirrored into a dense array.
  ///
  /// A descent needs nothing else — the cell can be halved on the way down
  /// instead of read back — and a node is `2·DIMS` floats wide, so walking the
  /// nodes touches one to three cache lines per level where this touches four
  /// bytes. At `DIMS = 6` it is the difference between striding a megabyte-scale
  /// arena and staying inside a few tens of kilobytes, which a layout needing
  /// `DIMS` levels per split feels `DIMS` times over.
  links: Vec<Option<NonZeroU32>>,
  /// Maximum depth in **levels** — not full splits. `ADF` converts.
  pub max_depth: u8,
  /// `fn() -> L` rather than `L`, so the tree's `Send`/`Sync`/`Copy` never
  /// depend on the marker.
  layout: PhantomData<fn() -> L>,
}

/// The `2^DIMS`-way tree — the historical name and the default layout.
pub type Quadtree<Data, Float, const DIMS: usize> = Tree<Data, Float, DIMS, Orthant>;

/// The binary, one-axis-per-level tree.
pub type KdTree<Data, Float, const DIMS: usize> = Tree<Data, Float, DIMS, Kd>;

/// One cell of a subtree being grafted onto a leaf: either it stays a leaf, or it
/// divides again. Lets a single [`Refine::Subdivide`] carry several levels, which
/// is what a layout needing `DIMS` cuts to halve every axis requires in order to
/// take its decisions on the same cell sizes as one that halves them all at once.
pub enum Split<Data, const DIMS: usize, L>
where
  L: Layout<DIMS>,
{
  /// This cell is final, and holds `Data`.
  Leaf(Data),
  /// This cell divides: it keeps `parent` for itself, and each of its children is
  /// another `Split`.
  Node {
    parent: Data,
    children: Box<L::Children<Split<Data, DIMS, L>>>,
  },
}

/// The action [`Tree::refine_leaves`] should apply to a visited leaf.
pub enum Refine<Data, const DIMS: usize, L>
where
  L: Layout<DIMS>,
{
  /// Leave the node unchanged.
  None,
  /// Replace the leaf's payload.
  SetData(Data),
  /// Divide the leaf, at least one level deep: it keeps `parent`, and each child
  /// (in [`Layout::child_rect`] order) is a [`Split`] that may divide further.
  Subdivide {
    parent: Data,
    children: Box<L::Children<Split<Data, DIMS, L>>>,
  },
}

impl<Data, _Float: Real, const DIMS: usize, L> Tree<Data, _Float, DIMS, L>
where
  L: Layout<DIMS>,
{
  /// A tree with a single root node covering the unit hypercube. `max_depth` is
  /// in tree *levels*; see [`Layout::LEVELS_PER_SPLIT`].
  pub fn new(max_depth: u8, init: Data) -> Self {
    Self::new_in(Aabb::unit(), max_depth, init)
  }

  /// A tree over an arbitrary root box. `max_depth` is in tree *levels*; see
  /// [`Layout::LEVELS_PER_SPLIT`].
  ///
  /// The domain's extents are the only place per-axis weights live: a root of
  /// extent `γ` makes axis `i` matter in proportion to `γᵢ`, which
  /// [`Widest`] reads to cut in descending weight. Seed the field with
  /// [`boundary_box`](crate::sdf::boundary_box) over the same box, or the walls
  /// will not agree with the domain.
  pub fn new_in(domain: Aabb<_Float, DIMS>, max_depth: u8, init: Data) -> Self {
    let root = Node {
      rect: domain,
      depth: 0,
      data: init,
      children: None,
    };
    Tree { nodes: vec![root], links: vec![None], max_depth, layout: PhantomData }
  }

  /// The root node.
  #[inline]
  pub fn root(&self) -> &Node<Data, _Float, DIMS> {
    &self.nodes[0]
  }

  /// Number of nodes in the arena (internal + leaves).
  #[inline]
  pub fn node_count(&self) -> usize {
    self.nodes.len()
  }

  /// Number of leaves — the cells the field is actually stored on.
  pub fn leaf_count(&self) -> usize {
    self.nodes.iter().filter(|n| n.is_leaf()).count()
  }

  /// Bytes held by the arena itself, excluding whatever the payloads own.
  #[inline]
  pub fn arena_bytes(&self) -> usize {
    self.nodes.len() * std::mem::size_of::<Node<Data, _Float, DIMS>>()
  }

  /// Apply `f` to every node (internal and leaf); order is unspecified. Stops
  /// early and returns the error if `f` fails.
  pub fn traverse(&self, f: &mut dyn FnMut(&Node<Data, _Float, DIMS>) -> Result<()>) -> Result<()> {
    for node in &self.nodes {
      f(node)?;
    }
    Ok(())
  }

  /// Depth-first visit of the leaves. `prune` is consulted for every node
  /// (internal and leaf); returning `true` skips that node's whole subtree.
  pub fn visit_leaves(
    &self,
    mut prune: impl FnMut(&Node<Data, _Float, DIMS>) -> bool,
    mut visit: impl FnMut(&Node<Data, _Float, DIMS>),
  ) {
    let mut stack = vec![0usize];
    while let Some(idx) = stack.pop() {
      let node = &self.nodes[idx];
      if prune(node) {
        continue;
      }
      match node.children {
        Some(first) => stack.extend(
          first.get() as usize..first.get() as usize + L::CHILDREN),
        None => visit(node),
      }
    }
  }

  /// The smallest node containing `pt`, or `None` if `pt` lies outside the root
  /// cell (including NaN coordinates). Descent is [`Layout::child_index`],
  /// which matches the half-open child cells exactly.
  pub fn pt_to_node(&self, pt: Point<_Float, DIMS>) -> Option<&Node<Data, _Float, DIMS>> {
    // Borrowed, not copied: the read-rect descent never needs the root's cell
    // again, and copying `2·DIMS` floats per query costs real time at DIMS = 6.
    if !self.root().rect.contains(&pt) {
      return None;
    }
    let mut idx = 0usize;
    if L::CARRY_CELL {
      // Every child cell follows from its parent's, so the walk halves a local
      // copy from the root's own cell and never loads a rect — touching four
      // bytes per level instead of a whole node. It must start from the root's
      // rect rather than the unit cube: `new_in` admits any domain, and starting
      // elsewhere silently descends into the wrong leaf.
      let mut cell = self.root().rect;
      let mut depth = 0u8;
      while let Some(first) = self.links[idx] {
        let child = L::child_index(&cell, depth, &pt);
        cell = L::child_rect(cell, depth, child);
        idx = first.get() as usize + child;
        depth += 1;
      }
      debug_assert!(self.nodes[idx].rect.min == cell.min && self.nodes[idx].rect.max == cell.max,
        "the descent's cell diverged from the node's own");
    } else {
      // Few enough levels that the arithmetic of carrying the cell outweighs the
      // load it saves; read each node's own rect instead.
      while let Some(first) = self.links[idx] {
        let node = &self.nodes[idx];
        idx = first.get() as usize + L::child_index(&node.rect, node.depth, &pt);
      }
    }
    Some(&self.nodes[idx])
  }

  /// Evaluate every leaf of the subtrees admitted by `keep` with `decide`
  /// (read-only), then apply the returned actions to the arena. A leaf that
  /// returns [`Refine::Subdivide`] is split; its fresh children are *not*
  /// revisited during the same call. Returns whether any node changed.
  ///
  /// `keep` is consulted for every node (internal and leaf); returning `false`
  /// skips that node's whole subtree — a child's cell is contained in its
  /// parent's, so a geometric (or field-bound) predicate prunes soundly.
  ///
  /// `decide` — the expensive per-leaf optimization — runs during a recursive
  /// descent that forks the children of each internal node onto the rayon pool,
  /// so independent subtrees are evaluated in parallel. Because `decide` is
  /// read-only this needs no aliasing tricks (shared `&Node` access is safe);
  /// the mutation is applied afterwards, sequentially, since growing the arena
  /// needs `&mut`.
  pub fn refine_leaves<K, F>(&mut self, keep: K, decide: F) -> bool
  where
    K: Fn(&Node<Data, _Float, DIMS>) -> bool + Sync,
    F: Fn(&Node<Data, _Float, DIMS>) -> Refine<Data, DIMS, L> + Send + Sync,
    Data: Send + Sync,
    // trivially satisfied — the GAT is always an array — but the compiler cannot
    // see through the projection
    L::Children<Split<Data, DIMS, L>>: Send,
    _Float: Sync,
  {
    let actions = self.collect_actions(0, &keep, &decide);

    let mut changed = false;
    for (i, action) in actions {
      match action {
        Refine::None => {}
        Refine::SetData(data) => {
          self.nodes[i].data = data;
          changed = true;
        }
        Refine::Subdivide { parent, children } => {
          self.nodes[i].data = parent;
          self.graft(i, *children);
          changed = true;
        }
      }
    }
    changed
  }

  /// Recursively evaluate the subtree rooted at `idx`, forking the children of
  /// each internal node across the rayon pool, and return the non-trivial
  /// `(index, action)` pairs. Read-only over the arena.
  fn collect_actions<K, F>(
    &self,
    idx: usize,
    keep: &K,
    decide: &F,
  ) -> Vec<(usize, Refine<Data, DIMS, L>)>
  where
    K: Fn(&Node<Data, _Float, DIMS>) -> bool + Sync,
    F: Fn(&Node<Data, _Float, DIMS>) -> Refine<Data, DIMS, L> + Sync,
    Data: Send + Sync,
    L::Children<Split<Data, DIMS, L>>: Send,
    _Float: Sync,
  {
    use rayon::prelude::*;

    let node = &self.nodes[idx];
    if !keep(node) {
      return Vec::new();
    }
    match node.children {
      None => match decide(node) {
        Refine::None => Vec::new(),
        action => vec![(idx, action)],
      },
      Some(first) => {
        let first = first.get() as usize;
        (first..first + L::CHILDREN).into_par_iter()
          .flat_map_iter(|child| self.collect_actions(child, keep, decide))
          .collect()
      }
    }
  }

  /// Append a node, keeping [`Self::links`] in step. The only place either grows.
  fn push(&mut self, node: Node<Data, _Float, DIMS>) {
    debug_assert!(node.children.is_none(), "a fresh node has no children yet");
    self.nodes.push(node);
    self.links.push(None);
  }

  /// Graft `children` under the node at `idx`, whose own payload is already in
  /// place. Child cells match [`Layout::child_rect`] for that node's depth.
  ///
  /// Siblings must stay contiguous, so a level is materialised in full before any
  /// of its own children are: the grandchildren of this call are appended only
  /// once every child is in the arena.
  fn graft(&mut self, idx: usize, children: L::Children<Split<Data, DIMS, L>>) {
    let (rect, depth) = (self.nodes[idx].rect, self.nodes[idx].depth);
    // Children are pushed after the root, so `first` is always >= 1.
    let first = NonZeroU32::new(self.nodes.len() as u32).expect("root occupies index 0");
    let mut deeper = Vec::new();
    for (i, child) in children.into_iter().enumerate() {
      let rect = L::child_rect(rect, depth, i);
      match child {
        Split::Leaf(data) =>
          self.push(Node { rect, depth: depth + 1, data, children: None }),
        Split::Node { parent, children } => {
          deeper.push((self.nodes.len(), children));
          self.push(Node { rect, depth: depth + 1, data: parent, children: None });
        }
      }
    }
    self.nodes[idx].children = Some(first);
    self.links[idx] = Some(first);
    for (at, children) in deeper {
      self.graft(at, *children);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // The 2D orthant instantiation must preserve the concrete layout: node = one
  // cache line, and children in TL, TR, BL, BR order.
  #[test] fn layout_2d() {
    use std::mem::size_of;
    assert_eq!(size_of::<Aabb<f64, 2>>(), 32);
    assert_eq!(size_of::<Node<Vec<u64>, f64, 2>>(), 64);
    // the layout marker is zero-sized: `Kd` costs the arena nothing
    assert_eq!(size_of::<Quadtree<Vec<u64>, f64, 2>>(),
               size_of::<KdTree<Vec<u64>, f64, 2>>());

    let quads = child_rects(Aabb::<f64, 2>::unit());
    assert_eq!(quads.map(|q| [q.min.x, q.min.y]),
      [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]);
  }

  #[test] fn octree_descent() {
    let mut tree = Quadtree::<u32, f64, 3>::new(2, 0);
    tree.refine_leaves(
      |_| true,
      |node| if node.depth == 0 {
        Refine::Subdivide {
          parent: 0,
          children: Box::new(std::array::from_fn(|i| Split::Leaf(i as u32))),
        }
      } else {
        Refine::None
      });
    assert_eq!(tree.node_count(), 9);

    // x >= cx sets bit 0, y < cy leaves bit 1 clear, z >= cz sets bit 2
    let node = tree.pt_to_node(Point::from([0.75, 0.25, 0.75])).unwrap();
    assert_eq!(node.data, 0b101);
    assert_eq!(node.rect.min, Point::from([0.5, 0.0, 0.5]));
    assert!(tree.pt_to_node(Point::from([0.5, 1.0, 0.5])).is_none()); // half-open root
  }

  // A k-d node splits one axis; after DIMS levels every axis has been halved
  // once, and the cells coincide with one orthant subdivision.
  #[test] fn kd_cuts_one_axis_per_level() {
    let mut tree = KdTree::<u32, f64, 3>::new(6, 0);
    // split the root (axis 0), then its lower child (axis 1), then that child's
    // lower child (axis 2) — a full round
    for step in 0..3u8 {
      tree.refine_leaves(
        |node| node.depth <= step,
        |node| if node.depth == step && node.rect.min == Point::origin() {
          Refine::Subdivide {
            parent: node.data,
            children: Box::new([
              Split::Leaf(node.data * 10),
              Split::Leaf(node.data * 10 + 1),
            ]),
          }
        } else {
          Refine::None
        });
    }
    // 1 root + 2 + 2 + 2 children
    assert_eq!(tree.node_count(), 7);

    // the thrice-cut lower corner is exactly one orthant cell
    let corner = tree.pt_to_node(Point::from([0.1, 0.1, 0.1])).unwrap();
    assert_eq!(corner.rect.min, Point::from([0.0, 0.0, 0.0]));
    assert_eq!(corner.rect.max, Point::from([0.5, 0.5, 0.5]));
    assert_eq!(corner.depth, 3);

    // and descent picks the cut axis per level: axis 0 first
    let upper = tree.pt_to_node(Point::from([0.9, 0.1, 0.1])).unwrap();
    assert_eq!(upper.rect.min, Point::from([0.5, 0.0, 0.0]));
    assert!(upper.is_leaf());
  }

  #[test] fn kd_needs_no_branching_impl() {
    // DIMS = 9 has no `Branching` impl, so `Orthant` cannot be instantiated
    // there at all; `Kd` has no such ceiling.
    let tree = KdTree::<u32, f64, 9>::new(9, 7);
    assert_eq!(tree.node_count(), 1);
    assert_eq!(<Kd as Layout<9>>::CHILDREN, 2);
    assert_eq!(<Kd as Layout<9>>::LEVELS_PER_SPLIT, 9);
    assert_eq!(tree.pt_to_node(Point::from([0.5; 9])).unwrap().data, 7);
  }
}
