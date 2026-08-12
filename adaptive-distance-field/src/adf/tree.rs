//! Region tree over the unit hypercube `[0, 1]^DIMS`, backed by a flat arena,
//! in one of two **compile-time** subdivision layouts.
//!
//! | layout | children per node | levels per full split | ceiling on `DIMS` |
//! |---|---|---|---|
//! | [`Orthant`] | `2^DIMS` | 1 | 6 |
//! | [`Kd`] | 2 | `DIMS` | none |
//!
//! [`Orthant`] splits every axis at once — a quadtree at `DIMS = 2`, an octree
//! at `DIMS = 3` — and is the historical layout. [`Kd`] splits a single axis per
//! level, cycling `depth % DIMS`, so its branching factor is 2 in every
//! dimension. Both reach the same cell sizes; they differ in how many nodes and
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

/// Splits one axis per level, cycling `depth % DIMS`: 2 children per node,
/// `DIMS` levels per full split.
///
/// Reaches the same cell sizes as [`Orthant`] while allocating only the cells a
/// descent actually enters, and carries no ceiling on `DIMS`, since it needs no
/// `2^DIMS`-sized array.
#[derive(Clone, Copy, Debug, Default)]
pub struct Kd;

impl<const DIMS: usize> Layout<DIMS> for Orthant
where
  Dim<DIMS>: Branching,
{
  const CHILDREN: usize = <Dim<DIMS> as Branching>::CHILDREN;
  const LEVELS_PER_SPLIT: usize = 1;
  const NAME: &'static str = "orthant";
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

impl<const DIMS: usize> Layout<DIMS> for Kd {
  const CHILDREN: usize = 2;
  const LEVELS_PER_SPLIT: usize = DIMS;
  const NAME: &'static str = "k-d";
  type Children<T> = [T; 2];

  #[inline]
  fn children_from_fn<T>(mut f: impl FnMut(usize) -> T) -> [T; 2] {
    [f(0), f(1)]
  }

  /// Child 0 is the lower half along the cut axis, child 1 the upper.
  #[inline]
  fn child_rect<F: Real>(rect: Aabb<F, DIMS>, depth: u8, i: usize) -> Aabb<F, DIMS> {
    let a = cut_axis::<DIMS>(depth);
    let mid = rect.center()[a];
    let mut out = rect;
    if i == 0 { out.max[a] = mid } else { out.min[a] = mid }
    out
  }

  #[inline]
  fn child_index<F: Real>(rect: &Aabb<F, DIMS>, depth: u8, pt: &Point<F, DIMS>) -> usize {
    let a = cut_axis::<DIMS>(depth);
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

/// The action [`Tree::refine_leaves`] should apply to a visited leaf.
pub enum Refine<Data, const DIMS: usize, L>
where
  L: Layout<DIMS>,
{
  /// Leave the node unchanged.
  None,
  /// Replace the leaf's payload.
  SetData(Data),
  /// Split the leaf into `L::CHILDREN` children carrying the given payloads
  /// (in [`Layout::child_rect`] order).
  Subdivide(L::Children<Data>),
}

impl<Data, _Float: Real, const DIMS: usize, L> Tree<Data, _Float, DIMS, L>
where
  L: Layout<DIMS>,
{
  /// A tree with a single root node covering the unit hypercube. `max_depth` is
  /// in tree *levels*; see [`Layout::LEVELS_PER_SPLIT`].
  pub fn new(max_depth: u8, init: Data) -> Self {
    let root = Node {
      rect: Aabb::unit(),
      depth: 0,
      data: init,
      children: None,
    };
    Tree { nodes: vec![root], max_depth, layout: PhantomData }
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
  /// hypercube (including NaN coordinates). Descent is [`Layout::child_index`],
  /// which matches the half-open child cells exactly.
  pub fn pt_to_node(&self, pt: Point<_Float, DIMS>) -> Option<&Node<Data, _Float, DIMS>> {
    let mut node = &self.nodes[0];
    if !node.rect.contains(&pt) {
      return None;
    }
    while let Some(first) = node.children {
      let child = L::child_index(&node.rect, node.depth, &pt);
      node = &self.nodes[first.get() as usize + child];
    }
    Some(node)
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
    L::Children<Data>: Send,
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
        Refine::Subdivide(data) => {
          self.subdivide(i, data);
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
    L::Children<Data>: Send,
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

  /// Append `L::CHILDREN` children (with the given payloads) to the arena and
  /// link them under `parent`. Child cells match [`Layout::child_rect`] for the
  /// parent's depth.
  fn subdivide(&mut self, parent: usize, data: L::Children<Data>) {
    let (rect, depth) = (self.nodes[parent].rect, self.nodes[parent].depth);
    let rects = L::children_from_fn(|i| L::child_rect(rect, depth, i));
    // Children are pushed after the root, so `first` is always >= 1.
    let first = NonZeroU32::new(self.nodes.len() as u32).expect("root occupies index 0");
    for (rect, data) in rects.into_iter().zip(data) {
      self.nodes.push(Node { rect, depth: depth + 1, data, children: None });
    }
    self.nodes[parent].children = Some(first);
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
        Refine::Subdivide(std::array::from_fn(|i| i as u32))
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
          Refine::Subdivide([node.data * 10, node.data * 10 + 1])
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
