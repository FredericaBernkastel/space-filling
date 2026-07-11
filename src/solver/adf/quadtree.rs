//! Region 2^N-tree over the unit hypercube `[0, 1]^DIMS`, backed by a flat
//! arena. The type keeps the name `Quadtree` across dimensions — `DIMS = 2` is
//! a quadtree, `DIMS = 3` an octree, and so on.
//!
//! Every node lives in a single [`Vec`]; a node references its `2^DIMS`
//! children by the arena index of the first child, and siblings are always
//! stored contiguously. Compared to a `Box`-linked tree this keeps siblings
//! adjacent in memory (cache-local traversal) and turns clone/drop of the whole
//! tree into a single allocation. It also removes the raw-pointer aliasing the
//! previous `Box`-per-subtree implementation relied on for parallel refinement.
//!
//! Dimension count is a compile-time constant: the branching factor, child
//! arrays and descent loops all monomorphize per `DIMS` ([`Branching`] supplies
//! the `[T; 2^DIMS]` arrays that stable Rust cannot yet express directly), so
//! the 2D instantiation compiles to the same code as the previous
//! quadtree-only implementation.

use {
  crate::geometry::{Aabb, Point, Real},
  anyhow::Result,
  nalgebra::Scalar,
  std::num::NonZeroU32,
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

/// The `i`-th sub-cell of `rect`: bit `a` of `i` selects the upper half along
/// axis `a`. For `DIMS = 2` this is the quadrant order TL, TR, BL, BR. Child
/// cells share their boundary coordinates bit-for-bit with the parent and each
/// other (min/max representation), so the tiling is exact.
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
/// `children` is the arena index of the first of `2^DIMS` contiguous children,
/// stored as `Option<NonZeroU32>`: children are always pushed after the root
/// (index 0), so a first-child index is never 0, which lets the niche shrink
/// the field to 4 bytes. For `Data = Vec<_>` and `DIMS = 2` this keeps the
/// whole node at 64 bytes — one cache line, and a power-of-two stride so arena
/// indexing is a shift.
#[derive(Clone)]
pub struct Node<Data, Float: Scalar, const DIMS: usize> {
  pub rect: Aabb<Float, DIMS>,
  pub depth: u8,
  pub data: Data,
  /// Arena index of the first child (in [`child_rect`] order), or `None` for
  /// a leaf.
  children: Option<NonZeroU32>,
}

impl<Data, Float: Scalar, const DIMS: usize> Node<Data, Float, DIMS> {
  #[inline]
  pub fn is_leaf(&self) -> bool {
    self.children.is_none()
  }
}

/// A 2^N-tree stored as a flat arena. `nodes[0]` is always the root.
#[derive(Clone)]
pub struct Quadtree<Data, Float: Scalar, const DIMS: usize> {
  nodes: Vec<Node<Data, Float, DIMS>>,
  pub max_depth: u8,
}

/// The action [`Quadtree::refine_leaves`] should apply to a visited leaf.
pub enum Refine<Data, const DIMS: usize>
where
  Dim<DIMS>: Branching,
{
  /// Leave the node unchanged.
  None,
  /// Replace the leaf's payload.
  SetData(Data),
  /// Split the leaf into `2^DIMS` children carrying the given payloads
  /// (in [`child_rect`] order).
  Subdivide(Children<Data, DIMS>),
}

impl<Data, _Float: Real, const DIMS: usize> Quadtree<Data, _Float, DIMS>
where
  Dim<DIMS>: Branching,
{
  /// A tree with a single root node covering the unit hypercube.
  pub fn new(max_depth: u8, init: Data) -> Self {
    let root = Node {
      rect: Aabb::unit(),
      depth: 0,
      data: init,
      children: None,
    };
    Quadtree { nodes: vec![root], max_depth }
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
          first.get() as usize..first.get() as usize + <Dim<DIMS> as Branching>::CHILDREN),
        None => visit(node),
      }
    }
  }

  /// The smallest node containing `pt`, or `None` if `pt` lies outside the
  /// root hypercube (including NaN coordinates). Descent picks the child by
  /// per-axis comparison against the node's centre — `pt[a] >= center[a]`
  /// sets bit `a` — matching the half-open child cells exactly.
  pub fn pt_to_node(&self, pt: Point<_Float, DIMS>) -> Option<&Node<Data, _Float, DIMS>> {
    let mut node = &self.nodes[0];
    if !node.rect.contains(&pt) {
      return None;
    }
    while let Some(first) = node.children {
      let center = node.rect.center();
      let mut child = 0usize;
      for a in 0..DIMS {
        child |= ((pt[a] >= center[a]) as usize) << a;
      }
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
  /// descent that forks the `2^DIMS` children of each internal node onto the
  /// rayon pool, so independent subtrees are evaluated in parallel. Because
  /// `decide` is read-only this needs no aliasing tricks (shared `&Node`
  /// access is safe); the mutation is applied afterwards, sequentially, since
  /// growing the arena needs `&mut`.
  pub fn refine_leaves<K, F>(&mut self, keep: K, decide: F) -> bool
  where
    K: Fn(&Node<Data, _Float, DIMS>) -> bool + Sync,
    F: Fn(&Node<Data, _Float, DIMS>) -> Refine<Data, DIMS> + Send + Sync,
    Data: Send + Sync,
    // trivially satisfied — the GAT is always `[Data; 2^DIMS]`, but the
    // compiler cannot see through the projection
    Children<Data, DIMS>: Send,
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

  /// Recursively evaluate the subtree rooted at `idx`, forking the `2^DIMS`
  /// children of each internal node across the rayon pool, and return the
  /// non-trivial `(index, action)` pairs. Read-only over the arena.
  fn collect_actions<K, F>(
    &self,
    idx: usize,
    keep: &K,
    decide: &F,
  ) -> Vec<(usize, Refine<Data, DIMS>)>
  where
    K: Fn(&Node<Data, _Float, DIMS>) -> bool + Sync,
    F: Fn(&Node<Data, _Float, DIMS>) -> Refine<Data, DIMS> + Sync,
    Data: Send + Sync,
    Children<Data, DIMS>: Send,
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
        (first..first + <Dim<DIMS> as Branching>::CHILDREN).into_par_iter()
          .flat_map_iter(|child| self.collect_actions(child, keep, decide))
          .collect()
      }
    }
  }

  /// Append `2^DIMS` children (with the given payloads) to the arena and link
  /// them under `parent`. Children cells match [`child_rects`].
  fn subdivide(&mut self, parent: usize, data: Children<Data, DIMS>) {
    let rects = child_rects(self.nodes[parent].rect);
    let child_depth = self.nodes[parent].depth + 1;
    // Children are pushed after the root, so `first` is always >= 1.
    let first = NonZeroU32::new(self.nodes.len() as u32).expect("root occupies index 0");
    for (rect, data) in rects.into_iter().zip(data) {
      self.nodes.push(Node { rect, depth: child_depth, data, children: None });
    }
    self.nodes[parent].children = Some(first);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // The 2D instantiation must preserve the concrete layout: node = one cache
  // line, and children in TL, TR, BL, BR order.
  #[test] fn layout_2d() {
    use std::mem::size_of;
    assert_eq!(size_of::<Aabb<f64, 2>>(), 32);
    assert_eq!(size_of::<Node<Vec<u64>, f64, 2>>(), 64);

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
}
