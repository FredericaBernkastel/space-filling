//! Region quadtree over the unit square `[0, 1]²`, backed by a flat arena.
//!
//! Every node lives in a single [`Vec`]; a node references its four children by
//! the arena index of the first child, and the four siblings are always stored
//! contiguously. Compared to a `Box`-linked tree this keeps siblings adjacent in
//! memory (cache-local traversal) and turns clone/drop of the whole tree into a
//! single allocation. It also removes the raw-pointer aliasing the previous
//! `Box`-per-subtree implementation relied on for parallel refinement.

use {
  crate::geometry::WorldSpace,
  anyhow::Result,
  euclid::{Point2D, Size2D, Rect},
  num_traits::Float,
  std::num::NonZeroU32,
};

type Point<T> = Point2D<T, WorldSpace>;

/// A single quadtree node. `Data` is the user-defined payload.
///
/// `children` is the arena index of the first of four contiguous children, and
/// is stored as `Option<NonZeroU32>`: children are always pushed after the root
/// (index 0), so a first-child index is never 0, which lets the niche shrink the
/// field to 4 bytes. For `Data = Vec<_>` this keeps the whole node at 64 bytes —
/// one cache line, and a power-of-two stride so arena indexing is a shift.
#[derive(Clone)]
pub struct Node<Data, Float> {
  pub rect: Rect<Float, WorldSpace>,
  pub depth: u8,
  pub data: Data,
  /// Arena index of the first child (in [`Quadtrant`] order: TL, TR, BL, BR),
  /// or `None` for a leaf.
  children: Option<NonZeroU32>,
}

impl<Data, Float> Node<Data, Float> {
  #[inline]
  pub fn is_leaf(&self) -> bool {
    self.children.is_none()
  }
}

/// A quadtree stored as a flat arena. `nodes[0]` is always the root.
#[derive(Clone)]
pub struct Quadtree<Data, Float> {
  nodes: Vec<Node<Data, Float>>,
  pub max_depth: u8,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
/// The four sub-sections of a rectangle.
pub enum Quadtrant {
  TL = 0,
  TR = 1,
  BL = 2,
  BR = 3,
}

/// Normalized top-left corners of the four quadrants, in [`Quadtrant`] order.
fn quadrant_origin<F: Float>() -> [Point<F>; 4] {
  let half = F::one() / (F::one() + F::one());
  [
    Point::new(F::zero(), F::zero()),
    Point::new(half, F::zero()),
    Point::new(F::zero(), half),
    Point::new(half, half),
  ]
}

impl Quadtrant {
  /// The quadrant of `rect` containing `pt`, if any.
  pub fn get<F: Float>(rect: Rect<F, WorldSpace>, pt: Point<F>) -> Option<Self> {
    use Quadtrant::*;
    [TL, TR, BL, BR].into_iter().find(|&quad| {
      let origin = rect.origin
        + quadrant_origin()[quad as usize].to_vector().component_mul(rect.size.to_vector());
      Rect { origin, size: rect.size / (F::one() + F::one()) }.contains(pt)
    })
  }
}

/// The four sub-rectangles of `rect`, in [`Quadtrant`] order.
pub fn child_rects<F: Float>(rect: Rect<F, WorldSpace>) -> [Rect<F, WorldSpace>; 4] {
  let two = F::one() + F::one();
  quadrant_origin::<F>().map(|origin| Rect {
    origin: rect.origin + origin.to_vector().component_mul(rect.size.to_vector()),
    size: rect.size / two,
  })
}

/// The action [`Quadtree::refine_leaves`] should apply to a visited leaf.
pub enum Refine<Data> {
  /// Leave the node unchanged.
  None,
  /// Replace the leaf's payload.
  SetData(Data),
  /// Split the leaf into four children carrying the given payloads
  /// (in [`Quadtrant`] order: TL, TR, BL, BR).
  Subdivide([Data; 4]),
}

impl<Data, _Float: Float> Quadtree<Data, _Float> {
  /// A tree with a single root node covering the unit square.
  pub fn new(max_depth: u8, init: Data) -> Self {
    let root = Node {
      rect: Rect::from_size(Size2D::splat(_Float::one())),
      depth: 0,
      data: init,
      children: None,
    };
    Quadtree { nodes: vec![root], max_depth }
  }

  /// The root node.
  #[inline]
  pub fn root(&self) -> &Node<Data, _Float> {
    &self.nodes[0]
  }

  /// Apply `f` to every node (internal and leaf); order is unspecified. Stops
  /// early and returns the error if `f` fails.
  pub fn traverse(&self, f: &mut dyn FnMut(&Node<Data, _Float>) -> Result<()>) -> Result<()> {
    for node in &self.nodes {
      f(node)?;
    }
    Ok(())
  }

  /// The smallest node containing `pt`, or `None` if `pt` falls outside an
  /// internal node's rectangle.
  pub fn pt_to_node(&self, pt: Point<_Float>) -> Option<&Node<Data, _Float>> {
    let mut node = &self.nodes[0];
    while let Some(first) = node.children {
      let quad = Quadtrant::get(node.rect, pt)? as usize;
      node = &self.nodes[first.get() as usize + quad];
    }
    Some(node)
  }

  /// Evaluate every leaf whose rectangle intersects `domain` with `decide`
  /// (read-only), then apply the returned actions to the arena. A leaf that
  /// returns [`Refine::Subdivide`] is split; its fresh children are *not*
  /// revisited during the same call. Returns whether any node changed.
  ///
  /// `decide` — the expensive per-leaf optimization — runs during a recursive
  /// descent that forks the four children of each internal node onto the rayon
  /// pool, so independent subtrees are evaluated in parallel. Because `decide`
  /// is read-only this needs no aliasing tricks (shared `&Node` access is safe);
  /// the mutation is applied afterwards, sequentially, since growing the arena
  /// needs `&mut`. Subtrees whose rectangle misses `domain` are pruned wholesale
  /// — a child's rectangle is contained in its parent's — so the descent stays
  /// proportional to the nodes near `domain`.
  pub fn refine_leaves<F>(&mut self, domain: Rect<_Float, WorldSpace>, decide: F) -> bool
  where
    F: Fn(&Node<Data, _Float>) -> Refine<Data> + Send + Sync,
    Data: Send + Sync,
    _Float: Sync,
  {
    let actions = self.collect_actions(0, &domain, &decide);

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

  /// Recursively evaluate the subtree rooted at `idx`, forking the four children
  /// of each internal node across the rayon pool, and return the non-trivial
  /// `(index, action)` pairs. Read-only over the arena.
  fn collect_actions<F>(
    &self,
    idx: usize,
    domain: &Rect<_Float, WorldSpace>,
    decide: &F,
  ) -> Vec<(usize, Refine<Data>)>
  where
    F: Fn(&Node<Data, _Float>) -> Refine<Data> + Sync,
    Data: Send + Sync,
    _Float: Sync,
  {
    use rayon::prelude::*;

    let node = &self.nodes[idx];
    if !node.rect.intersects(domain) {
      return Vec::new();
    }
    match node.children {
      None => match decide(node) {
        Refine::None => Vec::new(),
        action => vec![(idx, action)],
      },
      Some(first) => {
        let first = first.get() as usize;
        (first..first + 4).into_par_iter()
          .flat_map_iter(|child| self.collect_actions(child, domain, decide))
          .collect()
      }
    }
  }

  /// Append four children (with the given payloads) to the arena and link them
  /// under `parent`. Children rectangles match [`child_rects`].
  fn subdivide(&mut self, parent: usize, data: [Data; 4]) {
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
