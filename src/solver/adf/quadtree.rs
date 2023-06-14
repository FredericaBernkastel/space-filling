#![allow(dead_code)]
use {
  crate::{
    geometry::WorldSpace
  },
  std::{fmt::{Debug, Formatter}},
  anyhow::Result,
  euclid::{Point2D, Size2D, Rect},
  num_traits::Float
};

type Point<T> = Point2D<T, WorldSpace>;

#[derive(Clone)]
pub struct Quadtree<Data, Float> {
  pub rect: Rect<Float, WorldSpace>,
  pub children: Option<Box<[Quadtree<Data, Float>; 4]>>,
  pub depth: u8,
  pub max_depth: u8,
  pub data: Data
}

impl<Data: Debug, _Float: Float + Debug> Debug for Quadtree<Data, _Float> {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Quadtree")
      .field("rect", &self.rect)
      .field("children", &if self.children.is_some() { "Some(...)" } else { "None" })
      .field("depth", &self.depth)
      .field("data", &self.data)
      .finish()
  }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
/// 4 sections of a rectangle
pub enum Quadtrant {
  TL = 0,
  TR = 1,
  BL = 2,
  BR = 3
}

fn quadrant_origin<_Float: Float>() -> [Point<_Float>; 4] {
  let half = _Float::one() / (_Float::one() + _Float::one());
  [
    Point::new(_Float::zero(), _Float::zero()),
    Point::new(half, _Float::zero()),
    Point::new(_Float::zero(), half),
    Point::new(half, half)
  ]
}

impl Quadtrant {
  /// determine the section of a rectangle, containing `pt`
  pub fn get<_Float: Float>(rect: Rect<_Float, WorldSpace>, pt: Point<_Float>) -> Option<Self> {
    use Quadtrant::*;
    [TL, TR, BL, BR].iter()
      .find_map(|&quad| {
        let origin = rect.origin +
          quadrant_origin()[quad as usize].to_vector()
            .component_mul(rect.size.to_vector());
        Rect { origin, size: rect.size / (_Float::one() + _Float::one()) }
          .contains(pt)
          .then_some(quad)
      })
  }

  pub fn inv(self) -> Self {
    use Quadtrant::*;
    match self {
      TL => BR,
      BR => TL,
      TR => BL,
      BL => TR,
    }
  }

  pub fn mirror_x(self) -> Self {
    use Quadtrant::*;
    match self {
      TL => TR,
      TR => TL,
      BL => BR,
      BR => BL,
    }
  }

  pub fn mirror_y(self) -> Self {
    use Quadtrant::*;
    match self {
      TL => BL,
      TR => BR,
      BL => TL,
      BR => TR,
    }
  }
}

#[derive(PartialEq)]
pub enum TraverseCommand {
  Ok,
  Skip
}

impl<Data, _Float: Float> Quadtree<Data, _Float> {
  pub fn new(max_depth: u8, init: Data) -> Self {
    Quadtree {
      rect: Rect::from_size(Size2D::splat(_Float::one())),
      children: None,
      depth: 0,
      max_depth,
      data: init
    }
  }

  /// apply `f` to every node of the tree
  pub fn traverse(&self, f: &mut dyn FnMut(&Self) -> Result<()>) -> Result<()> {
    f(self)?;
    self.traverse_a(f)?;
    Ok(())
  }

  fn traverse_a(&self, f: &mut dyn FnMut(&Self) -> Result<()>) -> Result<()> {
    if let Some(children) = &self.children {
      for child in children.iter() {
        f(child)?;
      }
      for child in children.iter() {
        child.traverse_a(f)?;
      }
    }
    Ok(())
  }


  pub fn traverse_managed(&mut self, f: &mut impl FnMut(&mut Self) -> TraverseCommand) {
    if f(self) == TraverseCommand::Ok {
      self.traverse_managed_a(f);
    }
  }

  fn traverse_managed_a(&mut self, f: &mut impl FnMut(&mut Self) -> TraverseCommand) {
    if let Some(children) = &mut self.children {
      for child in children.iter_mut() {
        if f(child) == TraverseCommand::Ok {
          child.traverse_managed_a(f);
        }
      }
    }
  }

  pub fn traverse_managed_parallel(&mut self, f: impl Fn(&mut Self) -> TraverseCommand + Send + Sync) {
    if f(self) == TraverseCommand::Ok {
      self.traverse_managed_parallel_a(&f);
    }
  }

  fn traverse_managed_parallel_a(&mut self, f: &(impl Fn(&mut Self) -> TraverseCommand + Send + Sync)) {
    use rayon::prelude::*;

    if let Some(children) = self.children.as_deref_mut() {
      let mut children_ptr = [0; 4];
      for i in 0..4 {
        children_ptr[i] = &mut children[i] as *mut _ as usize;
      };

      children_ptr.into_par_iter()
        .for_each(move |child| {
          let child = unsafe { &mut *(child as *mut Self) };
          if f(child) == TraverseCommand::Ok {
            child.traverse_managed_parallel_a(f);
          }
        })
    }
  }

  pub fn subdivide(&mut self, f: impl Fn(Rect<_Float, WorldSpace>) -> Data) -> &mut Option<Box<[Quadtree<Data, _Float>; 4]>> {
    if self.depth < self.max_depth && self.children.is_none() {
      let rect = self.rect;
      let children: [Quadtree<Data, _Float>; 4] = [0, 1, 2, 3]
        .map(|i| {
          let rect = Rect {
            origin: rect.origin +
              quadrant_origin()[i as usize].to_vector()
                .component_mul(rect.size.to_vector()),
            size: rect.size / (_Float::one() + _Float::one())
          };
          Quadtree {
            rect,
            children: None,
            depth: self.depth + 1,
            max_depth: self.max_depth,
            data: f(rect)
          }
        });
      self.children = Some(Box::new(children));
    }
    &mut self.children
  }

  pub fn leaves_planar(&mut self) -> Vec<&mut Quadtree<Data, _Float>> {

    fn nodes_planar_a<Data, Float>(tree: &mut Quadtree<Data, Float>) -> Vec<*mut Quadtree<Data, Float>> {
      let mut result = vec![];
      if let Some(children) = tree.children.as_deref_mut() {
        for child in children.iter_mut() {
          result.append(&mut nodes_planar_a(child));
        }
      } else {
        result.push(tree)
      }
      result
    }

    nodes_planar_a(self)
      .into_iter()
      .map(|x| unsafe { x.as_mut().unwrap() })
      .collect()
  }

  /// return all nodes, containing `pt`
  pub fn path_to_pt(&self, pt: Point<_Float>) -> Vec<&Self> {
    let mut result = vec![self];
    if let Some(children) = self.children.as_deref() {
      if let Some(quad) = Quadtrant::get(self.rect, pt) {
        result.append(&mut children[quad as usize].path_to_pt(pt));
      }
    }
    result
  }

  /// find a smallest node containing pt
  pub fn pt_to_node(&self, pt: Point<_Float>) -> Option<&Self> {
    let mut node = self;
    while let Some(children) = node.children.as_deref() {
      node = &children[Quadtrant::get(node.rect, pt)? as usize]
    }
    Some(node)
  }
}

#[cfg(test)] mod tests {
  use super::*;

  impl<Data, _Float: Float> Quadtree<Data, _Float> {

    /// prints amount of total nodes in the tree, max subdivisions, and memory usage
    pub fn print_stats(&self) {
      use humansize::{FileSize, file_size_opts as options};

      let mut total_nodes = 0u64;
      let mut max_depth = 0u8;
      self.traverse(&mut |node| {
        total_nodes += 1;
        max_depth = (max_depth).max(node.depth);
        Ok(())
      }).ok();
      println!(
        "total nodes: {}\n\
      max subdivisions: {}\n\
      mem::size_of::<Quadtree<T>(): {}",
        total_nodes,
        max_depth,
        (std::mem::size_of::<Quadtree<Data, _Float>>() * total_nodes as usize)
          .file_size(options::BINARY).unwrap()
      );
    }
  }
}