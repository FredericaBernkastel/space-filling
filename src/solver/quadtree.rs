#![allow(dead_code)]
use {
  crate::{
    error::Result,
    geometry::WorldSpace
  },
  std::{fmt::{Debug, Formatter}},
  euclid::{Point2D, Size2D, Rect}
};

type Point<T> = Point2D<T, WorldSpace>;

pub struct Quadtree<T> {
  pub rect: Rect<f64, WorldSpace>,
  pub children: Option<Box<[Quadtree<T>; 4]>>,
  pub depth: u8,
  pub max_depth: u8,
  pub data: T
}

impl<T: Debug> Debug for Quadtree<T> {
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

const QUADRANT_ORIGIN: [Point<f64>; 4] = [
  Point::new(0.0, 0.0),
  Point::new(0.5, 0.0),
  Point::new(0.0, 0.5),
  Point::new(0.5, 0.5)
];

impl Quadtrant {
  /// determine the section of a rectangle, containing `pt`
  pub fn get(rect: Rect<f64, WorldSpace>, pt: Point<f64>) -> Option<Self> {
    use Quadtrant::*;
    [TL, TR, BL, BR].iter()
      .find_map(|&quad| {
        let origin = rect.origin +
          QUADRANT_ORIGIN[quad as usize].to_vector()
            .component_mul(rect.size.to_vector());
        Rect { origin, size: rect.size / 2.0 }
          .contains(pt)
          .then(|| quad)
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

impl<T> Quadtree<T> {
  pub fn new(max_depth: u8, init: T) -> Self {
    Quadtree {
      rect: Rect::from_size(Size2D::splat(1.0)),
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

  pub fn traverse_managed_parallel(&mut self, f: &mut (dyn FnMut(&mut Self) -> TraverseCommand)) {
    if f(self) == TraverseCommand::Ok {
      self.traverse_managed_parallel_a(f);
    }
  }

  fn traverse_managed_parallel_a(&mut self, f: &mut (dyn FnMut(&mut Self) -> TraverseCommand)) {
    use rayon::prelude::*;

    if let Some(children) = self.children.as_deref() {
      let mut children_ptr = [0; 4];
      for i in 0..4 {
        children_ptr[i] = &children[i] as *const _ as usize;
      };

      let f = &f as *const _ as usize;

      children_ptr.into_par_iter()
        .for_each(move |child| {
          let child = unsafe { &mut *(child as *mut Self) };
          let f = unsafe {
            &mut **(f as *const *mut (dyn FnMut(&mut Self) -> TraverseCommand))
          };
          if f(child) == TraverseCommand::Ok {
            child.traverse_managed_parallel_a(f);
          }
        })
    }
  }

  pub fn subdivide(&mut self, f: impl Fn(Rect<f64, WorldSpace>) -> T) -> &mut Option<Box<[Quadtree<T>; 4]>> {
    if self.depth < self.max_depth && self.children.is_none() {
      let rect = self.rect;
      let children: [Quadtree<T>; 4] = [0, 1, 2, 3]
        .map(|i| {
          let rect = Rect {
            origin: rect.origin +
              QUADRANT_ORIGIN[i as usize].to_vector()
                .component_mul(rect.size.to_vector()),
            size: rect.size / 2.0
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

  pub fn leaves_planar(&mut self) -> Vec<&mut Quadtree<T>> {

    fn nodes_planar_a<T>(tree: &mut Quadtree<T>) -> Vec<*mut Quadtree<T>> {
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
  pub fn path_to_pt(&self, pt: Point<f64>) -> Vec<&Self> {
    let mut result = vec![self];
    if let Some(children) = self.children.as_deref() {
      if let Some(quad) = Quadtrant::get(self.rect, pt) {
        result.append(&mut children[quad as usize].path_to_pt(pt));
      }
    }
    result
  }

  /// find a smallest node containing pt
  pub fn pt_to_node(&self, pt: Point<f64>) -> Option<&Self> {
    let mut node = self;
    while let Some(children) = node.children.as_deref() {
      node = &children[Quadtrant::get(node.rect, pt)? as usize]
    }
    Some(node)
  }
}

#[cfg(test)] mod tests {
  use super::*;

  impl<T> Quadtree<T> {

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
        (std::mem::size_of::<Quadtree<T>>() * total_nodes as usize)
          .file_size(options::BINARY).unwrap()
      );
    }
  }
}