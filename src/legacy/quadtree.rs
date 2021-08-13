use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use crate::{
  *,
  error::Result,
  geometry::{Point, Rect},
};

/// Example usage:
/// ```
/// let tree = Quadtree::new(512.0, Point { x: 256.0, y: 256.0 }, 9);
/// tree.insert_sdf(&|sample|
///   sdf::circle(sample, sdf::Circle {
///     xy: Point { x: 256.0, y: 256.0 },
///     r: 128.0
/// }));
/// ```
pub struct Quadtree<T> {
  pub rect: Rect<f32>,
  pub children: Option<Box<[Quadtree<T>; 4]>>,
  pub depth: u8,
  pub max_depth: u8,
  pub is_inside: bool,
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

const CENTER_MAT: [[f32; 2]; 4] = [
  [-1.0, -1.0],
  [ 1.0, -1.0],
  [-1.0,  1.0],
  [ 1.0,  1.0]
];

impl Quadtrant {
  /// determine the section of a rectangle, containing `pt`
  pub fn get(rect: Rect<f32>, pt: Point<f32>) -> Option<Self> {
    let center: [Point<f32>; 4] = (0..4).into_iter()
      .map(|i| Point {
        x: rect.size * CENTER_MAT[i][0] / 4.0 + rect.center.x,
        y: rect.size * CENTER_MAT[i][1] / 4.0 + rect.center.y,
      })
      .collect::<Vec<_>>()
      .try_into()
      .unwrap();
    let size = rect.size / 2.0;
    match pt {
      z if z.in_rect(Rect { center: center[Quadtrant::TL as usize], size }.into()) => Some(Self::TL),
      z if z.in_rect(Rect { center: center[Quadtrant::TR as usize], size }.into()) => Some(Self::TR),
      z if z.in_rect(Rect { center: center[Quadtrant::BL as usize], size }.into()) => Some(Self::BL),
      z if z.in_rect(Rect { center: center[Quadtrant::BR as usize], size }.into()) => Some(Self::BR),
      _ => None
    }
  }

  pub fn inv(self: Self) -> Self {
    use Quadtrant::*;
    match self {
      TL => BR,
      BR => TL,
      TR => BL,
      BL => TR,
    }
  }

  pub fn mirror_x(self: Self) -> Self {
    use Quadtrant::*;
    match self {
      TL => TR,
      TR => TL,
      BL => BR,
      BR => BL,
    }
  }

  pub fn mirror_y(self: Self) -> Self {
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
  /// apply `f` to every node of the tree
  pub fn traverse(&self, f: &mut impl FnMut(&Self) -> Result<()>) -> Result<()> {
    f(self)?;
    self.traverse_a(f)?;
    Ok(())
  }

  #[doc(hidden)]
  fn traverse_a(&self, f: &mut impl FnMut(&Self) -> Result<()>) -> Result<()> {
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

  /// apply `f` to every node of the tree, choosing next child randomly
  pub fn traverse_undeterministic(
    &self,
    f: &mut impl FnMut(&Self) -> Result<()>,
    rng: &mut (impl rand::Rng + ?Sized)
  ) -> Result<()> {
    f(self)?;
    self.traverse_undeterministic_a(f, rng)?;
    Ok(())
  }

  #[doc(hidden)]
  fn traverse_undeterministic_a(
    &self,
    f: &mut impl FnMut(&Self) -> Result<()>,
    rng: &mut (impl rand::Rng + ?Sized)
  ) -> Result<()> {
    use rand::prelude::*;
    if let Some(children) = &self.children {
      let mut ord: Vec<usize> = (0..4).collect();
      ord.shuffle(rng);
      for i in ord.iter() {
        f(&children[*i])?;
      }
      for i in ord.iter() {
        children[*i].traverse_undeterministic_a(f, rng)?;
      }
    }
    Ok(())
  }

  pub fn traverse_managed(&self, f: &mut impl FnMut(&Self) -> Result<TraverseCommand>) -> Result<()> {
    if f(self)? == TraverseCommand::Ok {
      self.traverse_managed_a(f)?;
    }
    Ok(())
  }

  #[doc(hidden)]
  fn traverse_managed_a(&self, f: &mut impl FnMut(&Self) -> Result<TraverseCommand>) -> Result<()> {
    if let Some(children) = &self.children {
      for child in children.iter() {
        if f(child)? == TraverseCommand::Ok {
          child.traverse_managed_a(f)?;
        }
      }
    }
    Ok(())
  }
}

impl<T: Default + Debug> Quadtree<T> {
  pub fn new(size: f32, center: Point<f32>, max_depth: u8) -> Self {
    Quadtree {
      rect: Rect {
        center,
        size,
      },
      children: None,
      depth: 0,
      max_depth,
      is_inside: false,
      data: T::default()
    }
  }

  pub fn subdivide(&mut self) -> &mut Option<Box<[Quadtree<T>; 4]>> {
    if self.depth < self.max_depth && self.children.is_none() {
      let rect = self.rect;
      let children: [Quadtree<T>; 4] = (0..4).into_iter()
        .map(|i| Quadtree {
          rect: Rect {
            center: Point {
              x: rect.size * CENTER_MAT[i][0] / 4.0 + rect.center.x,
              y: rect.size * CENTER_MAT[i][1] / 4.0 + rect.center.y,
            },
            size: rect.size / 2.0,
          },
          children: None,
          depth: self.depth + 1,
          max_depth: self.max_depth,
          is_inside: false,
          data: T::default()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
      self.children = Some(box children);
    }
    &mut self.children
  }

  /// subdivide recursively until reaching `depth`
  pub fn subdivide_deep(&mut self, depth: u8) {
    if depth == 0 { return; }
    if let Some(children) = self.subdivide() {
      for child in children.iter_mut() {
        child.subdivide_deep(depth - 1);
      }
    }
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
  pub fn path_to_pt(&self, pt: Point<f32>) -> Vec<&Self> {
    let mut result = vec![self];
    match self.children.as_deref() {
      Some(children) => {
        if let Some(quad) = Quadtrant::get(self.rect, pt) {
          result.append(&mut children[quad as usize].path_to_pt(pt));
        }
      },
      None => ()
    }
    result
  }

  /// find a smallest node containing pt
  pub fn pt_to_node(&self, pt: Point<f32>) -> Option<&Self> {
    let mut node = self;
    loop {
      match node.children.as_deref() {
        Some(children) =>
          node = &children[Quadtrant::get(node.rect, pt)? as usize],
        None => break
      }
    }
    Some(node)
  }

  /// find empty node, having max size. **too deterministic** for visualizations
  pub fn find_empty_rect(&self, rng: &mut (impl rand::Rng + ?Sized)) -> Option<Rect<f32>> {
    let mut candidate: Option<(Rect<f32>, u8)> = None;
    let mut count = 0;
    self.traverse_undeterministic(&mut |node| {
      if count > 16 && candidate.is_some() {
        error_chain::bail!("");
      }
      if node.children.is_none() && !node.is_inside {
        match &mut candidate {
          Some((_, depth))
            if *depth > node.depth
               => candidate = Some((node.rect, node.depth)),
          None => candidate = Some((node.rect, node.depth)),
          _ => ()
        }
        //result = Some(node.rect.center);
        //error_chain::bail!("");
      };
      count += 1;
      Ok(())
    }, rng).ok();

    candidate
      .map(|(rect, _)| rect)
  }

  pub fn find_max_empty_node(&self) -> Option<Point<f32>> {
    let mut candidate: Option<(Point<f32>, u8)> = None;

    self.traverse_managed(&mut |node| {
      match (node.children.is_none(), node.is_inside) {
        (_, true) => Ok(TraverseCommand::Skip), // is inside, skip
        (true, false) => {
          match &mut candidate {
            Some((_, depth))
              if node.depth < *depth
                 => candidate = Some((node.rect.center, node.depth)),
            None => candidate = Some((node.rect.center, node.depth)),
            _ => ()
          };
          Ok(TraverseCommand::Skip)
        },
        (false, false) =>
          if candidate.map(|x| x.1).unwrap_or(node.max_depth) > node.depth {
            Ok(TraverseCommand::Ok)
          } else {
            Ok(TraverseCommand::Skip)
          }
      }
    }).ok();

    candidate.map(|x| x.0)
  }

  /// subdivides the tree recursively on an edge of a shape, provided by `sdf` (signed distance function).
  /// marks nodes that are inside of a shape (`self.data`)
  pub fn insert_sdf(
    &mut self,
    sdf: &impl Fn(Point<f32>) -> f32,
    poly: impl geometry::Intersect<Rect<f32>> + Copy)
  {
    if self.is_inside { return; }
    let distance = sdf(self.rect.center);
    if poly.intersects(self.rect) &&
      distance.abs() < self.rect.size / 2.0 * std::f32::consts::SQRT_2 {
      self
        .subdivide()
        .as_deref_mut()
        .map(|children| children
          .iter_mut()
          .for_each(|child| child.insert_sdf(sdf, poly)));
    }
    match (self.children.as_deref_mut(), distance < 0.0) {
      (None, true) => self.is_inside = true,
      // for intersecting areas, slower
      /*(Some(children), true) => for child in children {
        child.insert_sdf(sdf);
      }*/
      _ => ()
    }
  }

  pub fn insert_sdf_strict(
    &mut self,
    sdf: &impl Fn(Point<f32>) -> f32,
    poly: impl geometry::Intersect<Rect<f32>> + Copy)
  {
    if self.is_inside { return; }
    let distance = sdf(self.rect.center);
    if poly.intersects(self.rect) &&
      distance.abs() < self.rect.size / 2.0 * std::f32::consts::SQRT_2 {
      self
        .subdivide()
        .as_deref_mut()
        .map(|children| children
          .iter_mut()
          .for_each(|child| child.insert_sdf_strict(sdf, poly)));
    }
    match (self.children.as_deref_mut(), distance < -self.rect.size / 2.0 * std::f32::consts::SQRT_2) {
      (None, true) => self.is_inside = true,
      _ => ()
    }
  }

  /// prints amount of total nodes in the tree, max subdivisions, and memory usage
  pub fn print_stats(&self) {
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
      std::mem::size_of::<Quadtree<T>>() * total_nodes as usize
    );
  }
}
