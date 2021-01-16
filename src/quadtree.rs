use crate::lib::{Result, Point};
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};

pub mod sdf;

#[derive(Copy, Clone, Debug)]
pub struct Rect {
  pub center: Point,
  pub size: f32,
}

/// Example usage:
/// ```
/// let tree = Quadtree::new(512.0, Point { x: 256.0, y: 256.0 }, 9);
/// tree.insert_sdf(&|sample|
///   sdf::circle(sample, sdf::Circle {
///     xy: Point { x: 256.0, y: 256.0 },
///     r: 128.0
/// }));
/// ```
pub struct Quadtree {
  pub rect: Rect,
  pub children: Option<Box<[Quadtree; 4]>>,
  pub depth: u8,
  pub max_depth: u8,
  pub data: bool
}

impl Debug for Quadtree {
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
#[derive(Debug)]
/// 4 sections of a rectangle
enum Quadtrant {
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
  pub fn get(rect: Rect, pt: Point) -> Option<Self> {
    let center: [Point; 4] = (0..4).into_iter()
      .map(|i| Point {
        x: rect.size * CENTER_MAT[i][0] / 4.0 + rect.center.x,
        y: rect.size * CENTER_MAT[i][1] / 4.0 + rect.center.y,
      })
      .collect::<Vec<_>>()
      .try_into()
      .unwrap();
    let size = rect.size / 2.0;
    match pt {
      z if z.in_rect(Rect { center: center[Quadtrant::TL as usize], size }) => Some(Self::TL),
      z if z.in_rect(Rect { center: center[Quadtrant::TR as usize], size }) => Some(Self::TR),
      z if z.in_rect(Rect { center: center[Quadtrant::BL as usize], size }) => Some(Self::BL),
      z if z.in_rect(Rect { center: center[Quadtrant::BR as usize], size }) => Some(Self::BR),
      _ => None
    }
  }
}

impl Quadtree {
  pub fn new(size: f32, center: Point, max_depth: u8) -> Self {
    Quadtree {
      rect: Rect {
        center,
        size,
      },
      children: None,
      depth: 0,
      max_depth,
      data: false
    }
  }

  fn subdivide(&mut self) -> &mut Option<Box<[Quadtree; 4]>> {
    if self.depth < self.max_depth && self.children.is_none() {
      let rect = self.rect;
      let children: [Quadtree; 4] = (0..4).into_iter()
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
          data: false
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
      self.children = Some(box children);
    }
    &mut self.children
  }

  /// subdivide recursively until reaching `depth`
  fn subdivide_deep(&mut self, depth: u8) {
    if depth == 0 { return; }
    if let Some(children) = self.subdivide() {
      for child in children.iter_mut() {
        child.subdivide_deep(depth - 1);
      }
    }
  }

  /// apply `f` to every node of the tree
  pub fn traverse(&self, f: &mut impl FnMut(u8, &Self) -> Result<()>) -> Result<()> {
    f(self.depth, self)?;
    self.traverse_a(f)?;
    Ok(())
  }

  #[doc(hidden)]
  fn traverse_a(&self, f: &mut impl FnMut(u8, &Self) -> Result<()>) -> Result<()> {
    if let Some(children) = &self.children {
      for child in children.iter() {
        f(child.depth, child)?;
      }
      for child in children.iter() {
        child.traverse_a(f)?;
      }
    }
    Ok(())
  }

  #[doc(hidden)]
  fn nodes_planar(&mut self) -> Vec<&mut Self> {
    let mut result = vec![];
    if let Some(children) = self.children.as_deref_mut() {
      for child in children.iter_mut() {
        result.push(child);
      }
    }
    result
  }

  /// return all nodes, containing `pt`
  pub fn path_to_pt(&self, pt: Point) -> Vec<&Self> {
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

  /// find empty node. **too deterministic** for visualizations
  pub fn find_empty_pt(&mut self) -> Option<Point> {
    let mut result = None;
    self.traverse(&mut |_, node| {
      if node.children.is_none() && !node.data {
        result = Some(node.rect.center);
        error_chain::bail!("");
      }
      Ok(())
    }).ok();
    result
  }

  /// subdivides the tree recursively on an edge of a shape, provided by `sdf` (signed distance function).
  /// marks nodes that are inside of a shape (`self.data`)
  pub fn insert_sdf(&mut self, sdf: &impl Fn(Point) -> f32) {
    if self.data { return; }
    let distance = sdf(self.rect.center);
    if distance.abs() < self.rect.size / 2.0 * std::f32::consts::SQRT_2 {
      if let Some(children) = self.subdivide() {
        for child in children.iter_mut() {
          child.insert_sdf(sdf);
        }
      }
    }

    match (self.children.as_deref_mut(), distance < 0.0) {
      (None, true) => self.data = true,
      // for intersecting cases, slower
      /*(Some(children), true) => for child in children {
        child.insert_sdf(sdf);
      }*/
      _ => ()
    }
  }

  /// prints amount of total nodes in the tree, max subdivisions, and memory usage
  pub fn print_stats(&self) {
    let mut total_nodes = 0u64;
    let mut max_depth = 0u8;
    self.traverse(&mut |_, node| {
      total_nodes += 1;
      max_depth = (max_depth).max(node.depth);
      Ok(())
    }).ok();
    println!(
      "total nodes: {}\n\
      max subdivisions: {}\n\
      mem::size_of::<Quadtree>(): {}",
      total_nodes,
      max_depth,
      std::mem::size_of::<Quadtree>() * total_nodes as usize
    );
  }
}

/*pub fn exec() -> Result<Quadtree> {
  use rayon::prelude::*;
  tree.subdivide_deep(1);
  tree
    .nodes_planar()
    .par_iter_mut()
    .for_each(|tree|
      tree.insert_sdf(&|sample|
        sdf::circle(sample, sdf::Circle {
          xy: Point { x: 512.0, y: 512.0 },
          r: 256.0
        })
      )
    );

  tree.insert_sdf(&|sample|
    sdf::circle(sample, sdf::Circle {
      xy: Point { x: 512.0, y: 512.0 },
      r: 256.0
    })
  );

  //let mut rng = rand_pcg::Pcg32::seed_from_u64(0);
  for _ in 0..10 {
    let x: f32 = rng.gen_range(0.0..511.0);
    let y: f32 = rng.gen_range(0.0..511.0);
    let nearest = tree
      .nearest(&[x, y], 1, &squared_euclidean)?
      .get(0)
      .cloned()?;
    let dist = nearest.0.sqrt() - nearest.1.r;
    if dist > 0.0 {
      println!("([{}, {}], {})", x, y, dist);
      tree.add([x, y], Circle { x, y, r: dist })?;
    }
  }
  Ok(tree)
}*/