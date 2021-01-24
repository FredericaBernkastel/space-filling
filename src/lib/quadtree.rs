use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use crate::{
  *,
  error::Result,
  geometry::{Point, TLBR, Rect},
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
pub struct Quadtree {
  pub rect: Rect<f32>,
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

impl Quadtree {
  pub fn new(size: f32, center: Point<f32>, max_depth: u8) -> Self {
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

  pub fn subdivide(&mut self) -> &mut Option<Box<[Quadtree; 4]>> {
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

  /// find empty node, having max size. **too deterministic** for visualizations
  pub fn find_empty_pt(&self, rng: &mut (impl rand::Rng + ?Sized)) -> Option<Point<f32>> {
    let mut candidate: Option<(Point<f32>, u8)> = None;
    let mut count = 0;
    self.traverse_undeterministic(&mut |node| {
      if count > 8192 && candidate.is_some() {
        error_chain::bail!("");
      }
      if node.children.is_none() && !node.data {
        match &mut candidate {
          Some((_, depth))
            if *depth > node.depth
               => candidate = Some((node.rect.center, node.depth)),
          None => candidate = Some((node.rect.center, node.depth)),
          _ => ()
        }
        //result = Some(node.rect.center);
        //error_chain::bail!("");
      };
      count += 1;
      Ok(())
    }, rng).ok();

    candidate
      .map(|(point, _)| point)
  }

  /// subdivides the tree recursively on an edge of a shape, provided by `sdf` (signed distance function).
  /// marks nodes that are inside of a shape (`self.data`)
  pub fn insert_sdf(
    &mut self,
    sdf: &impl Fn(Point<f32>) -> f32,
    poly: impl geometry::Intersect<Rhs = Rect<f32>> + Copy)
  {
    if self.data { return; }
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
      (None, true) => self.data = true,
      // for intersecting areas, slower
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
    self.traverse(&mut |node| {
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

  const fn vertex(rect: TLBR<f32>, quad: Quadtrant) -> Point<f32> {
    match quad {
      Quadtrant::TL => rect.tl,
      Quadtrant::BR => rect.br,
      Quadtrant::BL => Point { x: rect.tl.x, y: rect.br.y },
      Quadtrant::TR => Point { x: rect.br.x, y: rect.tl.y },
    }
  }

  pub fn find_max_free_area_attempt_6(&self, seed: Point<f32>) -> Result<TLBR<f32>> {

    fn walk(root: &Quadtree, dir: Quadtrant, dir_orig: Quadtrant) -> Point<f32> {
      match root.children.as_deref() {
        Some(children) => {
          if children[dir as usize].children.is_some() {
            walk(&children[dir as usize], dir, dir_orig)
          } else if children[dir.inv() as usize].children.is_some() {
            walk(&children[dir.inv() as usize], dir.inv(), dir_orig)
          } else if children[dir.mirror_x() as usize].children.is_some() {
            walk(&children[dir.mirror_x() as usize], dir.mirror_x(), dir_orig)
          } else if children[dir.mirror_y() as usize].children.is_some() {
            walk(&children[dir.mirror_y() as usize], dir.mirror_y(), dir_orig)
          } else {
            Quadtree::vertex(root.rect.into(), dir_orig)
          }
        },
        None => Quadtree::vertex(root.rect.into(), dir_orig)
      }
    }

    let path = self.path_to_pt(seed);

    let mut max_rect: TLBR<f32> = path.last()?.rect.into();

    if let Some(root) = path.get(path.len() - 2) {
      use Quadtrant::*;
      let mat = [TL, TR, BL, BR]
        .map(|quad| {
          walk(&root.children.as_ref().unwrap()[quad as usize], quad.inv(), quad)
        });

      println!("{:#?}", mat);

      max_rect = TLBR {
        tl: Point {
          x: mat[TL as usize].x.max(mat[BL as usize].x),
          //x: mat[TL as usize].x,
          y: mat[TL as usize].y.max(mat[TR as usize].y),
          //y: mat[TL as usize].y
        },
        br: Point {
          //x: max_rect.br.x,
          x: mat[TR as usize].x.min(mat[BR as usize].x),
          //y: max_rect.br.y,
          y: mat[BL as usize].y.min(mat[BR as usize].y)
        }
      };
      println!("{:?}", max_rect);
    }

    Ok(max_rect)
  }

  pub fn find_max_free_area_attempt_7(&self, seed: Point<f32>) -> Result<(TLBR<f32>, Vec<Point<f32>>)> {
    fn walk(tree: &Quadtree, domain: TLBR<f32>, seed: Point<f32>, mut nearest: Point<f32>, mut nearest_dist: f32) -> (Point<f32>, f32) {
      if let Some(children) = tree.children.as_deref() {
        for child in children.iter() {
          let dist = (child.rect.center - seed).length();
          if child.data && dist < nearest_dist && child.rect.center.in_rect(domain) {
            nearest = child.rect.center;
            nearest_dist = dist;
          }
          let (point, dist) = walk(child, domain, seed, nearest, nearest_dist);
          if dist < nearest_dist {
            nearest = point;
            nearest_dist = dist;
          }
        }
      }
      (nearest, nearest_dist)
    };

    let children = self.children.as_deref()?;

    use Quadtrant::*;

    let domain_tl: TLBR<f32> = children[TL as usize].rect.into();
    let domain_tr: TLBR<f32> = children[TR as usize].rect.into();
    let domain_bl: TLBR<f32> = children[BL as usize].rect.into();
    let domain_br: TLBR<f32> = children[BR as usize].rect.into();

    let tl = walk(self, domain_tl, seed, domain_tl.tl, (seed - domain_tl.tl).length()).0;
    let tr = walk(self, domain_tr, seed, domain_tr.tr(), (seed - domain_tr.tr()).length()).0;
    let bl = walk(self, domain_bl, seed, domain_bl.bl(), (seed - domain_bl.bl()).length()).0;
    let br = walk(self, domain_br, seed, domain_br.br, (seed - domain_br.br).length()).0;

    Ok((TLBR {
      tl: Point {
        x: tl.x.max(bl.x),
        y: tl.y.max(tr.y),
      },
      br: Point {
        x: tr.x.min(br.x),
        y: bl.y.min(br.y)
      }
    }, vec![tl, tr, bl, br]))
  }
}
