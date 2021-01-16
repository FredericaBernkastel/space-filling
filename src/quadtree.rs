use crate::lib::{Result, Point};

pub mod sdf;

#[derive(Copy, Clone, Debug)]
pub struct AABB {
  pub center: Point,
  pub size: f32,
}

#[derive(Debug)]
pub struct Quadtree {
  pub boundary: AABB,
  pub children: Option<Box<[Quadtree; 4]>>,
  pub depth: u8,
  pub max_depth: u8,
  pub data: bool
}

impl Quadtree {
  pub fn new(size: f32, center: Point, max_depth: u8) -> Self {
    Quadtree {
      boundary: AABB {
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
      let aabb = self.boundary;
      self.children = Some(box [
        Quadtree {
          boundary: AABB {
            center: Point {
              x: -aabb.size / 4.0 + aabb.center.x,
              y: -aabb.size / 4.0 + aabb.center.y,
            },
            size: aabb.size / 2.0,
          },
          children: None,
          depth: self.depth + 1,
          max_depth: self.max_depth,
          data: false
        },
        Quadtree {
          boundary: AABB {
            center: Point {
              x: aabb.size / 4.0 + aabb.center.x,
              y: -aabb.size / 4.0 + aabb.center.y,
            },
            size: aabb.size / 2.0,
          },
          children: None,
          depth: self.depth + 1,
          max_depth: self.max_depth,
          data: false
        },
        Quadtree {
          boundary: AABB {
            center: Point {
              x: -aabb.size / 4.0 + aabb.center.x,
              y: aabb.size / 4.0 + aabb.center.y,
            },
            size: aabb.size / 2.0,
          },
          children: None,
          depth: self.depth + 1,
          max_depth: self.max_depth,
          data: false
        },
        Quadtree {
          boundary: AABB {
            center: Point {
              x: aabb.size / 4.0 + aabb.center.x,
              y: aabb.size / 4.0 + aabb.center.y,
            },
            size: aabb.size / 2.0,
          },
          children: None,
          depth: self.depth + 1,
          max_depth: self.max_depth,
          data: false
        }
      ]);
    }
    &mut self.children
  }

  pub fn subdivide_deep(&mut self, depth: u8) {
    if depth == 0 { return; }
    if let Some(children) = self.subdivide() {
      for child in children.iter_mut() {
        child.subdivide_deep(depth - 1);
      }
    }
  }

  pub fn traverse(&self, f: &mut impl FnMut(u8, &Self) -> Result<()>) -> Result<()> {
    f(self.depth, self)?;
    self.traverse_a(f)?;
    Ok(())
  }

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

  fn nodes_planar(&mut self) -> Vec<&mut Self> {
    let mut result = vec![];
    if let Some(children) = self.children.as_deref_mut() {
      for child in children.iter_mut() {
        result.push(child);
      }
    }
    result
  }

  pub fn insert_sdf(&mut self, sdf: &impl Fn(Point) -> f32) {
    let distance = sdf(self.boundary.center);
    if distance.abs() < self.boundary.size / 2.0 * std::f32::consts::SQRT_2 {
      if let Some(children) = self.subdivide() {
        for child in children.iter_mut() {
          child.insert_sdf(sdf)
        }
      }
    }
    self.data = self.data || self.children.is_none() && distance < 0.0;
  }

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

pub fn exec() -> Result<Quadtree> {
  let mut tree = Quadtree::new(1024.0, Point { x: 512.0, y: 512.0 }, 10);

  tree.insert_sdf(&|sample|
    sdf::circle(sample, sdf::Circle {
      xy: Point { x: 512.0 - 128.0, y: 512.0 },
      r: 128.0
    })
  );
  tree.insert_sdf(&|sample|
    sdf::circle(sample, sdf::Circle {
      xy: Point { x: 512.0 + 128.0, y: 512.0 },
      r: 128.0
    })
  );

  /* use rayon::prelude::*;
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
    );*/

  /*tree.insert_sdf(&|sample|
    sdf::circle(sample, sdf::Circle {
      xy: Point { x: 512.0, y: 512.0 },
      r: 256.0
    })
  );*/

  //let mut rng = rand_pcg::Pcg32::seed_from_u64(0);
  /*for _ in 0..10 {
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
  }*/
  Ok(tree)
}