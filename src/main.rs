#![feature(try_trait)]
#![feature(box_syntax)]
#![allow(dead_code)]

use std::{thread};
use lib::{Result, Point};
use quadtree::{sdf, Quadtree};

mod quadtree;
mod drawing;
#[path = "util.rs"]
mod lib;

const WORLD_SIZE: f32 = 1024.0;

fn main() -> Result<()> {
  thread::Builder::new()
    .spawn(||{
      insert_10000()
    })?
    .join()
    .unwrap()
}

/// single circle in the middle, draw tree layout
fn basic_test() -> Result<()> {
  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    10
  );
  profile!("tree", {
    tree.insert_sdf(&|sample|
      sdf::circle(sample, sdf::Circle {
        xy: Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
        r: WORLD_SIZE / 4.0
    }));
  });
  tree.print_stats();
  profile!("draw", drawing::tree_test(tree, "out.png".into())?);
  open::that("out.png")?;
  Ok(())
}

/// 10'000 random non-intersecting circles
fn insert_10000() -> Result<()> {
  use rand::prelude::*;
  use rand_pcg::Pcg64;

  let mut rng = Pcg64::seed_from_u64(0);

  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    10
  );
  let mut circles = vec![];
  profile!("tree", {
    let c0 = sdf::Circle {
      xy: tree.rect.center,
      r: WORLD_SIZE / 4.0
    };
    tree.insert_sdf(&|sample|
      sdf::circle(sample, c0));
    circles.push(c0);
    let mut count = 0;
    while count < 10000 {
      let pt = Point { x: rng.gen_range(0.0..WORLD_SIZE), y: rng.gen_range(0.0..WORLD_SIZE) };

      let quad = *tree.path_to_pt(pt).last()?;
      if !quad.data {
        let rect = quad.rect;
        let c = sdf::Circle {
          xy: rect.center,
          r: rect.size / 2.0
        };
        tree.insert_sdf(&|sample|
          sdf::circle(sample, c));
        circles.push(c);
        count += 1;
      }
    };
  });
  tree.print_stats();
  profile!("draw", drawing::exec(circles, "out.png".into())?);
  open::that("out.png")?;
  Ok(())
}
