#![feature(try_trait)]
#![feature(box_syntax)]
#![feature(type_ascription)]
#![feature(array_map)]
#![allow(dead_code)]

use std::{thread};
use lib::{Result, Point};
use quadtree::{sdf, Quadtree};

mod quadtree;
mod drawing;
#[path = "util.rs"]
mod lib;

const WORLD_SIZE: f32 = 512.0;
/// final image resolution is `WORLD_SIZE` * `IMG_SCALE`
const IMG_SCALE: f32 = 8.0;

fn main() -> Result<()> {
  thread::Builder::new()
    .spawn(||{
      img_test()
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
    let c = sdf::Circle {
        xy: Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
        r: WORLD_SIZE / 4.0 - 1e-4
    };
    tree.insert_sdf(&|sample| sdf::circle(sample, c), c);
  });
  tree.print_stats();
  profile!("draw", { drawing::tree_test(&tree, "out.png".into())?; });
  open::that("out.png")?;
  Ok(())
}

fn img_test() -> Result<()> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    9
  );

  let mut circles = vec![];
  profile!("tree", {
    for _ in 0..6153 {
      //println!("{}", i);
      let rect = loop {
        let pt = Point { x: rng.gen_range(0.0..WORLD_SIZE), y: rng.gen_range(0.0..WORLD_SIZE)};
        //let pt = match tree.find_empty_pt(&mut rng)
        let path = tree.path_to_pt(pt);
        let node = path.last()?;
        if !node.data {
          break node.rect;
        }
      };
      //let rect = node.last()?.rect;
      let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(0.5) * rect.size / (3.0 + 1e-3)).max(0.5 - 1e-3);
      let delta = rect.size / 2.0 - r;
      let c = sdf::Circle {
        xy: Point {
          x: rng.gen_range(rect.center.x - delta..rect.center.x + delta),
          y: rng.gen_range(rect.center.y - delta..rect.center.y + delta)
        },
        r
      };
      tree.insert_sdf(&|sample| sdf::circle(sample, c), c);
      circles.push(c);
    };
  });
  tree.print_stats();
  profile!("draw", drawing::exec_img(circles, "out.png".into())?);
  open::that("out.png")?;
  Ok(())
}

fn expand_area_test() -> Result<()> {
  use rand::prelude::*;

  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    10
  );
  let big_circle = sdf::Circle {
    xy: Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    r: WORLD_SIZE / 4.0 - 1e-4
  };
  tree.insert_sdf(&|sample| sdf::circle(sample, big_circle), big_circle);

  /*let c = sdf::Circle {
    xy: Point { x: 165.0, y: 858.5 },
    r: 155.16904 - 1.0
  };
  tree.insert_sdf(&|sample| sdf::circle(sample, c), c);*/

  let path = format!("anim/#{:04}.png", 0);
  let img = drawing::tree_test(&tree, &path)?;
  drop(img);

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  for frame in 1..10 {
    //for _ in 1..16 {
      if let Some(pt) = tree.find_empty_pt(&mut rng) {
    //if let pt = { Point { x: 128.0, y: 128.0 } } {
        let path = tree.path_to_pt(pt);
        let (trbl, points) = path.get(path.len() - 2)?.find_max_free_area_attempt_7(pt)?;
        //let rect = tree.path_to_pt(pt).last()?.rect;
        //let trbl: sdf::TLBR = rect.into();
        let rect: sdf::Rect = trbl.into();
        println!("{:?}", rect);
        let c = sdf::Circle {
          xy: rect.center,
          r: rect.size / 2.0 - 1e-4
        };
        tree.insert_sdf(&|sample| sdf::circle(sample, c), c);

        use plotters::prelude::*;
        let path = format!("anim/#{:04}.png", frame);
        let mut img = drawing::tree_test(&tree, &path)?;
        {
          img.draw_rect(
            (trbl.tl.x as i32, trbl.tl.y as i32),
            (trbl.br.x as i32, trbl.br.y as i32),
            &WHITE,
            false
          ).ok()?;
          img
            .draw_circle((pt.x as i32, pt.y as i32), 6, &RED, true)
            .ok()?;
          for (i, pt) in points.iter().enumerate() {
            img
              .draw_circle((pt.x as i32, pt.y as i32), 6, &[
                WHITE, // TL
                GREEN, // TR
                BLUE,  // BL
                YELLOW // BR
              ][i], true)
              .ok()?;
            //img.draw_line((0, pt.y as i32), (WORLD_SIZE as i32, pt.y as i32), &WHITE).ok()?;
            //img.draw_line((pt.x as i32, 0), (pt.x as i32, WORLD_SIZE as i32), &WHITE).ok()?;
          }

        };
      }
  }
  open::that("anim\\#0001.png")?;
  Ok(())
}
