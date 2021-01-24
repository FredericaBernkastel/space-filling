#![feature(type_ascription)]
#![allow(dead_code)]
#![feature(array_map)]

use std::thread;
use lib::{
  error::Result,
  geometry::{Point, Circle, Rect},
  sdf::SDF,
  quadtree::Quadtree,
  argmax::Argmax,
  drawing,
  profile
};

fn main() -> Result<()> {
  thread::Builder::new()
    .spawn(||{
      let circles = sdf_argmax_bruteforce_test()?;

      profile!("draw", {
        drawing::exec_img_parallel(
          "out.png",
          circles.into_iter(),
          lib::find_files("H:\\Temp\\export\\bottle-fairy"),
          Point { x: 1024, y: 1024 },
          4
        )?;
      });
      Ok(())
    })?
    .join()
    .unwrap()
}

/// single circle in the middle, draw tree layout
fn basic_test() -> Result<()> {
  let mut tree = Quadtree::new(
    1.0,
    Point { x: 1.0 / 2.0, y: 1.0 / 2.0 },
    10
  );
  profile!("tree", {
    let c = Circle {
        xy: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 },
        r: 1.0 / 4.0 - 1e-6
    };
    tree.insert_sdf(&|sample| c.sdf(sample), c);
  });
  tree.print_stats();
  profile!("draw", {
    drawing::tree_test(
      "out.png".into(),
      &tree,
      Point { x: 1024, y: 1024 }
    )?;
  });
  open::that("out.png")?;
  Ok(())
}

/// 1'000'000 random non-intersecting circles
fn test_1000000() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut tree = Quadtree::new(
    1.0,
    Point { x: 1.0 / 2.0, y: 1.0 / 2.0 },
    12
  );
  let mut circles = vec![];

  profile!("tree", {
    for _ in 0..100 {
      let rect = loop {
        let pt = Point { x: rng.gen_range(0.0..1.0), y: rng.gen_range(0.0..1.0)};
        //let pt = match tree.find_empty_pt(&mut rng)
        let path = tree.path_to_pt(pt);
        let node = path.last()?;
        if !node.data {
          break node.rect;
        }
      };
      let r = rng.gen_range::<f32, _>(0.0..1.0).powf(1.0) * rect.size / (3.0 + 1e-3);
      let delta = rect.size / 2.0 - r;
      let c = Circle {
        xy: Point {
          x: rng.gen_range(rect.center.x - delta..rect.center.x + delta),
          y: rng.gen_range(rect.center.y - delta..rect.center.y + delta)
        },
        r
      };
      tree.insert_sdf(&|pixel| c.sdf(pixel), c);
      circles.push(c);
    }
  });
  tree.print_stats();
  Ok(circles)
}

fn sdf_argmax_bruteforce_test() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax::new(Point { x: 1024, y: 1024 });

  // insert boundary rect SDF
  argmax.insert_sdf(
    |pixel| {
      -Rect { center: Point {x: 1.0 / 2.0, y: 1.0 / 2.0}, size: 1.0 }
        .sdf(pixel)
    }
  )?;

  profile! ("argmax", {
    'argmax: for i in 0..6153 {
      let min_distance = 1.0 / argmax.size.x as f32;

      let argmax_ret = argmax.find_max();

      if argmax_ret.distance < min_distance {
        println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
        break 'argmax;
      }

      let circle = {
        use std::f32::consts::PI;

        let angle = rng.gen_range::<f32, _>(-PI..=PI);
        let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(0.5) * argmax_ret.distance)
          .min(1.0 / 6.0)
          .max(min_distance);
        let delta = argmax_ret.distance - r;
        let offset = Point { x: delta * angle.sin(), y: delta * angle.cos() };

        Circle {
          xy: { argmax_ret.point.into(): Point<f32> }.translate(offset), r
        }
      };

      if i % 1000 == 0 { println!("argmax #{}", i); }
      argmax.insert_sdf(|pixel| circle.sdf(pixel))?;
      circles.push(circle);
    };
  });

  argmax.display_debug("out_dist_map.exr", None)?;

  Ok(circles)
}