#![feature(type_ascription)]
#![allow(dead_code)]
#![feature(array_map)]
#![feature(total_cmp)]

use std::thread;
use lib::{
  error::Result,
  geometry::{Point, Circle, Rect},
  sdf::SDF,
  quadtree::Quadtree,
  argmax::Argmax,
  gpu,
  drawing,
  profile
};
use lib::argmax::ArgmaxResult;
use lib::geometry::TLBR;

fn main() -> Result<()> {
  thread::Builder::new()
    .spawn(||{
      let circles = sdf_argmax_gpu_test()?;
      drawing::exec("out_gpu.png", circles.into_iter(), Point { x: 4096, y: 4096 } )?;
      Ok(())
    })?
    .join()
    .unwrap()
}

/// single circle in the middle, draw tree layout
fn basic_test() -> Result<()> {
  let mut tree = Quadtree::<()>::new(
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
    drawing::tree_display(
      "out.png".into(),
      &tree,
      Point { x: 2048, y: 2048 }
    )?;
  });
  open::that("out.png")?;
  Ok(())
}

/// 1'000'000 random non-intersecting circles
fn test_1000000() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut tree = Quadtree::<()>::new(
    1.0,
    Point { x: 1.0 / 2.0, y: 1.0 / 2.0 },
    12
  );
  let mut circles = vec![];

  profile!("tree", {
    for _ in 0..1000000 {
      let rect = loop {
        let pt = Point { x: rng.gen_range(0.0..1.0), y: rng.gen_range(0.0..1.0)};
        //let pt = match tree.find_empty_pt(&mut rng)
        let path = tree.path_to_pt(pt);
        let node = path.last()?;
        if !node.is_inside {
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
  let mut argmax = Argmax::new(Point { x: 4096, y: 4096 });

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

      if i % 1000 == 0 {
        println!("#{} argmax = {}", i, argmax_ret.distance * argmax.size.x as f32);
      }

      argmax.insert_sdf(|pixel| circle.sdf(pixel))?;
      circles.push(circle);
    };
  });

  //argmax.display_debug("out_dist_map.exr", None)?;

  Ok(circles)
}

fn sdf_argmax_gpu_test() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax::new(Point { x: 1024, y: 1024 });
  let mut gpu_kernel = gpu::KernelWrapper::new(&argmax.dist_map)?;

  // insert boundary rect SDF
  {
    let rect = Rect { center: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 }, size: 1.0 };
    argmax.insert_sdf(|pixel| -rect.sdf(pixel))?;
  }

  //argmax.display_debug(&format!("anim/dist_map_{:04}.exr", 0), None)?;

  gpu_kernel.write_to_device(&argmax.dist_map)?;

  profile!("argmax", {
    'argmax: for i in 0..4 {
      let min_distance = 0.0 / argmax.size.x as f32;

      let argmax_ret = gpu_kernel.find_max()?;

      if argmax_ret.distance < min_distance {
        println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
        break 'argmax;
      }

      //gpu_kernel.read_from_device(&mut argmax.dist_map)?;
      //argmax.display_debug(&format!("anim/dist_map_{:04}.exr", i), Some(argmax_ret.point))?;

      let circle = {
        use std::f32::consts::PI;

        /*Circle {
          xy: { argmax_ret.point.into(): Point<f32> }, r: argmax_ret.distance.min(1.0 / 4.0)
        }*/

        let angle = rng.gen_range::<f32, _>(-PI..=PI);
        let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(0.5) * argmax_ret.distance)
        //let r = (0.5 * argmax_ret.distance)
          .min(1.0 / 6.0)
          .max(min_distance);
        let delta = argmax_ret.distance - r;
        let offset = Point { x: delta * angle.sin(), y: delta * angle.cos() };

        Circle {
          xy: { argmax_ret.point.into(): Point<f32> }.translate(offset), r
        }
      };

      if i % 1000 == 0 {
        println!("#{} argmax = {}", i, argmax_ret.distance * argmax.size.x as f32);
      }

      let domain = Rect {
        center: circle.xy,
        size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2
      }.into(): TLBR<f32>;
      gpu_kernel.insert_sdf_circle_domain(circle, domain)?;
      circles.push(circle);
    }
  });

  Ok(circles)
}

fn sdf_argmax_domain_hypothesis_test() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax::new(Point { x: 2048, y: 2048 });

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

      //argmax.display_debug(&format!("anim/dist_map_{:04}.exr", i), Some(argmax_ret.point))?;

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

      if i % 1000 == 0 {
        println!("#{} argmax = {}", i, argmax_ret.distance * argmax.size.x as f32);
      }

      argmax.insert_sdf_domain(
        Rect { center: circle.xy, size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2 }.into(): TLBR<f32>,
        |pixel| circle.sdf(pixel)
      )?;
      circles.push(circle);
    };
  });

  Ok(circles)
}