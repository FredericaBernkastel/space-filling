use crate::{
  error::{ErrorKind::NoneError, Result},
  geometry::{Point, Circle, Rect, TLBR},
  sdf::SDF,
  legacy::{self, quadtree::Quadtree, argmax::Argmax, gpu},
  profile
};

/// single circle in the middle, draw tree layout
pub fn basic_test() -> Result<()> {
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
    legacy::drawing::tree_display(
      "out.png".into(),
      &tree,
      Point { x: 2048, y: 2048 }
    )?;
  });
  open::that("out.png")?;
  Ok(())
}

/// 1'000'000 random non-intersecting circles
/// 1901ms on i5-6600K @ 3.5GHz, RUSTFLAGS="-C target-cpu=native"
pub fn test_1000000() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut tree = Quadtree::<()>::new(
    1.0,
    Point { x: 1.0 / 2.0, y: 1.0 / 2.0 },
    11
  );
  let mut circles = vec![];

  for _ in 0..1000000 {
    //let rect = tree.find_empty_rect(&mut rng)?;
    let rect = loop {
      let pt = Point { x: rng.gen_range(0.0..1.0), y: rng.gen_range(0.0..1.0)};
      let node = tree.pt_to_node(pt).ok_or(NoneError)?;
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
  };
  tree.print_stats();
  Ok(circles)
}

/// 1967ms, 1000 circrles, Δ = 2^-10
pub fn sdf_argmax_bruteforce_test() -> Result<Vec<Circle>> {
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

  'argmax: for i in 0..1000 {
    let min_distance = 1.0 / argmax.dist_map.width() as f32;

    let argmax_ret = argmax.find_max();
    //argmax.display_debug(&format!("tmp/{}_gt.exr", i), Some(argmax_ret.point))?;

    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax;
    }

    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(1.0) * argmax_ret.distance)
        .min(1.0 / 6.0)
        .max(min_distance);
      let delta = argmax_ret.distance - r;
      let offset = Point { x: delta * angle.sin(), y: delta * angle.cos() };

      Circle {
        xy: { argmax_ret.point.into(): Point<f32> }.translate(offset), r
      }
    };

    if i % 1000 == 0 {
      println!("#{} argmax = {}", i, argmax_ret.distance * argmax.dist_map.width() as f32);
    }

    argmax.insert_sdf(|pixel| circle.sdf(pixel))?;
    circles.push(circle);
  };

  Ok(circles)
}

/// 395ms on GTX1060 @ 1506MHz, 1000 circrles, Δ = 2^-10
pub fn sdf_argmax_gpu_test() -> Result<Vec<Circle>> {
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

  'argmax: for i in 0..1000 {
    let min_distance = 0.5 / argmax.dist_map.width() as f32;

    let argmax_ret = gpu_kernel.find_max()?;

    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax;
    }

    //gpu_kernel.read_from_device(&mut argmax.dist_map)?;
    //argmax.display_debug(&format!("anim/dist_map_{:04}.exr", i), Some(argmax_ret.point))?;

    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(1.0) * argmax_ret.distance)
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
      println!("#{} argmax = {}", i, argmax_ret.distance * argmax.dist_map.width() as f32);
    }

    let domain = Rect {
      center: circle.xy,
      size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2
    }.into(): TLBR<f32>;
    gpu_kernel.insert_sdf_circle_domain(circle, domain)?;
    circles.push(circle);
  }

  Ok(circles)
}

/// 978ms, 1000 circrles, Δ = 2^-10
pub fn sdf_argmax_domain_hypothesis_test() -> Result<Vec<Circle>> {
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

  'argmax: for i in 0..1000 {
    let min_distance = 1.0 / argmax.dist_map.width() as f32;

    let argmax_ret = argmax.find_max();

    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax;
    }

    //argmax.display_debug(&format!("anim/dist_map_{:04}.exr", i), Some(argmax_ret.point))?;

    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(1.0) * argmax_ret.distance)
        .min(1.0 / 6.0)
        .max(min_distance);
      let delta = argmax_ret.distance - r;
      let offset = Point { x: delta * angle.sin(), y: delta * angle.cos() };

      Circle {
        xy: { argmax_ret.point.into(): Point<f32> }.translate(offset), r
      }
    };

    if i % 1000 == 0 {
      println!("#{} argmax = {}", i, argmax_ret.distance * argmax.dist_map.width() as f32);
    }

    argmax.insert_sdf_domain(
      Rect { center: circle.xy, size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2 }.into(): TLBR<f32>,
      |pixel| circle.sdf(pixel)
    )?;
    circles.push(circle);
  };

  Ok(circles)
}

/// 92ms, 1000 circrles, Δ = 2^-10
pub fn sdf_convolution_domain_test() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax::new(Point { x: 1024, y: 1024 });

  // insert boundary rect SDF
  {
    let boundary = Rect { center: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 }, size: 1.0 };
    argmax.insert_sdf(|pixel| -boundary.sdf(pixel))?;
  }
  argmax.convolute_domain([0.0, 1.0]);

  'argmax_outer: for i in 0..1000 {
    let min_distance = 0.5 * std::f32::consts::SQRT_2 / argmax.dist_map.width() as f32;

    let argmax_ret = argmax.find_max_convolution();
    /*drawing::display_debug_convolution(
      &format!("tmp/{}_conv.png", i),
      &argmax,
      Some(argmax_ret.point)
    )?;*/

    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax_outer;
    }

    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(min_distance..1.0).powf(1.0) * argmax_ret.distance)
        //let r = (argmax_ret.distance / 4.0)
        .min(1.0 / 6.0);
      let delta = argmax_ret.distance - r;
      let offset = Point { x: delta * angle.cos(), y: delta * angle.sin() };

      Circle {
        xy: { argmax_ret.point.into(): Point<f32> }.translate(offset), r
      }
    };

    if i % 1000 == 0 {
      println!("#{} argmax = {}", i, argmax_ret.distance * argmax.dist_map.width() as f32);
    }

    let domain: TLBR<f32> = Rect {
      center: circle.xy,
      size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2
    }.into();

    argmax.insert_sdf_domain(
      domain,
      |pixel| circle.sdf(pixel)
    )?;
    argmax.convolute_domain([domain.tl.y, domain.br.y]);

    circles.push(circle)
  };

  Ok(circles)
}

/// only for testing, O(n^2)
pub fn intersection_test(circles: &Vec<Circle>) -> bool {
  for a in circles {
    for b in circles {
      if (a.xy - b.xy).length() + 1e-6 < a.r + b.r && a != b {
        println!("there are collisions:");
        println!("{:?}\n{:?}", a, b);
        println!("dist = {}, a.r: {}, b.r: {}", (a.xy - b.xy).length(), a.r, b.r);
        return true;
      }
    }
  }
  println!("there are no collisions");
  false
}