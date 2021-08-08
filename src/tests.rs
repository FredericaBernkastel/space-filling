use lib::{
  geometry::{Circle, Point, Rect, TLBR},
  error::Result,
  sdf::SDF,
  argmax2d::Argmax2D
};

/// 104ms, 1000 circrles, Δ = 2^-10
pub fn sdf_argmax2d_test() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax2D::new(1024, 16)?;

  // insert boundary rect SDF
  {
    let boundary = Rect { center: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 }, size: 1.0 };
    argmax.insert_sdf(|pixel| -boundary.sdf(pixel))?;
  }

  'argmax_outer: for i in 0..1000 {
    let min_distance = 0.5 * std::f32::consts::SQRT_2 / argmax.resolution as f32;

    let argmax_ret = argmax.find_max();

    //lib::drawing::display_argmax_debug(&argmax)
    //  .save(format!("tmp/{}.png", i))?;

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

    let domain: TLBR<f32> = Rect {
      center: circle.xy,
      size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2
    }.into();

    argmax.insert_sdf_domain(
      domain,
      |pixel| circle.sdf(pixel)
    )?;

    circles.push(circle)
  };

  Ok(circles)
}

/// A regular distribution embedded in a random one
/// 88.4s, 100'000 circrles, Δ = 2^-14
pub fn embedded() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax2D::new(16384, 64)?;

  // insert boundary rect SDF
  {
    let boundary = Rect { center: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 }, size: 1.0 };
    argmax.insert_sdf(|pixel| -boundary.sdf(pixel))?;
  }

  'argmax_outer: for i in 0..100000 {
    let min_distance = 3.0 * std::f32::consts::SQRT_2 / argmax.resolution as f32;

    let argmax_ret = argmax.find_max();
    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax_outer;
    }

    let circle = {
      use std::f32::consts::PI;

      let angle = rng.gen_range::<f32, _>(-PI..=PI);
      let r = (rng.gen_range::<f32, _>(min_distance..1.0).powf(1.0) * argmax_ret.distance)
        //let r = (argmax_ret.distance / 4.0)
        .min(1.0 / 4.0);
      let delta = argmax_ret.distance - r;
      let offset = Point { x: delta * angle.cos(), y: delta * angle.sin() };

      Circle {
        xy: { argmax_ret.point.into(): Point<f32> }.translate(offset), r
      }
    };

    if i % 1000 == 0 {
      println!("#{} argmax = {}", i, argmax_ret.distance * argmax.resolution as f32);
    }

    let domain: TLBR<f32> = Rect {
      center: circle.xy,
      size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2
    }.into();
    argmax.insert_sdf_domain(
      domain,
      |pixel| circle.sdf(pixel)
    )?;
  };

  argmax.invert();

  //embedded
  'argmax_outer2: for i in 0..100000 {
    let min_distance = 1.0 * std::f32::consts::SQRT_2 / argmax.resolution as f32;

    let argmax_ret = argmax.find_max();

    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax_outer2;
    }

    let circle = Circle {
      xy: { argmax_ret.point.into(): Point<f32> }, r: (argmax_ret.distance / 3.0)
    };

    if i % 1000 == 0 {
      println!("#{} argmax = {}", i, argmax_ret.distance * argmax.resolution as f32);
    }

    let domain: TLBR<f32> = Rect {
      center: circle.xy,
      size: argmax_ret.distance * 4.0 * std::f32::consts::SQRT_2
    }.into();
    argmax.insert_sdf_domain(
      domain,
      |pixel| circle.sdf(pixel)
    )?;

    circles.push(circle)
  };

  Ok(circles)
}
