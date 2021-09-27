use crate::{
  error::Result,
  geometry::{Point, Circle, Rect, TLBR},
  sdf::SDF,
  legacy::{self, argmax::Argmax},
  drawing,
  profile
};

/// generate a solver of circles, then display it
pub fn simple() -> Result<()> {
  let circles = profile!("argmax", {
    legacy::tests::sdf_convolution_domain_test()?
  });
  profile!("draw", {
    drawing::draw_circles("out.png", circles.into_iter(), (2048, 2048).into())?;
  });
  open::that("out.png")?;
  Ok(())
}

/// A regular solver embedded in a random one
/// 174s, 100'000 circrles, Δ = 2^-14
pub fn embedded() -> Result<Vec<Circle>> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(1);
  let mut circles: Vec<Circle> = vec![];
  let mut argmax = Argmax::new(Point { x: 16384, y: 16384 });

  // insert boundary rect SDF
  {
    let boundary = Rect { center: Point { x: 1.0 / 2.0, y: 1.0 / 2.0 }, size: 1.0 };
    argmax.insert_sdf(|pixel| -boundary.sdf(pixel))?;
  }
  argmax.convolute_domain([0.0, 1.0]);

  'argmax_outer: for i in 0..100000 {
    let min_distance = 3.0 * std::f32::consts::SQRT_2 / argmax.dist_map.width() as f32;

    let argmax_ret = argmax.find_max_convolution();
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
  };

  argmax.invert();
  argmax.convolute_domain([0.0, 1.0]);

  //embedded
  'argmax_outer2: for i in 0..100000 {
    let min_distance = 1.0 * std::f32::consts::SQRT_2 / argmax.dist_map.width() as f32;

    let argmax_ret = argmax.find_max_convolution();

    if argmax_ret.distance < min_distance {
      println!("#{}: reached minimum, breaking: {:?}", i, argmax_ret);
      break 'argmax_outer2;
    }

    let circle = Circle {
      xy: { argmax_ret.point.into(): Point<f32> }, r: (argmax_ret.distance / 3.0)
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

/// generate a solver, and use it to display a dataset, up to 100'000 images
pub fn image_dataset() -> Result<()> {
  use regex::Regex;

  let image_folder = "d:/picture/hydrus/dolls";

  std::thread::Builder::new().spawn(move ||{
    let circles = profile!("argmax", embedded()?);
    let files = crate::find_files(
      image_folder,
      |file| {
        Regex::new("^.+\\.(jpg|png)$")
          .unwrap()
          .is_match(file)
      }
    );
    profile!("draw", {
      drawing::draw_img_parallel(
        "out.png",
        circles.into_iter(),
        files,
        (16384, 16384).into(),
        4
      )?;
    });
    open::that("out.png")?;
    Ok(())
  })?
  .join()
  .unwrap()
}