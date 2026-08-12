//! ADF tests that need the shape catalogue *and* the rasterizer.
//!
//! These live here rather than in `adaptive-distance-field` because that crate
//! has neither: shapes and drawing are this crate's contribution. What is left
//! there are the soundness proofs, which need only a ball expressed as a closure.

#![cfg(feature = "drawing")]

use {
  space_filling::{
    geometry::{Aabb, Combinator, DistPoint, Hypersphere, P2, V2},
    drawing::{self, AdfDraw, QuadtreeDraw, Shape},
    sdf::{self, SDF},
    solver::{ADF, LineSearch, Orthant, Primitive},
    util,
  },
  anyhow::Result,
  image::{Rgba, RgbaImage},
  std::{cell::Cell, sync::Arc},
};

#[test] fn draw_layout() -> Result<()> {
  let mut image = RgbaImage::new(512, 512);
  let mut adf = ADF::<f64, 2, Orthant>::new(8, vec![Primitive::new(|_| f64::MAX / 2.0)]);
  let domain = Aabb::unit();

  let t0 = std::time::Instant::now();
  adf.insert_sdf_domain(domain, Arc::new(|p| Hypersphere
    .scale(0.25)
    .translate(V2::repeat(0.5))
    .sdf(p)
  ));
  adf.insert_sdf_domain(domain, Arc::new(|p| Hypersphere
    .scale(0.125)
    .translate(V2::repeat(0.125))
    .sdf(p)
  ));
  println!("{}us", t0.elapsed().as_micros());

  drawing::display_sdf(|p| adf.sdf(p), &mut image, 4.0);
  adf.tree.draw_layout(&mut image);
  image.save("test/test_adf.png")?;
  Ok(())
}

// profile: 4.85s, 100k circles, adf_subdiv = 7
#[test] #[ignore] fn gradient_adf() -> Result<()> {
  use rand::prelude::*;

  let mut image = RgbaImage::new(1024, 1024);
  let representation = ADF::<f64, 2, Orthant>::new(7, vec![Primitive::new(sdf::boundary_rect)]);
  let mut primitives = vec![];
  let trials = Cell::new(0u64);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let t0 = std::time::Instant::now();

  util::local_maxima_iter(
    Box::new(|p: P2<f64>| representation.sdf(p)),
    32, 0, LineSearch::default()
  ).inspect(|_| trials.set(trials.get() + 1))
    .filter_map(|local_max| {
      let circle = {
        use std::f64::consts::PI;

        let angle = rng.random_range(-PI..=PI);
        let r = (rng.random_range(1e-6..1.0f64).powf(5.0) * local_max.distance)
          .min(1.0 / 6.0);
        let delta = local_max.distance - r;
        // polar to cartesian
        let offset = V2::new(angle.cos(), angle.sin()) * delta;

        Hypersphere.translate(local_max.point.coords - offset)
          .scale(r)
      };
      // alternately use safe RwLock<ADF> for 1.5x slowdown
      unsafe { representation.as_mut() }.insert_at_maximum(
        local_max,
        Primitive::from_shape(circle)
      ).then(|| circle)
    })
    .enumerate()
    .take(100000)
    .for_each(|(i, c)| {
      if i % 1000 == 0 { println!("#{}", i); };
      primitives.push(c);
    });

  println!("profile: {}ms", t0.elapsed().as_millis());
  // `adf_error_margin` is the fraction of located maxima whose insertion did not
  // lower the field (`insert_at_maximum` returned `false`) — e.g. a near-duplicate
  // maximum, or a candidate lost to the optimizer's numeric tolerance.
  println!("adf_error_margin: {:+.3e}", trials.get() as f64 / primitives.len() as f64 - 1.0);
  println!("{representation:#?}");
  use {image::Pixel, drawing::Draw};
  representation
    .texture(image::Luma([255]).to_rgba())
    .draw(&mut image);

  image.save("test/test_adf.png")?;
  Ok(())
}

#[test] #[ignore] fn animation() -> Result<()> {
  use rand::prelude::*;
  use drawing::Draw;

  std::fs::create_dir("test\\anim").ok();

  let mut representation = ADF::<f64, 2, Orthant>::new(11, vec![Primitive::new(sdf::boundary_rect)]);
  let mut circles = vec![];
  let mut rng = rand_pcg::Pcg64::seed_from_u64(2);

  let mut i = 0;
  'main: while i < 32 {
    let mut local_max = None;
    for _ in 0..50 {
      let p0 = P2::new(
        rng.random_range(0.0..1.0),
        rng.random_range(0.0..1.0),
      );
      let ret = LineSearch::default().optimize(|p| representation.sdf(p), p0);
      let ret = DistPoint { distance: representation.sdf(ret), point: ret};
      if ret.distance > 0.0 { local_max = Some(ret); break; }
    };
    let local_max = match local_max {
      Some(r) => r,
      None => {
        println!("failed to find local max, breaking");
        break 'main;
      }
    };

    let mut image = RgbaImage::new(512, 512);
    representation
      .display_sdf(&mut image, 3.5)
      .draw_bucket_weights(&mut image)
      .tree
      .draw_layout(&mut image);
    image.save(format!("test/anim/#{}_0.png", i))?;


    {
      let mut image = image.clone();
      Hypersphere
        .translate(local_max.point.coords)
        .scale(local_max.distance)
        .texture(Rgba([0x45, 0x8F, 0xF5, 0x7F]))
        .draw(&mut image);
      image.save(format!("test/anim/#{}_1.png", i))?;
    }

    let circle = {
      use std::f64::consts::PI;

      let angle = rng.random_range::<f64, _>(-PI..=PI);
      let r = (rng.random_range::<f64, _>(0.0..1.0).powf(1.0) * local_max.distance)
        .min(1.0 / 6.0);
      let delta = local_max.distance - r;
      let offset = V2::new(angle.cos(), angle.sin()) * delta;

      Hypersphere.translate(local_max.point.coords - offset)
        .scale(r)
    };
    let domain = representation.update_domain(local_max);

    circle.texture(Rgba([0x45, 0x8F, 0xF5, 0xFF]))
      .draw(&mut image);

    image.save(format!("test/anim/#{}_2.png", i))?;
    {
      let mut image = image.clone();
      Hypersphere
        .translate(local_max.point.coords)
        .scale(local_max.distance * 4.0)
        .texture(Rgba([0xFF, 0, 0, 0x7F]))
        .draw(&mut image);
      image.save(format!("test/anim/#{}_3.png", i))?;
    }
    representation.tree.draw_bounding(domain, &mut image);
    image.save(format!("test/anim/#{}_4.png", i))?;

    representation.insert_at_maximum(
      local_max,
      Primitive::from_shape(circle)
    ).then(|| {
      circles.push(circle);
      i += 1;
    });
  };

  println!("{representation:#?}");

  Ok(())
}
