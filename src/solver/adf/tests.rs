use {
  super::*,
  crate::{
    geometry::{Circle, Shape, P2},
    drawing,
    sdf,
    solver::{
      adf::ADF, line_search::LineSearch
    },
    util
  },
  anyhow::Result,
  image::{Rgba, RgbaImage},
  euclid::{Vector2D, Size2D},
  std::cell::Cell
};
use crate::geometry::DistPoint;

#[test] fn draw_layout() -> Result<()> {
  let mut image = RgbaImage::new(512, 512);
  let mut adf = ADF::new(8, vec![Arc::new(|_| f64::MAX / 2.0)]);
  let domain = Rect::from_size(Size2D::splat(1.0));

  let t0 = std::time::Instant::now();
  adf.insert_sdf_domain(domain, Arc::new(|p| Circle
    .scale(0.25)
    .translate(Vector2D::splat(0.5))
    .sdf(p)
  ));
  adf.insert_sdf_domain(domain, Arc::new(|p| Circle
    .scale(0.125)
    .translate(Vector2D::splat(0.125))
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
  let representation = ADF::<f64>::new(7, vec![Arc::new(sdf::boundary_rect)]);
  let mut primitives = vec![];
  let trials = Cell::new(0u64);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let t0 = std::time::Instant::now();

  util::local_maxima_iter(
    Box::new(|p| representation.sdf(p)),
    32, 0, LineSearch::default()
  ).inspect(|_| trials.set(trials.get() + 1))
    .filter_map(|local_max| {
      let circle = {
        use std::f64::consts::PI;

        let angle = rng.gen_range(-PI..=PI);
        let r = (rng.gen_range(1e-6..1.0).powf(5.0) * local_max.distance)
          .min(1.0 / 6.0);
        let delta = local_max.distance - r;
        // polar to cartesian
        let offset = P2::from([angle.cos(), angle.sin()]) * delta;

        Circle.translate(local_max.point - offset)
          .scale(r)
      };
      // alternately use safe RwLock<ADF> for 1.5x slowdown
      unsafe { representation.as_mut() }.insert_sdf_domain(
        util::domain_empirical(local_max),
        Arc::new(move |p| circle.sdf(p))
      ).then(|| circle)
    })
    .enumerate()
    .take(100000)
    .for_each(|(i, c)| {
      if i % 1000 == 0 { println!("#{}", i); };
      primitives.push(c);
    });

  println!("profile: {}ms", t0.elapsed().as_millis());
  // TODO: Fix sdf insertion method
  /* Here, `adf_error_margin` denotes failed attempts to instert a shape in ADF due to
     imperfect primitive elimination method. See `solver::adf::ADF::higher_all` for more details.
   */
  println!("adf_error_margin: {:+.3e}", trials.get() as f64 / primitives.len() as f64 - 1.0);
  println!("{representation:#?}");
  //drawing::display_sdf(|p| representation.sdf(p), &mut image, 3.5);
  //representation.draw_layout(&mut image);
  use {image::Pixel, drawing::Draw};
  /*primitives.into_iter()
    .for_each(|p| p.texture(image::Luma([255]).to_rgba())
    .draw(&mut image));*/
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

  let mut representation = ADF::new(11, vec![Arc::new(sdf::boundary_rect)]);
  let mut circles = vec![];
  let mut rng = rand_pcg::Pcg64::seed_from_u64(2);

  let mut i = 0;
  'main: while i < 32 {
    let mut local_max = None;
    for _ in 0..50 {
      let p0 = P2::new(
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
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
      Circle
        .translate(local_max.point.to_vector())
        .scale(local_max.distance)
        .texture(Rgba([0x45, 0x8F, 0xF5, 0x7F]))
        .draw(&mut image);
      image.save(format!("test/anim/#{}_1.png", i))?;
    }

    let circle = {
      use std::f64::consts::PI;

      let angle = rng.gen_range::<f64, _>(-PI..=PI);
      let r = (rng.gen_range::<f64, _>(0.0..1.0).powf(1.0) * local_max.distance)
        .min(1.0 / 6.0);
      let delta = local_max.distance - r;
      let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

      Circle.translate(local_max.point - offset)
        .scale(r)
    };
    let domain = util::domain_empirical(local_max);

    circle.texture(Rgba([0x45, 0x8F, 0xF5, 0xFF]))
      .draw(&mut image);

    image.save(format!("test/anim/#{}_2.png", i))?;
    {
      let mut image = image.clone();
      Circle
        .translate(local_max.point.to_vector())
        .scale(local_max.distance * 4.0)
        .texture(Rgba([0xFF, 0, 0, 0x7F]))
        .draw(&mut image);
      image.save(format!("test/anim/#{}_3.png", i))?;
    }
    representation.tree.draw_bounding(domain, &mut image);
    image.save(format!("test/anim/#{}_4.png", i))?;

    representation.insert_sdf_domain(
      domain,
      Arc::new(move |p| circle.sdf(p))
    ).then(|| {
      circles.push(circle);
      i += 1;
    });
  };

  println!("{representation:#?}");

  Ok(())
}