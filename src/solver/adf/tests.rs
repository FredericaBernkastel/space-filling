use {
  super::*,
  crate::{
    error::Result,
    geometry::{Circle, Shape},
    drawing,
    sdf,
    solver::{
      gradient_descent::{GradientDescent, LineSearch, LineSearchConfig},
      argmax2d::Argmax2D
    }
  },
  image::{Rgba, RgbaImage},
  euclid::Vector2D
};

impl ADF {
  pub fn print_stats_adf(&self) -> &Self {
    use humansize::{FileSize, file_size_opts as options};

    let mut total_nodes = 0u64;
    let mut total_size = 0usize;
    let mut max_depth = 0u8;
    self.traverse(&mut |node| {
      total_nodes += 1;
      total_size += std::mem::size_of::<Self>()
        + node.data.capacity() * std::mem::size_of::<Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>>();
      max_depth = (max_depth).max(node.depth);
      Ok(())
    }).ok();
    println!(
      "total nodes: {}\n\
      max subdivisions: {}\n\
      mem::size_of::<Quadtree<T>(): {}",
      total_nodes,
      max_depth,
      total_size.file_size(options::BINARY).unwrap()
    );
    self
  }
}

#[test] #[ignore] fn draw_layout() -> Result<()> {
  let mut image = RgbaImage::new(512, 512);
  let mut adf = ADF::new(8, vec![Rc::new(|_| f64::MAX / 2.0)]);
  let domain = Rect::from_size(Size2D::splat(1.0));

  let t0 = std::time::Instant::now();
  adf.insert_sdf_domain(domain, Rc::new(|p| Circle
    .scale(0.25)
    .translate(Vector2D::splat(0.5))
    .sdf(p)
  ));
  adf.insert_sdf_domain(domain, Rc::new(|p| Circle
    .scale(0.125)
    .translate(Vector2D::splat(0.125))
    .sdf(p)
  ));
  println!("{}us", t0.elapsed().as_micros());

  drawing::display_sdf(|p| adf.sdf(p), &mut image, 4.0);
  adf.draw_layout(&mut image);
  image.save("test/test_adf.png")?;
  Ok(())
}

#[test] #[ignore] fn gradient_adf() -> Result<()> {
  use rand::prelude::*;

  let config = LineSearchConfig {
    Δ: (-16f64).exp2(),
    decay_factor: 0.85,
    step_limit: Some(256),
    ..Default::default()
  };
  let mut image = RgbaImage::new(512, 512);
  let mut representation = ADF::new(11, vec![Rc::new(sdf::boundary_rect)]);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
  let mut circles = vec![];

  let t0 = std::time::Instant::now();
  GradientDescent::<&mut ADF, _>::new(config, &mut representation).iter().build()
    .filter_map(|(local_max, grad)| {
      let circle = {
        use std::f64::consts::PI;

        let angle = rng.gen_range::<f64, _>(-PI..=PI);
        let r = (rng.gen_range::<f64, _>(config.Δ..1.0).powf(1.0) * local_max.distance)
          .min(1.0 / 6.0);
        let delta = local_max.distance - r;
        let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

        Circle.translate(local_max.point - offset)
          .scale(r)
      };
      grad.insert_sdf_domain(
        Argmax2D::domain_empirical(local_max.point, local_max.distance),
        move |p| circle.sdf(p)
      ).then(|| circle)
    })
    .enumerate()
    .take(10)
    .for_each(|(i, c)| {
      if i % 1000 == 0 { println!("#{}", i); };
      circles.push(c);
    });

  println!("profile: {}ms", t0.elapsed().as_millis());
  representation.print_stats_adf();
  drawing::display_sdf(|p| representation.sdf(p), &mut image, 3.5);
  representation.draw_layout(&mut image);
  /*use {image::Pixel, drawing::Draw};
  circles.into_iter()
    .for_each(|c| c.texture(image::Luma([255]).to_rgba())
    .draw(&mut image));*/
  image.save("test/test_adf.png")?;
  Ok(())
}

#[test] #[ignore] fn animation() -> Result<()> {
  use rand::prelude::*;
  use drawing::Draw;

  let config = LineSearchConfig {
    Δ: (-16f64).exp2(),
    ..Default::default()
  };
  let mut representation = ADF::new(11, vec![Rc::new(sdf::boundary_rect)]);
  let mut circles = vec![];
  let mut rng = rand_pcg::Pcg64::seed_from_u64(2);


  let mut i = 0;
  'main: while i < 32 {
    let mut grad = GradientDescent::<&mut ADF, _>::new(config, &mut representation);
    let mut local_max = None;
    for _ in 0..config.max_attempts {
      local_max = grad.find_local_max(&mut rng);
      if local_max.is_some() { break; }
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
      let r = (rng.gen_range::<f64, _>(config.Δ..1.0).powf(1.0) * local_max.distance)
        .min(1.0 / 6.0);
      let delta = local_max.distance - r;
      let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

      Circle.translate(local_max.point - offset)
        .scale(r)
    };
    let domain = Argmax2D::domain_empirical(local_max.point, local_max.distance);

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
    representation.draw_bounding(domain, &mut image);
    image.save(format!("test/anim/#{}_4.png", i))?;

    representation.insert_sdf_domain(
      domain,
      Rc::new(move |p| circle.sdf(p))
    ).then(|| {
      circles.push(circle);
      i += 1;
    });
  };

  representation.print_stats_adf();

  Ok(())
}