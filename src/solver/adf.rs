use {
  crate::{
    solver::quadtree::{
      Quadtree, TraverseCommand
    },
    geometry::WorldSpace,
    sdf::SDF
  },
  std::rc::Rc,
  euclid::{Point2D, Size2D, Rect}
};

pub type ADF = Quadtree<Vec<Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>>>;

impl SDF<f64> for &[Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>] {
  fn sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    self.into_iter()
      .map(|f| f(pixel))
      .reduce(|a, b| if a <= b { a } else { b })
      .unwrap_or(f64::MAX / 2.0)
  }
}

impl ADF {
  pub fn prune(&mut self, domain: Rect<f64, WorldSpace>) {
    if self.rect.intersects(&domain) && self.children.is_some() {
      for child in self.children.as_deref_mut().unwrap() {
        child.prune(domain);
      }

      let children = self.children.as_deref_mut().unwrap();
      let data = self.data.as_slice();
      let rect = self.rect;

      if children.iter()
        // only leafs
        .all(|child| child.children.is_none())
      && children.iter()
        // equality of SDF functions
          .all(|child|
            Self::error(&child.sdf_vec(), &|p| data.sdf(p), rect)) {
        self.data = children[0].data.clone();
        self.children = None
      }
    }
  }

  fn error(
    sdf_1: &dyn Fn(Point2D<f64, WorldSpace>) -> f64,
    sdf_2: &dyn Fn(Point2D<f64, WorldSpace>) -> f64,
    rect: Rect<f64, WorldSpace>
  ) -> bool {
    let epsilon = 1e-6;

    let control_points = |rect: Rect<_, _>| {
      let p = [0.0, 0.25, 0.5, 0.75, 1.0];
      itertools::iproduct!(p, p).map(move |p| {
        rect.origin + rect.size.to_vector().component_mul(p.into())
      })
    };

    control_points(rect)
      .map(|point| {
        let ground_truth = sdf_1(point).min(sdf_2(point));
        (ground_truth - sdf_1(point)).abs()
      })
      .sum::<f64>() < epsilon
  }

  fn sdf_vec(&self) -> impl Fn(Point2D<f64, WorldSpace>) -> f64 + '_ {
    move |p| self.data.as_slice().sdf(p)
  }

  pub fn insert_sdf_domain(&mut self, domain: Rect<f64, WorldSpace>, sdf: Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>) -> bool {
    let mut change_exists = false;

    self.traverse_managed(&mut |node| {
      // no intersection with domain
      if !node.rect.intersects(&domain) {
        return TraverseCommand::Skip;
      }

      // no refinement is required
      if Self::error(&node.sdf_vec(), sdf.as_ref(), node.rect) {
        return TraverseCommand::Skip;
      }

      // new minimum
      if Self::error(sdf.as_ref(), &node.sdf_vec(), node.rect) {
        node.data = vec![sdf.clone()];
        change_exists = true;
        return TraverseCommand::Ok;
      };

      if node.children.is_none() {
        change_exists = true;
        let data = node.data.clone();
        let sdf_old = |p| data.as_slice().sdf(p);
        match node.subdivide(|rect_ch|
          if Self::error(sdf.as_ref(), &sdf_old, rect_ch)  {
            vec![sdf.clone()]
          } else {
            data.clone()
          })
          .as_deref_mut() {
          Some(children) => children.iter_mut()
            .for_each(|child| {
              child.insert_sdf_domain(domain, sdf.clone());
            }),
          // max subdivisions
          None => node.data.push(sdf.clone())
        }
        return TraverseCommand::Skip;
      }
      TraverseCommand::Ok
    });

    change_exists
  }
}

impl SDF<f64> for ADF {
  fn sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    match self.pt_to_node(pixel) {
      Some(node) => node.sdf_vec()(pixel),
      None => self.sdf_vec()(pixel),
    }}}

#[cfg(test)] mod tests {
  use {
    super::*,
    crate::{
      error::Result,
      geometry::{Circle, Shape},
      drawing,
      sdf,
      solver::{
        gradient_descent::{GradientDescent, LineSearchConfig},
        argmax2d::Argmax2D
      }
    },
    image::RgbaImage,
    euclid::Vector2D
  };

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
    adf.prune(domain);
    adf.insert_sdf_domain(domain, Rc::new(|p| Circle
      .scale(0.125)
      .translate(Vector2D::splat(0.125))
      .sdf(p)
    ));
    adf.prune(domain);
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
      ..Default::default()
    };
    let mut image = RgbaImage::new(2048, 2048);
    let mut adf = ADF::new(11, vec![Rc::new(sdf::boundary_rect)]);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);
    let mut circles = vec![];

    let t0 = std::time::Instant::now();
    GradientDescent::<&mut ADF, _>::new(config, &mut adf).iter().build()
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
      .take(1000000)
      .for_each(|(i, c)| {
        if i % 1000 == 0 { println!("#{}", i); };
        circles.push(c);
      });

    println!("profile: {}ms", t0.elapsed().as_millis());
    adf.print_stats();
    //drawing::display_sdf(|p| adf.sdf(p), &mut image, 3.5);
    //adf.draw_layout(&mut image);
    use {image::Pixel, drawing::Draw};
    circles.into_iter()
      .for_each(|c| c.texture(image::Luma([255]).to_rgba())
      .draw(&mut image));
    image.save("test/test_adf.png")?;
    Ok(())
  }
}