use {
  crate::{
    solver::quadtree::{
      Quadtree, TraverseCommand
    },
    geometry::WorldSpace,
    sdf::SDF
  },
  std::rc::Rc,
  euclid::{Point2D, Rect}
};

pub type ADF = Quadtree<Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>>;

impl ADF {
  pub fn insert_sdf(&mut self, sdf: Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>) {

    let error = |sdf_1: &dyn Fn(_) -> f64, sdf_2: &dyn Fn(_) -> f64, rect| {
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
    };

    self.traverse_managed(&mut |node| {
      if error(node.data.as_ref(), sdf.as_ref(), node.rect) {
        return TraverseCommand::Skip;
      }

      // TODO: node pruning

      if node.children.is_none() {
        let sdf_old = node.data.clone();
        let rect = node.rect;
        match node.subdivide(|rect_ch|
          if error(sdf.as_ref(), sdf_old.as_ref(), rect_ch)  {
            sdf.clone()
          } else {
            sdf_old.clone()
          })
          .as_deref_mut() {
          Some(children) => children.iter_mut()
            .for_each(|child| child
              .insert_sdf(sdf.clone())
            ),
          None if error(sdf.as_ref(), sdf_old.as_ref(), rect)
            => node.data = sdf.clone(),
          _ => ()
        }
        return TraverseCommand::Skip;
      }
      TraverseCommand::Ok
    });
  }
}

impl SDF<f64> for ADF {
  fn sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    match self.pt_to_node(pixel) {
      Some(node) => (node.data)(pixel),
      None => (self.data)(pixel),
    }}}

#[cfg(test)] mod tests {
  use {
    super::*,
    crate::{
      error::Result,
      geometry::{Circle, Shape},
      drawing,
      sdf,
      solver::gradient_descent::{GradientDescent, LineSearchConfig}
    },
    image::RgbaImage,
    euclid::Vector2D
  };

  #[test] #[ignore] fn draw_layout() -> Result<()> {
    let mut image = RgbaImage::new(512, 512);
    let mut adf = ADF::new(8, Rc::new(sdf::boundary_rect));

    let t0 = std::time::Instant::now();
    adf.insert_sdf(Rc::new(|p| Circle
      .scale(0.25)
      .translate(Vector2D::splat(0.5))
      .sdf(p)
    ));
    adf.insert_sdf(Rc::new(|p| Circle
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

    let config = LineSearchConfig::default();
    let mut image = RgbaImage::new(512, 512);
    let mut adf = ADF::new(8, Rc::new(sdf::boundary_rect));
    let mut grad = GradientDescent::<&mut ADF, _>::new(config, &mut adf);
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

    let t0 = std::time::Instant::now();
    grad.iter().build()
      .take(2)
      .for_each(|(local_max, grad)| {
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
        grad.insert_sdf(move |p| circle.sdf(p));
      });
    println!("profile: {}ms", t0.elapsed().as_millis());
    adf.print_stats();
    drawing::display_sdf(|p| adf.sdf(p), &mut image, 4.0);
    adf.draw_layout(&mut image);
    image.save("test/test_adf.png")?;
    Ok(())
  }
}