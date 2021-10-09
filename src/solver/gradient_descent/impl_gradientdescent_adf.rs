use {
  super::{
    GradientDescent,
    LineSearchConfig,
    LineSearch
  },
  crate::{
    geometry::WorldSpace,
    solver::adf::ADF,
    sdf::SDF
  },
  std::rc::Rc,
  euclid::{Point2D, Size2D, Rect},
};

impl<'a> GradientDescent<&'a mut ADF, f64> {
  pub fn new(line_config: LineSearchConfig<f64>, adf: &'a mut ADF) -> Self {
    Self {
      dist_field: adf,
      line_config,
    }
  }
  pub fn insert_sdf(&mut self, sdf: impl Fn(Point2D<f64, WorldSpace>) -> f64 + 'static) {
    let domain = Rect::new(
      Point2D::splat(0.0),
      Size2D::splat(1.0),
    );
    self.insert_sdf_domain(domain, sdf);
  }
  pub fn insert_sdf_domain(&mut self, domain: Rect<f64, WorldSpace>, sdf: impl Fn(Point2D<f64, WorldSpace>) -> f64 + 'static) {
    self.dist_field.insert_sdf_domain(domain, Rc::new(sdf));
    self.dist_field.prune(domain);
  }
}

impl<'a> LineSearch<f64> for GradientDescent<&'a mut ADF, f64> {
  fn config(&self) -> LineSearchConfig<f64> { self.line_config }
  fn sample_sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    self.dist_field.sdf(pixel)
  }
}