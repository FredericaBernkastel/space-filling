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
  euclid::Point2D,
};

impl<'a> GradientDescent<&'a mut ADF, f64> {
  pub fn new(line_config: LineSearchConfig<f64>, adf: &'a mut ADF) -> Self {
    Self {
      dist_field: adf,
      line_config,
    }
  }
  pub fn insert_sdf(&mut self, sdf: impl Fn(Point2D<f64, WorldSpace>) -> f64 + 'static) {
    self.dist_field.insert_sdf(Rc::new(sdf));
  }
}

impl<'a> LineSearch<f64> for GradientDescent<&'a mut ADF, f64> {
  fn config(&self) -> LineSearchConfig<f64> { self.line_config }
  fn sample_sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    self.dist_field.sdf(pixel)
  }
}