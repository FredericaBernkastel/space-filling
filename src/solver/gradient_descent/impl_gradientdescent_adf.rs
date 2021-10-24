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
use std::borrow::{BorrowMut, Borrow};

impl<T> GradientDescent<T, f64> where T: BorrowMut<ADF> {
  pub fn new(line_config: LineSearchConfig<f64>, adf: T) -> Self {
    Self {
      dist_field: adf,
      line_config,
    }
  }
  pub fn insert_sdf(&mut self, sdf: impl Fn(Point2D<f64, WorldSpace>) -> f64 + 'static) -> bool {
    let domain = Rect::new(
      Point2D::splat(0.0),
      Size2D::splat(1.0),
    );
    self.insert_sdf_domain(domain, sdf)
  }
  pub fn insert_sdf_domain(&mut self, domain: Rect<f64, WorldSpace>, sdf: impl Fn(Point2D<f64, WorldSpace>) -> f64 + 'static) -> bool {
    let ret = self.dist_field.borrow_mut().insert_sdf_domain(domain, Rc::new(sdf));
    //self.dist_field.borrow_mut().prune(domain);
    ret
  }
}

impl<T> LineSearch<f64> for GradientDescent<T, f64>
  where T: Borrow<ADF>
{
  fn config(&self) -> LineSearchConfig<f64> { self.line_config }
  fn sample_sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    self.dist_field.borrow().sdf(pixel)
  }
}