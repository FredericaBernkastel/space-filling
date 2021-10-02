use {
  super::{
    GradientDescent,
    LineSearchConfig,
    LineSearch
  },
  crate::{geometry::WorldSpace},
  euclid::Point2D,
  num_traits::{Float, Signed},
};

impl <T, P> GradientDescent<Vec<T>, P> {
  pub fn new(line_config: LineSearchConfig<P>) -> Self {
    Self {
      dist_field: vec![],
      line_config,
    }}

  pub fn insert_sdf(&mut self, sdf: T) {
    self.dist_field.push(sdf);
  }
}

impl <P> LineSearch<P> for GradientDescent<Vec<Box<dyn Fn(Point2D<P, WorldSpace>) -> P + Send + Sync>>, P>
  where P: Float + Signed + Send + Sync {
  fn config(&self) -> LineSearchConfig<P> { self.line_config }
  fn sample_sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    use rayon::prelude::*;

    self.dist_field.par_iter()
      .map(move |s| s(pixel))
      .reduce(
        || P::max_value() / (P::one() + P::one()),
        |a, b| a.min(b)
      )
  }}