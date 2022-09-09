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

impl <'a, P> GradientDescent<&'a dyn Fn(Point2D<P, WorldSpace>) -> P, P> {
  pub fn new(line_config: LineSearchConfig<P>, sdf: &'a dyn Fn(Point2D<P, WorldSpace>) -> P) -> Self {
    Self {
      dist_field: sdf,
      line_config,
    }}
}

impl <P> LineSearch<P> for GradientDescent<&dyn Fn(Point2D<P, WorldSpace>) -> P, P>
  where P: Float + Signed{
  fn config(&self) -> LineSearchConfig<P> { self.line_config }
  fn sample_sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    (self.dist_field)(pixel)
  }}