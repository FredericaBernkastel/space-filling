use {
  super::{
    GradientDescent,
    LineSearchConfig,
    LineSearch
  },
  crate::{
    geometry::WorldSpace,
    solver::z_order_storage::ZOrderStorage,
    error::Result,
    sdf
  },
  euclid::{Point2D, Size2D, Rect, Box2D},
  num_traits::{Float, Signed}
};

impl <P> GradientDescent<ZOrderStorage<Vec<P>>, P>
  where P: Float + Send + Sync {
  pub fn new(line_config: LineSearchConfig<P>, resolution: u64, chunk_size: u64) -> Result<Self> {
    let storage = ZOrderStorage::new(
      resolution, chunk_size, P::max_value() / (P::one() + P::one())
    )?;
    Ok(Self {
      dist_field: storage,
      line_config,
    })
  }

  pub fn insert_sdf(&mut self, sdf: impl Fn(Point2D<P, WorldSpace>) -> P + Sync + Send) {
    self.insert_sdf_domain(
      Rect::new(
        Point2D::splat(P::zero()),
        Size2D::splat(P::one()),
      ),
      sdf
    );
  }

  pub fn insert_sdf_domain(&mut self, domain: Rect<P, WorldSpace>, sdf: impl Fn(Point2D<P, WorldSpace>) -> P + Sync + Send) {
    use rayon::prelude::*;

    self.dist_field.chunks_domain_par_iter(domain)
      .for_each(move |chunk_xy| {
        let chunk = self.dist_field.get_chunk_xy(chunk_xy);
        chunk.pixels_mut().for_each(|(xy_normalized, value)| {
          *value = (*value).min(sdf(xy_normalized));
        })
      });
  }
}

impl <P> LineSearch<P> for GradientDescent<ZOrderStorage<Vec<P>>, P>
  where P: Float + Signed
{
  fn config(&self) -> LineSearchConfig<P> { self.line_config }

  fn sample_sdf(&self, pixel: Point2D<P, WorldSpace>) -> P {
    // check whether pixel is out of bounds
    match Box2D::from_size(Size2D::new(P::one(), P::one()))
      .contains(pixel) {
      true => {
        let pixel = (pixel.to_f64() * self.dist_field.resolution as f64)
          .cast_unit()
          .cast::<u64>();
        self.dist_field.pixel(pixel)
      },
      false => sdf::boundary_rect(pixel)
    }
  }
}