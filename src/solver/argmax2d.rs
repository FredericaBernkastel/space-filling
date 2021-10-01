use {
  crate::{
    error::Result,
    geometry::{PixelSpace, WorldSpace}
  },
  super::{
    DistPoint,
    z_order_storage::ZOrderStorage
  },
  euclid::{Rect, Point2D, Vector2D as V2, Size2D},
  num_traits::{Float, FloatConst}
};

pub type ArgmaxResult = DistPoint<f32, f32, WorldSpace>;

pub struct Argmax2D {
  pub (crate) dist_map: ZOrderStorage<Vec<f32>>,
  chunk_argmax: Vec<ArgmaxResult>
}

impl Argmax2D {
  pub fn new(resolution: u64, chunk_size: u64) -> Result<Self> {
    let storage = ZOrderStorage::new(resolution, chunk_size, f32::MAX / 2.0)?;
    let chunk_count = storage.chunk_count() as usize;
    Ok(Self {
      dist_map: storage,
      chunk_argmax: vec![ArgmaxResult::default(); chunk_count]
    })
  }

  pub fn resolution(&self) -> u64 {
    self.dist_map.resolution
  }

  #[inline]
  fn write_cache(&self, id: u64, dist: ArgmaxResult) {
    unsafe { *(&self.chunk_argmax[id as usize] as *const _ as *mut _) = dist }
  }

  pub fn find_max(&self) -> ArgmaxResult {
    *self.chunk_argmax.iter()
      .max()
      .unwrap()
  }

  pub fn insert_sdf(&mut self, sdf: impl Fn(Point2D<f32, WorldSpace>) -> f32 + Sync + Send) {
    self.insert_sdf_domain(
      Rect::new(
        Point2D::splat(0.0),
        Size2D::splat(1.0),
      ),
      sdf
    );
  }

  pub fn insert_sdf_domain(&mut self, domain: Rect<f32, WorldSpace>, sdf: impl Fn(Point2D<f32, WorldSpace>) -> f32 + Sync + Send) {
    use rayon::prelude::*;

    self.dist_map.chunks_domain_par_iter(domain)
      .for_each(move |chunk_xy| {
        let chunk = self.dist_map.get_chunk_xy(chunk_xy);
        let max_dist = chunk.pixels_mut().map(|(xy_normalized, value)| {
          *value = (*value).min(sdf(xy_normalized));
          ArgmaxResult {
            distance: *value,
            point: xy_normalized
          }
        }).max()
          .unwrap();
        self.write_cache(chunk.id, max_dist);
      });
  }

  /// Invert distance field.
  pub fn invert(&mut self) {
    use rayon::prelude::*;

    self.dist_map.chunks_par_iter().for_each(|chunk| {
      let max_dist = chunk.pixels_mut().map(|(xy_normalized, value)| {
        *value = -*value;
        ArgmaxResult {
          distance: *value,
          point: xy_normalized
        }
      }).max()
        .unwrap();
      self.write_cache(chunk.id, max_dist);
    });
  }

  pub fn domain_empirical<P: Float + FloatConst>(center: Point2D<P, WorldSpace>, max_dist: P) -> Rect<P, WorldSpace> {
    let size = max_dist * P::from(4.0).unwrap() * P::SQRT_2();
    Rect {
      origin: (center.to_vector() - V2::splat(size) / (P::one() + P::one())).to_point(),
      size: Size2D::splat(size)
    }
  }

  pub fn iter(&mut self) -> ArgmaxIter {
    let min_dist = 0.5 * std::f32::consts::SQRT_2 / self.dist_map.resolution as f32;
    ArgmaxIter {
      argmax: self,
      min_dist
    }
  }

  pub fn pixels(&self) -> impl Iterator<Item = DistPoint<f32, u64, PixelSpace>> + '_ {
    self.dist_map.pixels()
  }
}

pub struct ArgmaxIter<'a> {
  argmax: &'a mut Argmax2D,
  min_dist: f32
}

impl<'a> ArgmaxIter<'a> {
  pub fn min_dist(mut self, value: f32) -> Self {
    self.min_dist = value;
    self
  }

  pub fn min_dist_px(mut self, value: f32) -> Self {
    self.min_dist = value / self.argmax.dist_map.resolution as f32;
    self
  }

  pub fn build(self) -> impl Iterator<Item = (ArgmaxResult, &'a mut Argmax2D)> {
    let min_dist = self.min_dist;
    (0..).map(move |_| {
      // this is awkward...
      let argmax = unsafe { &mut *(self.argmax as *const _ as *mut Argmax2D) };
      (argmax.find_max(), argmax)
    })
      .take_while(move |(dist, _)| dist.distance >= min_dist)
  }
}