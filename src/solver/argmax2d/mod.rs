use {
  crate::{
    geometry::{DistPoint, PixelSpace, WorldSpace}
  },
  z_order_storage::ZOrderStorage,
  anyhow::Result,
  euclid::{Rect, Point2D, Size2D},
};

pub mod z_order_storage;

pub struct Argmax2D {
  pub (crate) dist_map: ZOrderStorage<Vec<f32>>,
  chunk_argmax: Vec<DistPoint<f32, f32, WorldSpace>>
}

impl Argmax2D {
  pub fn new(resolution: u64, chunk_size: u64) -> Result<Self> {
    let storage = ZOrderStorage::new(resolution, chunk_size, f32::MAX / 2.0)?;
    let chunk_count = storage.chunk_count() as usize;
    Ok(Self {
      dist_map: storage,
      chunk_argmax: vec![DistPoint::default(); chunk_count]
    })
  }

  pub fn resolution(&self) -> u64 {
    self.dist_map.resolution
  }

  #[inline]
  fn write_cache(&self, id: u64, dist: DistPoint<f32, f32, WorldSpace>) {
    let ptr = &self.chunk_argmax[id as usize] as *const _ as usize;
    unsafe { *(ptr as *const DistPoint<f32, f32, WorldSpace> as *mut _) = dist }
  }

  pub fn find_max(&self) -> DistPoint<f32, f32, WorldSpace> {
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
          DistPoint {
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
        DistPoint {
          distance: *value,
          point: xy_normalized
        }
      }).max()
        .unwrap();
      self.write_cache(chunk.id, max_dist);
    });
  }

  pub fn pixels(&self) -> impl Iterator<Item = DistPoint<f32, u64, PixelSpace>> + '_ {
    self.dist_map.pixels()
  }
}
