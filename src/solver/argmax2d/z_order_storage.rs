use {
  super::PixelPoint,
  crate::geometry::{DistPoint, Aabb, P2},
  rayon::iter::ParallelIterator,
  anyhow::{Result, bail},
};
use num_traits::Float;

pub struct ZOrderStorage<T> {
  data: T,
  pub resolution: u64,
  pub chunk_size: u64,
}

impl <T> ZOrderStorage<T> {
  pub fn chunk_count(&self) -> u64 {
    (self.resolution / self.chunk_size).pow(2)
  }

  /// Chunk coordinates covering `domain ∩ [0, 1]²`, rounded outwards.
  pub fn chunks_domain_par_iter<P>(&self, domain: Aabb<P, 2>)
    -> impl ParallelIterator<Item = PixelPoint>
    where P: Float + nalgebra::Scalar {
    use rayon::prelude::*;

    let chunks = self.resolution as f64 / self.chunk_size as f64;
    let lo = |v: P| (v.to_f64().unwrap().clamp(0.0, 1.0) * chunks).floor() as u64;
    let hi = |v: P| (v.to_f64().unwrap().clamp(0.0, 1.0) * chunks).ceil() as u64;
    let (min_x, max_x) = (lo(domain.min.x), hi(domain.max.x));
    let (min_y, max_y) = (lo(domain.min.y), hi(domain.max.y));

    (min_y..max_y)
      .into_par_iter()
      .flat_map(move |chunk_y|
        (min_x..max_x)
          .into_par_iter().map(move |chunk_x| PixelPoint::new(chunk_x, chunk_y))
      )
  }
}

impl <T: Clone> ZOrderStorage<Vec<T>> {
  pub fn new(resolution: u64, chunk_size: u64, default: T) -> Result<Self> {
    if resolution % chunk_size != 0 {
      bail!("distance map resolution is not divisible by the chunk resolution")
    };
    let chunk_area = resolution.pow(2);
    Ok(Self {
      data: vec![default; chunk_area as usize],
      resolution,
      chunk_size
    })
  }

  pub fn get_chunk(&self, id: u64) -> Chunk<'_, T> {
    let chunk_area = self.chunk_size.pow(2);
    Chunk {
      slice: &self.data[(chunk_area * id) as usize .. (chunk_area * (id + 1)) as usize],
      top_left: offset_to_xy(id, self.resolution / self.chunk_size) * self.chunk_size,
      id,
      size: self.chunk_size,
      global_size: self.resolution
    }
  }

  pub fn get_chunk_xy(&self, xy: PixelPoint) -> Chunk<'_, T> {
    self.get_chunk(xy_to_offset(xy, self.resolution / self.chunk_size))
  }

  pub fn chunks(&self) -> impl Iterator<Item = Chunk<'_, T>> {
    let chunk_count = (self.resolution / self.chunk_size).pow(2);
    (0..chunk_count).map(move |id| self.get_chunk(id))
  }

  pub fn pixel(&self, xy: PixelPoint) -> T {
    let chunk = self.get_chunk_xy(xy / self.chunk_size);
    let offset = PixelPoint::from(xy - chunk.top_left);
    let offset = xy_to_offset(offset, self.chunk_size) as usize;
    chunk.slice[offset].clone()
  }

  pub fn pixels(&self) -> impl Iterator<Item = DistPoint<T, u64, 2>> + '_ {
    self.chunks().flat_map(move |chunk| {
      chunk.slice.iter().enumerate().map(move |(i, pixel)|
        DistPoint {
          distance: pixel.clone(),
          point: offset_to_xy(i as u64, chunk.size) + chunk.top_left.coords
        }
      )
    })
  }
}

impl<T> ZOrderStorage<Vec<T>> where T: Clone + Send + Sync {
  pub fn chunks_par_iter(&self) -> impl ParallelIterator<Item = Chunk<'_, T>> {
    use rayon::prelude::*;

    let chunk_count = (self.resolution / self.chunk_size).pow(2);
    (0..chunk_count).into_par_iter()
      .map(move |id| self.get_chunk(id))
  }
}

pub struct Chunk<'a, T> {
  pub slice: &'a [T],
  pub top_left: PixelPoint,
  pub id: u64,
  pub size: u64,
  pub global_size: u64
}

impl<'a, T> Chunk<'a, T> {
  fn offset_to_xy_normalized<P: Float + nalgebra::Scalar>(&self, offset: u64) -> P2<P> {
    let xy = offset_to_xy(offset, self.size) + self.top_left.coords;
    let global = P::from(self.global_size).unwrap();
    P2::new(
      P::from(xy.x).unwrap() / global,
      P::from(xy.y).unwrap() / global,
    )
  }

  pub(crate) fn pixels_mut<P: Float + nalgebra::Scalar>(&self) -> impl Iterator<Item = (P2<P>, &mut T)> {
    unsafe { std::slice::from_raw_parts_mut(self.slice.as_ptr() as *mut T, self.slice.len()) }
      .iter_mut()
      .enumerate()
      .map(move |(i, value)| (
        self.offset_to_xy_normalized(i as u64),
        value
      ))
  }
}

fn offset_to_xy(offset: u64, width: u64) -> PixelPoint {
  PixelPoint::new(
    offset % width,
    offset / width,
  )
}

fn xy_to_offset(xy: PixelPoint, width: u64) -> u64 {
  xy.y * width + xy.x
}
