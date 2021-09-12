use crate::{
  error::Result,
  geometry::{WorldSpace, PixelSpace}
};
use euclid::{Rect, Box2D, Point2D, Vector2D as V2, Size2D};
use error_chain::bail;

pub struct Argmax2D {
  dist_map: Vec<f32>,
  pub resolution: u64,
  pub chunk_size: u64,
  chunk_argmax: Vec<ArgmaxResult<f32, WorldSpace>>
}

pub struct Chunk<'a> {
  pub slice: &'a [f32],
  pub argmax_ref: &'a ArgmaxResult<f32, WorldSpace>,
  pub top_left: Point2D<u64, PixelSpace>,
  pub id: u64,
  size: u64,
  global_size: u64
}

impl<'a> Chunk<'a> {
  unsafe fn slice_mut(&self) -> &mut [f32] {
    std::slice::from_raw_parts_mut(self.slice.as_ptr() as *mut f32, self.slice.len())
  }

  unsafe fn argmax_ref_mut(&self) -> &mut ArgmaxResult<f32, WorldSpace> {
    &mut *(self.argmax_ref as *const _ as *mut ArgmaxResult<f32, WorldSpace>)
  }

  fn offset_to_xy_normalized(&self, offset: u64) -> Point2D<f32, WorldSpace> {
    let xy = offset_to_xy(offset, self.size) + self.top_left.to_vector();
    (xy.to_f32() / self.global_size as f32).cast_unit()
  }

  fn pixels_mut(&self) -> impl Iterator<Item = (Point2D<f32, WorldSpace>, &mut f32)> {
    unsafe { self.slice_mut() }
      .iter_mut()
      .enumerate()
      .map(move |(i, value)| (
        self.offset_to_xy_normalized(i as u64),
        value
      ))
  }
}

#[derive(Copy, Clone, Debug)]
pub struct ArgmaxResult<T, Space> {
  pub distance: f32,
  pub point: Point2D<T, Space>
}

impl<T: Default, S> Default for ArgmaxResult<T, S> {
  fn default() -> Self {
    ArgmaxResult {
      distance: f32::MAX / 2.0,
      point: Point2D::default()
    }
  }
}

impl<T, S> Eq for ArgmaxResult<T, S> {}

impl<T, S> PartialEq for ArgmaxResult<T, S> {
  fn eq(&self, other: &Self) -> bool {
    self.distance.eq(&other.distance)
  }
}

impl<T, S> PartialOrd for ArgmaxResult<T, S> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.distance.partial_cmp(&other.distance)
  }
}

impl<T, S> std::cmp::Ord for ArgmaxResult<T, S> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // waiting for #![feature(total_cmp)]
    fn total_cmp(left: f32, right: f32) -> std::cmp::Ordering {
      let mut left = left.to_bits() as i32;
      let mut right = right.to_bits() as i32;
      left ^= (((left >> 31) as u32) >> 1) as i32;
      right ^= (((right >> 31) as u32) >> 1) as i32;

      left.cmp(&right)
    }
    total_cmp(self.distance, other.distance)
  }
}

fn offset_to_xy(offset: u64, width: u64) -> Point2D<u64, PixelSpace> {
  [ offset % width,
    offset / width,
  ].into()
}

fn xy_to_offset(xy: Point2D<i64, PixelSpace>, width: u64) -> u64 {
  xy.y as u64 * width + xy.x as u64
}

impl Argmax2D {
  pub fn new(resolution: u64, chunk_size: u64) -> Result<Self> {
    if resolution % chunk_size != 0 {
      bail!("distance map resolution is not divisible by the chunk resolution")
    };
    let chunk_area = resolution.pow(2);
    let chunk_count = (resolution / chunk_size).pow(2);
    Ok(Self {
      dist_map: vec![f32::MAX / 2.0; chunk_area as usize],
      resolution,
      chunk_size,
      chunk_argmax: vec![ArgmaxResult::default(); chunk_count as usize]
    })
  }

  fn get_chunk(&self, id: u64) -> Chunk {
    let chunk_area = self.chunk_size.pow(2);
    Chunk {
      slice: &self.dist_map[(chunk_area * id) as usize .. (chunk_area * (id + 1)) as usize],
      argmax_ref: &self.chunk_argmax[id as usize],
      top_left: offset_to_xy(id, self.resolution / self.chunk_size) * self.chunk_size,
      id,
      size: self.chunk_size,
      global_size: self.resolution
    }
  }

  fn chunks(&self) -> impl Iterator<Item = Chunk> {
    let chunk_count = (self.resolution / self.chunk_size).pow(2);
    (0..chunk_count).into_iter()
      .map(move |id| self.get_chunk(id))
  }

  fn chunks_par_iter(&self) -> impl rayon::iter::ParallelIterator<Item = Chunk> {
    use rayon::prelude::*;

    let chunk_count = (self.resolution / self.chunk_size).pow(2);
    (0..chunk_count).into_par_iter()
      .map(move |id| self.get_chunk(id))
  }

  pub fn find_max(&self) -> ArgmaxResult<f32, WorldSpace> {
    self.chunk_argmax.iter().cloned()
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

    let domain = domain.to_box2d().intersection_unchecked(
      &Box2D::new(
        Point2D::splat(0.0),
        Point2D::splat(1.0)
      )
    ) * self.resolution as f32;
    let chunk_span = (domain / self.chunk_size as f32)
      .round_out()
      .to_i64();

    (chunk_span.min.y .. chunk_span.max.y)
      .into_par_iter()
      .flat_map(move |chunk_y|
        (chunk_span.min.x .. chunk_span.max.x)
          .into_par_iter().map(move |chunk_x| [chunk_x, chunk_y].into())
      )
      .for_each(move |chunk_xy| {
        let chunk = self.get_chunk(xy_to_offset(chunk_xy, self.resolution / self.chunk_size));
        let max_dist = chunk.pixels_mut().map(|(xy_normalized, value)| {
          *value = (*value).min(sdf(xy_normalized));
          ArgmaxResult {
            distance: *value,
            point: xy_normalized
          }
        }).max()
          .unwrap();
        // each thread is guaranteed to have distinct chunks
        *unsafe { chunk.argmax_ref_mut() } = max_dist;
      });
  }

  pub fn pixels(&self) -> impl Iterator<Item = ArgmaxResult<u64, PixelSpace>> + '_ {
    self.chunks().flat_map(move |chunk| {
      chunk.slice.iter().enumerate().map(move |(i, pixel)|
        ArgmaxResult {
          distance: *pixel,
          point: offset_to_xy(i as u64, chunk.size) + chunk.top_left.to_vector()
        }
      )
    })
  }

  pub fn invert(&mut self) {
    use rayon::prelude::*;

    self.chunks_par_iter().for_each(|chunk| {
      let max_dist = chunk.pixels_mut().map(|(xy_normalized, value)| {
        *value = -*value;
        ArgmaxResult {
          distance: *value,
          point: xy_normalized
        }
      }).max()
        .unwrap();
      *unsafe { chunk.argmax_ref_mut() } = max_dist;
    });
  }

  pub fn domain_empirical(center: Point2D<f32, WorldSpace>, max_dist: f32) -> Rect<f32, WorldSpace> {
    let size = max_dist * 4.0 * std::f32::consts::SQRT_2;
    Rect {
      origin: (center.to_vector() - V2::splat(size) / 2.0).to_point(),
      size: Size2D::splat(size)
    }
  }

  pub fn iter(&mut self) -> ArgmaxIter {
    let min_dist = 0.5 * std::f32::consts::SQRT_2 / self.resolution as f32;
    ArgmaxIter {
      argmax: self,
      min_dist
    }
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
    self.min_dist = value / self.argmax.resolution as f32;
    self
  }

  pub fn build(self) -> impl Iterator<Item = (ArgmaxResult<f32, WorldSpace>, &'a mut Argmax2D)> {
    let min_dist = self.min_dist;

    (0..).map(move |_| {
      // this is awkward...
      let argmax = unsafe { &mut *(self.argmax as *const _ as *mut Argmax2D) };
      (argmax.find_max(), argmax)
    })
      .take_while(move |(dist, _)| dist.distance >= min_dist)
  }
}