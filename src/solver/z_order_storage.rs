use {
  crate::{
    error::Result,
    geometry::{WorldSpace, PixelSpace}
  },
  super::DistPoint,
  euclid::{Point2D, Rect, Box2D},
  rayon::iter::ParallelIterator,
  error_chain::bail
};

pub struct ZOrderStorage<T> {
  data: T,
  pub resolution: u64,
  pub chunk_size: u64,
}

impl <T> ZOrderStorage<T> {
  pub fn chunk_count(&self) -> u64 {
    (self.resolution / self.chunk_size).pow(2)
  }

  pub fn chunks_domain_par_iter(&self, domain: Rect<f32, WorldSpace>)
    -> impl ParallelIterator<Item = Point2D<u64, PixelSpace>> {
    use rayon::prelude::*;

    let domain = domain.to_box2d().intersection_unchecked(
      &Box2D::new(
        Point2D::splat(0.0),
        Point2D::splat(1.0)
      )
    ) * self.resolution as f32;
    let chunk_span = (domain / self.chunk_size as f32)
      .round_out()
      .cast::<u64>();

    (chunk_span.min.y .. chunk_span.max.y)
      .into_par_iter()
      .flat_map(move |chunk_y|
        (chunk_span.min.x .. chunk_span.max.x)
          .into_par_iter().map(move |chunk_x| [chunk_x, chunk_y].into())
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
}

impl<T> ZOrderStorage<Vec<T>> where T: Clone + Send + Sync {
  pub fn get_chunk(&self, id: u64) -> Chunk<T> {
    let chunk_area = self.chunk_size.pow(2);
    Chunk {
      slice: &self.data[(chunk_area * id) as usize .. (chunk_area * (id + 1)) as usize],
      top_left: offset_to_xy(id, self.resolution / self.chunk_size) * self.chunk_size,
      id,
      size: self.chunk_size,
      global_size: self.resolution
    }
  }

  pub fn get_chunk_xy(&self, xy: Point2D<u64, PixelSpace>) -> Chunk<T> {
    self.get_chunk(xy_to_offset(xy, self.resolution / self.chunk_size))
  }

  pub fn chunks(&self) -> impl Iterator<Item = Chunk<T>> {
    let chunk_count = (self.resolution / self.chunk_size).pow(2);
    (0..chunk_count).into_iter()
      .map(move |id| self.get_chunk(id))
  }

  pub fn chunks_par_iter(&self) -> impl ParallelIterator<Item = Chunk<T>> {
    use rayon::prelude::*;

    let chunk_count = (self.resolution / self.chunk_size).pow(2);
    (0..chunk_count).into_par_iter()
      .map(move |id| self.get_chunk(id))
  }

  pub fn pixels(&self) -> impl Iterator<Item = DistPoint<T, u64, PixelSpace>> + '_ {
    self.chunks().flat_map(move |chunk| {
      chunk.slice.iter().enumerate().map(move |(i, pixel)|
        DistPoint {
          distance: pixel.clone(),
          point: offset_to_xy(i as u64, chunk.size) + chunk.top_left.to_vector()
        }
      )
    })
  }
}

pub struct Chunk<'a, T> {
  pub slice: &'a [T],
  pub top_left: Point2D<u64, PixelSpace>,
  pub id: u64,
  pub size: u64,
  pub global_size: u64
}

impl<'a, T> Chunk<'a, T> {
  unsafe fn slice_mut(&self) -> &mut [f32] {
    std::slice::from_raw_parts_mut(self.slice.as_ptr() as *mut f32, self.slice.len())
  }

  fn offset_to_xy_normalized(&self, offset: u64) -> Point2D<f32, WorldSpace> {
    let xy = offset_to_xy(offset, self.size) + self.top_left.to_vector();
    (xy.to_f32() / self.global_size as f32).cast_unit()
  }

  pub(crate) fn pixels_mut(&self) -> impl Iterator<Item = (Point2D<f32, WorldSpace>, &mut f32)> {
    unsafe { self.slice_mut() }
      .iter_mut()
      .enumerate()
      .map(move |(i, value)| (
        self.offset_to_xy_normalized(i as u64),
        value
      ))
  }
}

fn offset_to_xy(offset: u64, width: u64) -> Point2D<u64, PixelSpace> {
  [ offset % width,
    offset / width,
  ].into()
}

fn xy_to_offset(xy: Point2D<u64, PixelSpace>, width: u64) -> u64 {
  xy.y * width + xy.x
}