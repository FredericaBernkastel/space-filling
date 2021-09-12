use {
  std::{
    thread, sync::Arc, ops::Deref
  },
  image::{
    DynamicImage, GenericImageView, ImageBuffer, Luma, Rgba, Pixel,
    imageops::FilterType, RgbaImage
  },
  euclid::{Point2D, Box2D, Rect, Size2D, Vector2D},

  crate::{
    error::{ErrorKind::NoneError, Result},
    geometry::{Circle, WorldSpace, PixelSpace, BoundingBox},
    sdf::SDF,
    argmax2d::{ArgmaxResult, Argmax2D}
  }
};

pub trait Draw<Backend> {
  fn draw(&self, image: &mut Backend);
}

pub trait DrawSync<Backend>: Draw<Backend> + Shape + Send + Sync {}
impl <T, Backend> DrawSync<Backend> for T where T: Draw<Backend> + Shape + Send + Sync {}

pub trait Shape: SDF<f32> + BoundingBox<f32, WorldSpace> {
  fn texture<T>(self, texture: T) -> Texture<Self, T> where Self: Sized {
    Texture { shape: self, texture }
  }
}
impl <T> Shape for T where T: SDF<f32> + BoundingBox<f32, WorldSpace> {}

impl<B> SDF<f32> for Arc<dyn DrawSync<B>> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 { self.deref().sdf(pixel) } }
impl<B> BoundingBox<f32, WorldSpace> for Arc<dyn DrawSync<B>> {
  fn bounding_box(&self) -> Box2D<f32, WorldSpace> { self.deref().bounding_box() } }
impl<B> Draw<B> for Arc<dyn DrawSync<B>> {
  fn draw(&self, image: &mut B) { self.deref().draw(image) } }

impl<B> Draw<B> for Circle<f32, WorldSpace> { fn draw(&self, _: &mut B) { unreachable!(); } }
impl<B> Draw<B> for Rect<f32, WorldSpace> { fn draw(&self, _: &mut B) { unreachable!(); } }

pub struct Texture<S, T> {
  pub shape: S,
  pub texture: T
}
impl <S, T> SDF<f32> for Texture<S, T> where S: SDF<f32> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 { self.shape.sdf(pixel) } }
impl <S, T> BoundingBox<f32, WorldSpace> for Texture<S, T> where S: BoundingBox<f32, WorldSpace> {
  fn bounding_box(&self) -> Box2D<f32, WorldSpace> { self.shape.bounding_box() } }

impl <Cutie> Draw<RgbaImage> for Texture<Cutie, Rgba<u8>>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace>
{
  fn draw(&self, image: &mut RgbaImage) {
    let resolution: Size2D<_, PixelSpace> = image.dimensions().into();
    let (bounding_box, offset, min_side) = rescale_bounding_box(self.bounding_box(), resolution);
    let bounding_box = match bounding_box {
      Some(x) => x,
      None => return // bounding box has no intersection with screen at all
    };

    itertools::iproduct!(bounding_box.y_range(), bounding_box.x_range())
      .map(|(y, x)| Point2D::<_, PixelSpace>::from([x, y]))
      .for_each(|pixel| {
        let pixel_world = ((pixel.to_f32() - offset).to_vector() / min_side)
          .cast_unit().to_point();
        if self.sdf(pixel_world) <= 0.0 {
          let pixel = image.get_pixel_mut(pixel.x, pixel.y);
          *pixel = pixel.map2(&self.texture, |p1, p2| p1.max(p2));
        }
      });
  }
}

impl <'a, Cutie> Draw<RgbaImage> for Texture<Cutie, &'a DynamicImage>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace>
{
  fn draw(&self, image: &mut RgbaImage) {
    let resolution: Size2D<_, PixelSpace> = image.dimensions().into();
    let (bounding_box, offset, min_side) = rescale_bounding_box(self.shape.bounding_box(), resolution);
    let bounding_box = match bounding_box {
      Some(x) => x,
      None => return
    };
    let tex = rescale_texture(self.texture, bounding_box.size());

    itertools::iproduct!(bounding_box.y_range(), bounding_box.x_range())
      .map(|(y, x)| Point2D::<_, PixelSpace>::from([x, y]))
      .for_each(|pixel| {
        let pixel_world = ((pixel.to_f32() - offset).to_vector() / min_side)
          .cast_unit().to_point();
        let pixel_tex = pixel - bounding_box.min.to_vector();

        if self.shape.sdf(pixel_world) <= 0.0 {
          let pixel = image.get_pixel_mut(pixel.x, pixel.y);
          *pixel = pixel.map2(&tex.get_pixel(pixel_tex.x, pixel_tex.y), |p1, p2| p1.max(p2));
        }
      });
  }
}

impl <Cutie> Draw<RgbaImage> for Texture<Cutie, DynamicImage>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace> + Clone {
  fn draw(&self, image: &mut RgbaImage) {
    Texture {
      shape: self.shape.clone(),
      texture: &self.texture
    }.draw(image)
  }
}

impl <Cutie> Draw<RgbaImage> for Texture<Cutie, Arc<DynamicImage>>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace> + Clone {
  fn draw(&self, image: &mut RgbaImage) {
    Texture {
      shape: self.shape.clone(),
      texture: self.texture.as_ref()
    }.draw(image)
  }
}

impl <Cutie> Draw<RgbaImage> for Texture<Cutie, Arc<Rgba<u8>>>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace> + Clone {
  fn draw(&self, image: &mut RgbaImage) {
    Texture {
      shape: self.shape.clone(),
      texture: *self.texture
    }.draw(image)
  }
}

/*impl <Cutie> Draw<RgbaImage> for Texture<Cutie, PathBuf>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace> + Clone + Send + Sync {
  fn draw(&self, image: &mut RgbaImage) {
    let shape = self.shape.clone();
    image::open(&self.texture)
      .map(move |tex| {
        println!("{:?} -> {:?}", shape.bounding_box(), self.texture);
        shape.texture(tex).draw(image);
      })
      .map_err(|_| println!("unable to open {:?}", self.texture))
      .ok();
  }
}*/

// try to fit world in the center of image, preserving aspect ratio
fn rescale_bounding_box(
  bounding_box: Box2D<f32, WorldSpace>,
  resolution: Size2D<u32, PixelSpace>
) -> (
  Option<Box2D<u32, PixelSpace>>, // bounding_box,
  Vector2D<f32, PixelSpace>, // offset
  f32 // min_side
) {
  let min_side = resolution.width.min(resolution.height) as f32;
  let offset = (resolution.to_vector().to_f32() - Vector2D::splat(min_side)) / 2.0;
  let bounding_box = bounding_box
    .scale(min_side, min_side).cast_unit()
    .round_out()
    .translate(offset)
    .intersection(&Box2D::from_size(resolution.to_f32()))
    .map(|x| x.to_u32());
  (bounding_box, offset, min_side)
}

// resize the image to cover the entire container,
// even if it has to cut off one of the edges
fn rescale_texture(texture: &DynamicImage, size: Size2D<u32, PixelSpace>) -> DynamicImage {
  let tex_size = Size2D::from(texture.dimensions()).to_f32();
  let scaling_factor = tex_size.to_vector()
    .component_div(size.to_f32().to_vector());
  let scaling_factor = scaling_factor.x.min(scaling_factor.y);
  let bound_inner = size.to_f32() * scaling_factor;
  let bound_inner = Rect::new(
    ((tex_size - bound_inner) / 2.0).to_vector().to_point(),
    bound_inner
  ).to_u32();
  texture.crop_imm(
    bound_inner.origin.x,
    bound_inner.origin.y,
    bound_inner.size.width,
    bound_inner.size.height
  ).resize_exact(size.width, size.height, FilterType::Triangle)
}

/// draw shapes, parallel.
/// will use resolution.x * resolution.y * num_threads * 4 bytes of memory
pub fn draw_parallel(
  shapes: impl Iterator<Item = std::sync::Arc<dyn DrawSync<RgbaImage>>>,
  resolution: Point2D<u32, PixelSpace>,
  num_threads: usize
) -> Result<image::RgbaImage> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let mut draw_data = shapes
    .collect::<Vec<_>>();
  // will distribute the load between threads [statistically] evenly
  draw_data.shuffle(&mut rng);

  let num_threads = num_threads.min(draw_data.len());

  let draw_data_chunks = draw_data
    .chunks((draw_data.len() as f32 / num_threads as f32).ceil() as usize)
    .map(|chunk| chunk.to_vec())
    .collect::<Vec<_>>();

  if draw_data_chunks.len() != num_threads {
    error_chain::bail!("chunks are unsatisfied");
  }

  let partial_buffers = draw_data_chunks.into_iter().map(|chunk| {
    thread::spawn( move || {
      let mut framebuffer: ImageBuffer<Rgba<u8>, _> =
        ImageBuffer::new(resolution.x, resolution.y);

      chunk.into_iter()
        .for_each(|shape| {
          shape.draw(&mut framebuffer);
        });

      framebuffer
    })
  }).collect::<Vec<_>>() // thread handles
    .into_iter()
    .map(|thread| thread.join().unwrap())
    .collect::<Vec<_>>();

  let mut final_buffer = partial_buffers.get(0).cloned().ok_or(NoneError)?;

  // merge partial buffers
  partial_buffers
    .into_iter()
    .skip(1)
    .for_each(|buffer|
      image::imageops::overlay(&mut final_buffer, &buffer, 0, 0)
    );

  Ok(final_buffer)
}

/// draw shapes, parallel.
/// faster, low memory usage. will cause undefined behaviour if two shapes intersect.
pub fn draw_parallel_unsafe(
  shapes: impl rayon::iter::ParallelIterator<Item = Box<dyn DrawSync<RgbaImage>>>,
  resolution: Point2D<u32, PixelSpace>
) -> Result<image::RgbaImage> {
  let framebuffer: RgbaImage = ImageBuffer::new(resolution.x, resolution.y);

  shapes.for_each(|shape|
    shape.draw(unsafe { &mut *(&framebuffer as *const _ as *mut RgbaImage) })
  );

  Ok(framebuffer)
}

impl Argmax2D {
  pub fn display_debug(&self) -> image::RgbImage {
    let mut image = ImageBuffer::<image::Rgb<u8>, _>::new(
      self.resolution as u32,
      self.resolution as u32
    );
    let max_dist = self.find_max().distance;
    self.pixels().for_each(|ArgmaxResult { distance, point }| {
      let color = Luma::from([(distance / max_dist * 255.0) as u8]);
      *image.get_pixel_mut(point.x as u32, point.y as u32) = color.to_rgb();
    });
    image
  }
}

#[test] fn texture() -> Result<()> {
  let mut image = image::RgbaImage::new(128, 128);
  Circle {
    xy: [0.5, 0.5].into(),
    r: 0.25
  } .texture(&image::open("doc/embedded.jpg")?)
    .draw(&mut image);
  image.save("test_texture.png")?;
  Ok(())
}

#[test] fn polymorphic() -> Result<()> {
  let mut image = image::RgbaImage::new(128, 128);
  let shapes: Vec<Arc<dyn DrawSync<RgbaImage>>> = vec![
    Arc::new(Circle { xy: [0.25, 0.25].into(), r: 0.25 }),
    Arc::new(Rect { origin: [0.5, 0.5].into(), size: [0.5, 0.5].into() })
  ];
  shapes.into_iter()
    .map(|shape| Box::new(shape.texture(Luma([255u8]).to_rgba())) as Box<dyn DrawSync<_>>)
    .for_each(|shape| shape.draw(&mut image));
  image.save("test_polymorphic.png")?;
  Ok(())
}