use {
  std::{thread, sync::Arc},
  euclid::{Point2D, Rect, Size2D},
  image::{
    DynamicImage, GenericImageView, Pixel, Rgba, RgbaImage,
    imageops::FilterType
  },
  crate::{
    drawing::{Draw, DrawSync, Texture, rescale_bounding_box},
    error::{Result, ErrorKind::NoneError},
    geometry::{BoundingBox, PixelSpace, WorldSpace},
    sdf::SDF
  }
};

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

/// Draw shapes, parallel.
/// Will use resolution.x * resolution.y * num_threads * 4 bytes of memory
pub fn draw_parallel(
  shapes: impl Iterator<Item = Arc<dyn DrawSync<RgbaImage>>>,
  resolution: Point2D<u32, PixelSpace>,
  num_threads: usize
) -> Result<RgbaImage> {
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
      let mut framebuffer = RgbaImage::new(resolution.x, resolution.y);

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
