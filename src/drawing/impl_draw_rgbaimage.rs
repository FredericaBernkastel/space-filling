#![allow(non_snake_case)]
use {
  std::{thread, sync::Arc, ops::Fn},
  euclid::{Point2D, Rect, Size2D},
  image::{
    DynamicImage, GenericImageView, Pixel, Rgba, RgbaImage,
    imageops::FilterType
  },
  num_traits::{NumCast, AsPrimitive},
  anyhow::Result,
  crate::{
    drawing::{Draw, DrawSync, Shape, Texture, rescale_bounding_box},
    geometry::{BoundingBox, PixelSpace, WorldSpace},
    sdf::SDF
  }
};

impl <Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, Rgba<u8>>
  where Cutie: Shape<P> + Clone,
        P: NumCast + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    self.shape.clone()
      .texture(|_| self.texture)
      .draw(image);
  }
}

impl <'a, Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, &'a DynamicImage>
  where Cutie: Shape<P>,
        P: NumCast + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    let resolution: Size2D<_, PixelSpace> = image.dimensions().into();
    let (bounding_box, offset, min_side) =
      rescale_bounding_box(self.shape.bounding_box().to_f64(), resolution);
    let bounding_box = match bounding_box {
      Some(x) => x,
      None => return
    };
    let Δp = 1.0 / min_side;
    let tex = rescale_texture(self.texture, bounding_box.size().to_u32());

    itertools::iproduct!(bounding_box.y_range(), bounding_box.x_range())
      .map(|(y, x)| Point2D::<_, PixelSpace>::from([x, y]))
      .for_each(|pixel| {
        let pixel_world = ((pixel.to_f64() - offset).to_vector() / min_side)
          .cast_unit().to_point();
        let tex_px = pixel - bounding_box.min.to_vector();
        let tex_px = tex.get_pixel(tex_px.x, tex_px.y);

        let sdf = self.sdf(pixel_world.cast::<P>()).as_();
        let pixel = image.get_pixel_mut(pixel.x, pixel.y);
        *pixel = sdf_overlay_aa(sdf, Δp, *pixel, tex_px);
      });
  }
}

/// F: Fn(pixel: Point2D) -> Rgba<u8>
/// where pixel is in normalized texture coordinates.
impl <Cutie, F, P> Draw<P, RgbaImage> for Texture<Cutie, F>
  where Cutie: Shape<P>,
        F: Fn(Point2D<P, WorldSpace>) -> Rgba<u8>,
        P: NumCast + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    let resolution: Size2D<_, PixelSpace> = image.dimensions().into();
    let (bounding_box, offset, min_side) =
      rescale_bounding_box(self.bounding_box().to_f64(), resolution);
    let bounding_box = match bounding_box {
      Some(x) => x,
      None => return // bounding box has no intersection with screen at all
    };
    let Δp = 1.0 / min_side;
    let tex_scale = bounding_box.size().width.min(bounding_box.size().height) as f64;

    itertools::iproduct!(bounding_box.y_range(), bounding_box.x_range())
      .map(|(y, x)| Point2D::<_, PixelSpace>::from([x, y]))
      .for_each(|pixel| {
        let pixel_world = ((pixel.to_f64() - offset).to_vector() / min_side)
          .cast_unit().to_point();
        let sdf = self.sdf(pixel_world.cast::<P>()).as_();

        let tex_px = ((pixel - bounding_box.min.to_vector()).to_f64() / tex_scale).cast_unit();
        let tex_px = (self.texture)(tex_px.cast::<P>());

        let pixel = image.get_pixel_mut(pixel.x, pixel.y);
        *pixel = sdf_overlay_aa(sdf, Δp, *pixel, tex_px);
      });
  }
}

impl <Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, DynamicImage>
  where Cutie: Shape<P> + Clone,
        P: NumCast + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    Texture {
      shape: self.shape.clone(),
      texture: &self.texture
    }.draw(image)
  }
}

impl <Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, Arc<DynamicImage>>
  where Cutie: Shape<P> + Clone,
        P: NumCast + AsPrimitive<f64>
{
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

fn sdf_overlay_aa(sdf: f64, Δp: f64, mut col1: Rgba<u8>, mut col2: Rgba<u8>) -> Rgba<u8> {
  let Δf = (0.5 * Δp - sdf) // antialias
    .clamp(0.0, Δp);
  let alpha = Δf / Δp;
  // overlay blending with premultiplied alpha
  col2.0[3] = ((col2.0[3] as f64) * alpha) as u8;
  col1.blend(&col2);
  col1
}

/// Draw shapes, parallel.
/// Will use `resolution.x * resolution.y * num_threads * 4` bytes of memory.
pub fn draw_parallel<P>(
  shapes: impl Iterator<Item = Arc<dyn DrawSync<P, RgbaImage>>>,
  resolution: Point2D<u32, PixelSpace>,
  num_threads: usize
) -> Result<RgbaImage>
  where P: 'static
{
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let mut draw_data = shapes
    .collect::<Vec<_>>();
  // will distribute the load between threads [statistically] evenly
  draw_data.shuffle(&mut rng);

  let num_threads = num_threads.min(draw_data.len());

  let draw_data_chunks = draw_data
    .chunks((draw_data.len() as f64 / num_threads as f64).ceil() as usize)
    .map(|chunk| chunk.to_vec())
    .collect::<Vec<_>>();

  if draw_data_chunks.len() != num_threads {
    anyhow::bail!("chunks are unsatisfied");
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

  let mut final_buffer = partial_buffers[0].clone();

  // merge partial buffers
  partial_buffers
    .into_iter()
    .skip(1)
    .for_each(|buffer|
      image::imageops::overlay(&mut final_buffer, &buffer, 0, 0)
    );

  Ok(final_buffer)
}
