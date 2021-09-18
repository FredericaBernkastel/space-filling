use {
  crate::{
    argmax2d::{Argmax2D, ArgmaxResult},
    geometry::{
      BoundingBox, Shape,
      PixelSpace, WorldSpace,
      Translation, Rotation, Scale
    },
    sdf::SDF
  },
  euclid::{Box2D, Point2D, Size2D, Vector2D as V2},
  image::{
    ImageBuffer, Luma, Pixel
  },
  std::{
    ops::Deref
  }
};

mod impl_draw_rgbaimage;
#[cfg(test)] mod tests;
pub use impl_draw_rgbaimage::draw_parallel;

pub trait Draw<Backend>: Shape {
  fn draw(&self, image: &mut Backend);
}

pub trait DrawSync<Backend>: Draw<Backend> + Send + Sync {}
impl <T, Backend> DrawSync<Backend> for T where T: Draw<Backend> + Send + Sync {}

impl<B> SDF<f32> for Box<dyn Draw<B>> { fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 { self.deref().sdf(pixel) } }
impl<B> BoundingBox<f32, WorldSpace> for Box<dyn Draw<B>> { fn bounding_box(&self) -> Box2D<f32, WorldSpace> { self.deref().bounding_box() } }

//impl<B> Draw<B> for Circle { fn draw(&self, _: &mut B) { unreachable!(); } }
//impl<B> Draw<B> for Square { fn draw(&self, _: &mut B) { unreachable!(); } }

impl <B, S, T> Draw<B> for Translation<S, T> where Translation<S, T>: Shape { fn draw(&self, _: &mut B) { unreachable!("Draw is only implemented for Texture") } }
impl <B, S, T> Draw<B> for Rotation<S, T> where Rotation<S, T>: Shape { fn draw(&self, _: &mut B) { unreachable!("Draw is only implemented for Texture") } }
impl <B, S, T> Draw<B> for Scale<S, T> where Scale<S, T>: Shape { fn draw(&self, _: &mut B) { unreachable!("Draw is only implemented for Texture") } }

#[derive(Debug, Copy, Clone)]
pub struct Texture<S, T> {
  pub shape: S,
  pub texture: T
}
impl <S, T> SDF<f32> for Texture<S, T> where S: SDF<f32> {
  fn sdf(&self, pixel: Point2D<f32, WorldSpace>) -> f32 { self.shape.sdf(pixel) } }
impl <S, T> BoundingBox<f32, WorldSpace> for Texture<S, T> where S: BoundingBox<f32, WorldSpace> {
  fn bounding_box(&self) -> Box2D<f32, WorldSpace> { self.shape.bounding_box() } }

// try to fit world in the center of image, preserving aspect ratio
fn rescale_bounding_box(
  bounding_box: Box2D<f32, WorldSpace>,
  resolution: Size2D<u32, PixelSpace>
) -> (
  Option<Box2D<u32, PixelSpace>>, // bounding_box,
  V2<f32, PixelSpace>, // offset
  f32 // min_side
) {
  let min_side = resolution.width.min(resolution.height) as f32;
  let offset = (resolution.to_vector().to_f32() - V2::splat(min_side)) / 2.0;
  let bounding_box = bounding_box
    .scale(min_side, min_side).cast_unit()
    .round_out()
    .translate(offset)
    .intersection(&Box2D::from_size(resolution.to_f32()))
    .map(|x| x.to_u32());
  (bounding_box, offset, min_side)
}

/// Draw shapes, parallel.
/// Faster compared to [`draw_parallel`], low memory usage.
/// Will cause undefined behaviour if two shapes intersect.
pub fn draw_parallel_unsafe<B>(
  framebuffer: &mut B,
  shapes: impl rayon::iter::ParallelIterator<Item = Box<dyn DrawSync<B>>>
) -> &mut B where B: Sync + Send {
  shapes.for_each(|shape|
    shape.draw(unsafe { &mut *(framebuffer as *const _ as *mut B) })
  );
  framebuffer
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