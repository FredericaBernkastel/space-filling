use {
  crate::{
    solver::{Argmax2D, DistPoint},
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

pub trait Draw<Prec, Backend>: Shape<Prec> {
  fn draw(&self, image: &mut Backend);
}

pub trait DrawSync<Prec, Backend>: Draw<Prec, Backend> + Send + Sync {}
impl <T, P, Backend> DrawSync<P, Backend> for T where T: Draw<P, Backend> + Send + Sync {}

impl<P, B> SDF<P> for Box<dyn Draw<P, B>> { fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P { self.deref().sdf(pixel) } }
impl<P, B> BoundingBox<P> for Box<dyn Draw<P, B>> { fn bounding_box(&self) -> Box2D<P, WorldSpace> { self.deref().bounding_box() } }

impl <B, S, P> Draw<P, B> for Translation<S, P> where Translation<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("Draw is only implemented for Texture") } }
impl <B, S, P> Draw<P, B> for Rotation<S, P> where Rotation<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("Draw is only implemented for Texture") } }
impl <B, S, P> Draw<P, B> for Scale<S, P> where Scale<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("Draw is only implemented for Texture") } }

#[derive(Debug, Copy, Clone)]
pub struct Texture<S, T> {
  pub shape: S,
  pub texture: T
}
impl <P, S, T> SDF<P> for Texture<S, T> where S: SDF<P> {
  fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P { self.shape.sdf(pixel) } }
impl <P, S, T> BoundingBox<P> for Texture<S, T> where S: BoundingBox<P> {
  fn bounding_box(&self) -> Box2D<P, WorldSpace> { self.shape.bounding_box() } }

// try to fit world in the center of image, preserving aspect ratio
fn rescale_bounding_box(
  bounding_box: Box2D<f64, WorldSpace>,
  resolution: Size2D<u32, PixelSpace>
) -> (
  Option<Box2D<u32, PixelSpace>>, // bounding_box,
  V2<f64, PixelSpace>, // offset
  f64 // min_side
) {
  let min_side = resolution.width.min(resolution.height) as f64;
  let offset = (resolution.to_vector().to_f64() - V2::splat(min_side)) / 2.0;
  let bounding_box = bounding_box
    .scale(min_side, min_side).cast_unit()
    .round_out()
    .translate(offset)
    .intersection(&Box2D::from_size(resolution.to_f64()))
    .map(|x| x.cast::<u32>());
  (bounding_box, offset, min_side)
}

/// Draw shapes, parallel.
/// Faster compared to [`draw_parallel`], low memory usage.
/// Will cause undefined behaviour if two shapes intersect.
pub fn draw_parallel_unsafe<P, B>(
  framebuffer: &mut B,
  shapes: impl rayon::iter::ParallelIterator<Item = Box<dyn DrawSync<P, B>>>
) -> &mut B where B: Sync + Send {
  shapes.for_each(|shape|
    shape.draw(unsafe { &mut *(framebuffer as *const _ as *mut B) })
  );
  framebuffer
}

impl Argmax2D {
  pub fn display_debug(&self) -> image::RgbImage {
    let mut image = ImageBuffer::<image::Rgb<u8>, _>::new(
      self.dist_map.resolution as u32,
      self.dist_map.resolution as u32
    );
    let max_dist = self.find_max().distance;
    self.dist_map.pixels().for_each(|DistPoint { distance, point }| {
      let color = Luma::from([(distance / max_dist * 255.0) as u8]);
      *image.get_pixel_mut(point.x as u32, point.y as u32) = color.to_rgb();
    });
    image
  }
}