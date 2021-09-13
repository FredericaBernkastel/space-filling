use {
  crate::{
    argmax2d::{Argmax2D, ArgmaxResult},
    geometry::{BoundingBox, Circle, PixelSpace, WorldSpace},
    sdf::SDF
  },
  euclid::{Box2D, Point2D, Rect, Size2D, Vector2D},
  image::{
    ImageBuffer, Luma, Pixel, Rgba
  },
  std::{
    ops::Deref, sync::Arc
  }
};

mod impl_draw_rgbaimage;
pub use impl_draw_rgbaimage::draw_parallel;

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

impl <Cutie, B> Draw<B> for Texture<Cutie, Arc<Rgba<u8>>>
  where Cutie: SDF<f32> + BoundingBox<f32, WorldSpace> + Clone,
        Texture<Cutie, Rgba<u8>>: Draw<B> {
  fn draw(&self, image: &mut B) {
    Texture {
      shape: self.shape.clone(),
      texture: *self.texture
    }.draw(image)
  }
}

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

#[test] fn texture() -> crate::error::Result<()> {
  let mut image = image::RgbaImage::new(128, 128);
  Circle {
    xy: [0.5, 0.5].into(),
    r: 0.25
  } .texture(&image::open("doc/embedded.jpg")?)
    .draw(&mut image);
  image.save("test_texture.png")?;
  Ok(())
}

#[test] fn polymorphic() -> crate::error::Result<()> {
  let mut image = image::RgbaImage::new(128, 128);
  let shapes: Vec<Arc<dyn DrawSync<image::RgbaImage>>> = vec![
    Arc::new(Circle { xy: [0.25, 0.25].into(), r: 0.25 }),
    Arc::new(Rect { origin: [0.5, 0.5].into(), size: [0.5, 0.5].into() })
  ];
  shapes.into_iter()
    .map(|shape| Box::new(shape.texture(Luma([255u8]).to_rgba())) as Box<dyn DrawSync<_>>)
    .for_each(|shape| shape.draw(&mut image));
  image.save("test_polymorphic.png")?;
  Ok(())
}