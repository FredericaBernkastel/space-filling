#![allow(non_snake_case)]
use {
  crate::{
    solver::{
      Argmax2D, adf::{ADF, quadtree::Quadtree}
    },
    geometry::{
      self, BoundingBox, Shape,
      PixelSpace, WorldSpace, DistPoint,
      Translation, Rotation, Scale
    },
    sdf::SDF
  },
  euclid::{Box2D, Point2D, Size2D, Vector2D as V2},
  image::{
    ImageBuffer, Luma, Rgba, Pixel, RgbaImage
  },
  num_traits::{Float, AsPrimitive, Signed}
};

mod impl_draw_rgbaimage;
#[cfg(test)] mod tests;

pub trait Draw<Prec, Backend>: Shape<Prec> {
  fn draw(&self, image: &mut Backend);
}

static MSG: &str = "Draw is only implemented for Texture";

// Rust doesn't support trait upcasting yet
impl <B, S, P> Draw<P, B> for Translation<S, P> where Translation<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }
impl <B, S, P> Draw<P, B> for Rotation<S, P> where Rotation<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }
impl <B, S, P> Draw<P, B> for Scale<S, P> where Scale<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }

impl <B, P> Draw<P, B> for geometry::Line<P> where geometry::Line<P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }
impl <B, P, U> Draw<P, B> for geometry::Polygon<U> where P: num_traits::Float, U: AsRef<[Point2D<P, WorldSpace>]> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }

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
/// May cause undefined behaviour.
pub fn draw_parallel<Float, Backend, Sh>(
  framebuffer: &mut Backend,
  shapes: impl rayon::iter::ParallelIterator<Item =Sh>
) -> &mut Backend
  where Backend: Sync + Send,
        Sh: AsRef<dyn Draw<Float, Backend> + Send + Sync>
{
  let ptr = framebuffer as *mut _ as usize;
  shapes.for_each(|shape|
    shape.as_ref().draw(unsafe { &mut *(ptr as *mut Backend) })
  );
  framebuffer
}

pub fn display_sdf(sdf: impl Fn(Point2D<f64, WorldSpace>) -> f64, image: &mut RgbaImage, brightness: f64) {
  let resolution = image.width();
  let Δp = 1.0 / resolution as f64;

  // distance scalar field
  image.enumerate_pixels_mut()
    .for_each(|(x, y, pixel)| {
      let pixel_world = Point2D::new(x, y).to_f64() / resolution as f64;
      let sdf = sdf(pixel_world);
      let mut alpha = (Δp  - sdf.abs()).clamp(0.0, Δp) / Δp;
      alpha *= (x > 0 && y > 0) as u8 as f64;
      let mut color = Luma([
        ((sdf * brightness).powf(1.0) * 255.0) as u8
      ]).to_rgba();
      color.blend(&Rgba([255, 0, 0, (alpha * 128.0) as u8]));
      *pixel = color;
    });
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

impl <Data, _Float: Float> Quadtree<Data, _Float> {
  pub fn draw_layout(&self, image: &mut RgbaImage) -> &Self {
    use geometry::Line;

    let px = 1.0 / image.width() as f64;
    self.traverse(&mut |node| {
      if node.children.is_some() { return Ok(()) };

      let rect = node.rect.cast();
      let lines = [
        [[0.0, 0.0], [rect.size.width, 0.0]],
        [[rect.size.width, 0.0], rect.size.into()],
        [rect.size.into(), [0.0, rect.size.height]],
        [[0.0, rect.size.height], [0.0, 0.0]]
      ];
      let alpha = 1.0 - (node.depth as f64 / self.max_depth as f64);
      lines.iter().for_each(|&[a, b]| {
        Line { a: a.into(), b: b.into(), thickness: px }
          .translate(rect.origin.to_vector())
          .texture(Rgba([
            ((1.0 - alpha).powi(2) * 255.0) as u8,
            0,
            128,
            ((1.0 - alpha).powf(0.5) * 255.0) as u8])
          )
          .draw(image);
      });
      Ok(())
    }).ok();
    self
  }

  pub fn draw_bounding(&self, domain: euclid::Rect<_Float, WorldSpace>, image: &mut RgbaImage) -> &Self {
    self.traverse(&mut |node| {
      if node.children.is_none() && node.rect.intersects(&domain) {
        let rect = node.rect.cast();
        geometry::Rect {
          size: rect.size.to_vector().to_point()
        } .translate(rect.origin.to_vector() + rect.size.to_vector() * 0.5)
          .texture(Rgba([0xFF, 0, 0, 0x7F]))
          .draw(image)
      }
      Ok(())
    }).ok();
    self
  }
}

impl <_Float: Float + Signed + AsPrimitive<f64>> ADF<_Float> {
  pub fn display_sdf(&self, image: &mut RgbaImage, brightness: f64) -> &Self {
    display_sdf(|p| self.sdf(p.cast()).to_f64().unwrap(), image, brightness);
    self
  }
  pub fn draw_bucket_weights(&self, image: &mut RgbaImage) -> &Self {
    self.tree.traverse(&mut |node| {
      if node.children.is_none() {
        let rect = node.rect;
        let alpha = (((node.data.len() - 1) as f64 / 3.0).powf(1.75)
          * 0.33 * 255.0) as u8;
        geometry::Rect {
          size: rect.size.to_vector().to_point()
        } .translate(rect.origin.to_vector() + rect.size.to_vector() * _Float::from(0.5).unwrap())
          .texture(Rgba([0x7F, 0xFF, 0, alpha]))
          .draw(image)
      }
      Ok(())
    }).ok();
    self
  }
}
