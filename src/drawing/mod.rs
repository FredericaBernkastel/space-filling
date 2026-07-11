#![allow(non_snake_case)]
use {
  crate::{
    solver::{
      Argmax2D, adf::{ADF, quadtree::Quadtree}
    },
    geometry::{
      self, BoundingBox, Shape, Real,
      DistPoint, Translation, Rotation, Scale,
      Aabb, Point, P2, V2
    },
    sdf::SDF
  },
  image::{
    ImageBuffer, Luma, Rgba, Pixel, RgbaImage
  },
  nalgebra::Scalar,
  num_traits::{AsPrimitive, Signed}
};

mod impl_draw_rgbaimage;
#[cfg(test)] mod tests;

pub trait Draw<Float: Scalar, Backend>: Shape<Float, 2> {
  fn draw(&self, image: &mut Backend);
}

static MSG: &str = "Draw is only implemented for Texture";

// Rust doesn't support trait upcasting yet
impl <B, S, P: Scalar> Draw<P, B> for Translation<S, P, 2> where Translation<S, P, 2>: Shape<P, 2> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }
impl <B, S, P: Scalar> Draw<P, B> for Rotation<S, P, 2> where Rotation<S, P, 2>: Shape<P, 2> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }
impl <B, S, P: Scalar> Draw<P, B> for Scale<S, P> where Scale<S, P>: Shape<P, 2> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }

impl <B, P: Scalar> Draw<P, B> for geometry::Line<P, 2> where geometry::Line<P, 2>: Shape<P, 2> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }
impl <B, P, U> Draw<P, B> for geometry::Polygon<U> where P: Real, U: AsRef<[P2<P>]> {
  fn draw(&self, _: &mut B) { unreachable!("{}", MSG) } }

#[derive(Debug, Copy, Clone)]
pub struct Texture<S, T> {
  pub shape: S,
  pub texture: T
}
impl <P: Scalar, S, T, const D: usize> SDF<P, D> for Texture<S, T> where S: SDF<P, D> {
  fn sdf(&self, pixel: Point<P, D>) -> P { self.shape.sdf(pixel) } }
impl <P: Scalar, S, T, const D: usize> BoundingBox<P, D> for Texture<S, T> where S: BoundingBox<P, D> {
  fn bounding_box(&self) -> Aabb<P, D> { self.shape.bounding_box() } }

// try to fit world in the center of image, preserving aspect ratio
fn rescale_bounding_box(
  bounding_box: Aabb<f64, 2>,
  resolution: V2<u32>
) -> (
  Option<Aabb<u32, 2>>, // bounding_box,
  V2<f64>, // offset
  f64 // min_side
) {
  let min_side = resolution.x.min(resolution.y) as f64;
  let res_f = V2::new(resolution.x as f64, resolution.y as f64);
  let offset = (res_f - V2::repeat(min_side)) / 2.0;
  let bounding_box = Aabb::<f64, 2>::new(
      (bounding_box.min * min_side).map(f64::floor),
      (bounding_box.max * min_side).map(f64::ceil),
    )
    .translate(offset)
    .intersection(&Aabb::new(P2::new(0.0, 0.0), P2::from(res_f)))
    .map(|b| Aabb::new(b.min.map(|x| x as u32), b.max.map(|x| x as u32)));
  (bounding_box, offset, min_side)
}

/// Draw shapes in parallel, each rayon task writing directly to `framebuffer`.
///
/// The tasks share `framebuffer` through an aliased `&mut` with no
/// synchronization, so overlapping writes would be a data race (potential UB).
/// Sound only when the shapes cover disjoint regions — which space-filling
/// distributions guarantee, their shapes occupying disjoint balls. See
/// `examples/gd_adf/04_polymorphic.rs`.
pub fn draw_parallel<Float: Scalar, Backend, Sh>(
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

pub fn display_sdf(sdf: impl Fn(P2<f64>) -> f64, image: &mut RgbaImage, brightness: f64) {
  let resolution = image.width();
  let Δp = 1.0 / resolution as f64;

  // distance scalar field
  image.enumerate_pixels_mut()
    .for_each(|(x, y, pixel)| {
      let pixel_world = P2::new(x as f64, y as f64) / resolution as f64;
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

impl <Data, _Float: Real> Quadtree<Data, _Float, 2> {
  pub fn draw_layout(&self, image: &mut RgbaImage) -> &Self {
    use geometry::Line;

    let px = 1.0 / image.width() as f64;
    self.traverse(&mut |node| {
      if !node.is_leaf() { return Ok(()) };

      let min = node.rect.min.map(|x| x.to_f64().unwrap());
      let size = node.rect.size().map(|x| x.to_f64().unwrap());
      let lines = [
        [[0.0, 0.0], [size.x, 0.0]],
        [[size.x, 0.0], [size.x, size.y]],
        [[size.x, size.y], [0.0, size.y]],
        [[0.0, size.y], [0.0, 0.0]]
      ];
      let alpha = 1.0 - (node.depth as f64 / self.max_depth as f64);
      lines.iter().for_each(|&[a, b]| {
        Line { a: a.into(), b: b.into(), thickness: px }
          .translate(min.coords)
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

  pub fn draw_bounding(&self, domain: Aabb<_Float, 2>, image: &mut RgbaImage) -> &Self {
    self.traverse(&mut |node| {
      if node.is_leaf() && node.rect.intersects(&domain) {
        let center = node.rect.center().map(|x| x.to_f64().unwrap());
        let size = node.rect.size().map(|x| x.to_f64().unwrap());
        geometry::Rect { size }
          .translate(center.coords)
          .texture(Rgba([0xFF, 0, 0, 0x7F]))
          .draw(image)
      }
      Ok(())
    }).ok();
    self
  }
}

impl <_Float: Real + Signed + AsPrimitive<f64> + Send + Sync> ADF<_Float, 2> {
  pub fn display_sdf(&self, image: &mut RgbaImage, brightness: f64) -> &Self {
    display_sdf(|p| self.sdf(p.map(|x| _Float::from(x).unwrap())).to_f64().unwrap(), image, brightness);
    self
  }
  pub fn draw_bucket_weights(&self, image: &mut RgbaImage) -> &Self {
    self.tree.traverse(&mut |node| {
      if node.is_leaf() {
        let center = node.rect.center().map(|x| x.to_f64().unwrap());
        let size = node.rect.size().map(|x| x.to_f64().unwrap());
        let alpha = (((node.data.len() - 1) as f64 / 3.0).powf(1.75)
          * 0.33 * 255.0) as u8;
        geometry::Rect { size }
          .translate(center.coords)
          .texture(Rgba([0x7F, 0xFF, 0, alpha]))
          .draw(image)
      }
      Ok(())
    }).ok();
    self
  }
}
