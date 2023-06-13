#![allow(non_snake_case)]

use num_traits::Float;
use {
  std::{sync::Arc, ops::Fn},
  euclid::{Point2D, Rect, Size2D, Box2D},
  image::{
    DynamicImage, GenericImageView, Pixel, Rgba, RgbaImage,
    imageops::FilterType
  },
  num_traits::{NumCast, AsPrimitive},
  crate::{
    drawing::{Draw, Shape, Texture, rescale_bounding_box},
    geometry::{BoundingBox, PixelSpace, WorldSpace},
    sdf::SDF
  }
};

impl<Ty, P> SDF<P> for Ty where Ty: AsRef<dyn Draw<P, RgbaImage>> { fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P { self.as_ref().sdf(pixel) } }
impl<Ty, P> BoundingBox<P> for Ty where Ty: AsRef<dyn Draw<P, RgbaImage>> { fn bounding_box(&self) -> Box2D<P, WorldSpace> { self.as_ref().bounding_box() } }

impl <Cutie, P: Float> Draw<P, RgbaImage> for Texture<Cutie, Rgba<u8>>
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
        P: Float + AsPrimitive<f64>
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

/// `F: Fn(v: Point2D) -> Rgba<u8>`
/// where `v` is in normalized texture coordinates.
impl <Cutie, F, P> Draw<P, RgbaImage> for Texture<Cutie, F>
  where Cutie: Shape<P>,
        F: Fn(Point2D<P, WorldSpace>) -> Rgba<u8>,
        P: Float + AsPrimitive<f64>
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
        P: Float + AsPrimitive<f64>
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
        P: Float + AsPrimitive<f64>
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
