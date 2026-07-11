#![allow(non_snake_case)]

use num_traits::Float;
use {
  std::{sync::Arc, ops::Fn},
  image::{
    DynamicImage, GenericImageView, Pixel, Rgba, RgbaImage,
    imageops::FilterType
  },
  nalgebra::Scalar,
  num_traits::AsPrimitive,
  crate::{
    drawing::{Draw, Shape, Texture, rescale_bounding_box},
    geometry::{BoundingBox, Aabb, P2, V2},
    sdf::SDF
  }
};

impl<Ty, P: Scalar> SDF<P, 2> for Ty where Ty: AsRef<dyn Draw<P, RgbaImage>> { fn sdf(&self, pixel: P2<P>) -> P { self.as_ref().sdf(pixel) } }
impl<Ty, P: Scalar> BoundingBox<P, 2> for Ty where Ty: AsRef<dyn Draw<P, RgbaImage>> { fn bounding_box(&self) -> Aabb<P, 2> { self.as_ref().bounding_box() } }

impl <Cutie, P: Float + Scalar> Draw<P, RgbaImage> for Texture<Cutie, Rgba<u8>>
  where Cutie: Shape<P, 2> + Clone,
        P: AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    self.shape.clone()
      .texture(|_| self.texture)
      .draw(image);
  }
}

impl <'a, Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, &'a DynamicImage>
  where Cutie: Shape<P, 2>,
        P: Float + Scalar + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    let resolution = V2::from(<[u32; 2]>::from(image.dimensions()));
    let bb = self.shape.bounding_box();
    let (bounding_box, offset, min_side) = rescale_bounding_box(
      Aabb::new(bb.min.map(|x| x.as_()), bb.max.map(|x| x.as_())),
      resolution);
    let bounding_box = match bounding_box {
      Some(x) => x,
      None => return
    };
    let Δp = 1.0 / min_side;
    let tex = rescale_texture(self.texture, bounding_box.size());

    itertools::iproduct!(
      bounding_box.min.y..bounding_box.max.y,
      bounding_box.min.x..bounding_box.max.x
    ).for_each(|(y, x)| {
        let pixel_world = (P2::new(x as f64, y as f64) - offset) / min_side;
        let tex_px = tex.get_pixel(x - bounding_box.min.x, y - bounding_box.min.y);

        let sdf = self.sdf(pixel_world.map(|v| P::from(v).unwrap())).as_();
        let pixel = image.get_pixel_mut(x, y);
        *pixel = sdf_overlay_aa(sdf, Δp, *pixel, tex_px);
      });
  }
}

/// `F: Fn(v: P2) -> Rgba<u8>`
/// where `v` is in normalized texture coordinates.
impl <Cutie, F, P> Draw<P, RgbaImage> for Texture<Cutie, F>
  where Cutie: Shape<P, 2>,
        F: Fn(P2<P>) -> Rgba<u8>,
        P: Float + Scalar + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    let resolution = V2::from(<[u32; 2]>::from(image.dimensions()));
    let bb = self.bounding_box();
    let (bounding_box, offset, min_side) = rescale_bounding_box(
      Aabb::new(bb.min.map(|x| x.as_()), bb.max.map(|x| x.as_())),
      resolution);
    let bounding_box = match bounding_box {
      Some(x) => x,
      None => return // bounding box has no intersection with screen at all
    };
    let Δp = 1.0 / min_side;
    let size = bounding_box.size();
    let tex_scale = size.x.min(size.y) as f64;

    itertools::iproduct!(
      bounding_box.min.y..bounding_box.max.y,
      bounding_box.min.x..bounding_box.max.x
    ).for_each(|(y, x)| {
        let pixel_world = (P2::new(x as f64, y as f64) - offset) / min_side;
        let sdf = self.sdf(pixel_world.map(|v| P::from(v).unwrap())).as_();

        let tex_px = P2::new(
          (x - bounding_box.min.x) as f64 / tex_scale,
          (y - bounding_box.min.y) as f64 / tex_scale,
        );
        let tex_px = (self.texture)(tex_px.map(|v| P::from(v).unwrap()));

        let pixel = image.get_pixel_mut(x, y);
        *pixel = sdf_overlay_aa(sdf, Δp, *pixel, tex_px);
      });
  }
}

impl <Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, DynamicImage>
  where Cutie: Shape<P, 2> + Clone,
        P: Float + Scalar + AsPrimitive<f64>
{
  fn draw(&self, image: &mut RgbaImage) {
    Texture {
      shape: self.shape.clone(),
      texture: &self.texture
    }.draw(image)
  }
}

impl <Cutie, P> Draw<P, RgbaImage> for Texture<Cutie, Arc<DynamicImage>>
  where Cutie: Shape<P, 2> + Clone,
        P: Float + Scalar + AsPrimitive<f64>
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
fn rescale_texture(texture: &DynamicImage, size: V2<u32>) -> DynamicImage {
  let tex_size = V2::new(texture.dimensions().0 as f32, texture.dimensions().1 as f32);
  let size_f = V2::new(size.x as f32, size.y as f32);
  let scaling_factor = (tex_size.x / size_f.x).min(tex_size.y / size_f.y);
  let bound_inner = size_f * scaling_factor;
  let origin = (tex_size - bound_inner) / 2.0;
  texture.crop_imm(
    origin.x as u32,
    origin.y as u32,
    bound_inner.x as u32,
    bound_inner.y as u32
  ).resize_exact(size.x, size.y, FilterType::Triangle)
}

fn sdf_overlay_aa(sdf: f64, Δp: f64, mut col1: Rgba<u8>, mut col2: Rgba<u8>) -> Rgba<u8> {
  let Δf = (0.5 * Δp - sdf) // antialias
    .clamp(0.0, Δp);
  let alpha = if sdf == 0.0 { 1.0 } else { Δf / Δp };
  // overlay blending with premultiplied alpha
  col2.0[3] = ((col2.0[3] as f64) * alpha) as u8;
  col1.blend(&col2);
  col1
}
