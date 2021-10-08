#![allow(non_snake_case)]
use {
  crate::{
    solver::{Argmax2D, GradientDescent, DistPoint, gradient_descent::LineSearch, quadtree::Quadtree},
    geometry::{
      self, BoundingBox, Shape,
      PixelSpace, WorldSpace,
      Translation, Rotation, Scale,
      Circle
    },
    sdf::SDF
  },
  euclid::{Box2D, Point2D, Size2D, Vector2D as V2},
  image::{
    ImageBuffer, Luma, Rgba, Pixel, RgbaImage
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

// rust doesn't support trait upcast yet
impl<P, B> SDF<P> for Box<dyn Draw<P, B>> { fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P { self.deref().sdf(pixel) } }
impl<P, B> BoundingBox<P> for Box<dyn Draw<P, B>> { fn bounding_box(&self) -> Box2D<P, WorldSpace> { self.deref().bounding_box() } }
impl<P, B> SDF<P> for Box<dyn DrawSync<P, B>> { fn sdf(&self, pixel: Point2D<P, WorldSpace>) -> P { self.deref().sdf(pixel) } }
impl<P, B> BoundingBox<P> for Box<dyn DrawSync<P, B>> { fn bounding_box(&self) -> Box2D<P, WorldSpace> { self.deref().bounding_box() } }

static MSG: &str = "Draw is only implemented for Texture";

impl <B, S, P> Draw<P, B> for Translation<S, P> where Translation<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!(MSG) } }
impl <B, S, P> Draw<P, B> for Rotation<S, P> where Rotation<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!(MSG) } }
impl <B, S, P> Draw<P, B> for Scale<S, P> where Scale<S, P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!(MSG) } }

impl <B, P> Draw<P, B> for geometry::Line<P> where geometry::Line<P>: Shape<P> {
  fn draw(&self, _: &mut B) { unreachable!(MSG) } }
impl <B, P, U> Draw<P, B> for geometry::Polygon<U> where P: num_traits::Float, U: AsRef<[Point2D<P, WorldSpace>]> {
  fn draw(&self, _: &mut B) { unreachable!(MSG) } }

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

impl <T> GradientDescent<T, f64>
  where GradientDescent<T, f64>: LineSearch<f64> {

  fn display_vec (
    p1: Point2D<f64, WorldSpace>,
    p2: Point2D<f64, WorldSpace>,
    alpha: f64,
    Δp: f64,
    img: &mut RgbaImage
  ) {
    let shapes: Vec<Box<dyn Draw<_, RgbaImage>>> = vec![
      Box::new(geometry::Line {
        a: p1,
        b: p2,
        thickness: Δp
      }),
      Box::new(Circle
        .scale(Δp * 3.0)
        .translate(p2.to_vector()))
    ];
    shapes.into_iter().for_each(|s| s
      .texture(|_| Rgba([255, 0, 0, (alpha * 255.0) as u8]))
      .draw(img));
  }

  pub fn display_sdf<'a>(&self, image: &'a mut RgbaImage, brightness: f64, gradient_marks: Option<u64>) -> &'a mut RgbaImage {
    let Δp = 1.0 / image.width() as f64;

    display_sdf(|p| self.sample_sdf(p), image, brightness);

    // gradiend vector field
    let grid_count = gradient_marks.unwrap_or(0);
    itertools::iproduct!(0..grid_count, 0..grid_count)
      .map(|(x, y)| (Point2D::new(x, y).to_f64() + V2::splat(0.5))
        / grid_count as f64)
      .for_each(|p| {
        let grad = self.Δf(p) * Δp * 20.0;
        Self::display_vec(p, p + grad, 1.0, Δp, image);
      });

    image
  }

  pub fn trajectory_animation<'a>(&self, image: &'a mut RgbaImage, brightness: f64, f: impl Fn(usize, RgbaImage)) -> &'a mut RgbaImage {
    let image = self.display_sdf(image, brightness, None);
    let resolution = image.width();
    let Δp = 1.0 / resolution as f64;

    let trajectory_trail = move |p1, p2, p3, img: &mut _| {
      if let (Some(p1), Some(p2)) = (p1, p2) {
        Self::display_vec(p1, p2, 0.25, Δp, img);
      }
      if let (Some(p2), Some(p3)) = (p2, p3) {
        Self::display_vec(p2, p3, 0.5, Δp, img);
      }
      if let Some(p3) = p3 {
        Self::display_vec(p3, p3, 1.0, Δp, img);
      }
    };

    let grid_count = 9;
    let trajectories = itertools::iproduct!(0..grid_count, 0..grid_count)
      .map(|(x, y)| (Point2D::new(x, y).to_f64() + V2::splat(0.5))
        / grid_count as f64)
      .map(|p| self.trajectory(p))
      .collect::<Vec<_>>();

    let max_len = trajectories.iter().map(|x| x.len()).max().unwrap_or(0);
    (0isize..max_len as isize).for_each(|i| {
      let mut img = image.clone();
      trajectories.iter()
        .for_each(|trajectory| {
          trajectory_trail(
            trajectory.get((i - 2).max(0) as usize).cloned(),
            trajectory.get((i - 1).max(0) as usize).cloned(),
            trajectory.get(i as usize).cloned(),
            &mut img
          );
        });
      f(i as usize, img);
    });
    image
  }
}

impl <T> Quadtree<T> {
  pub fn draw_layout(&self, image: &mut RgbaImage) {
    use geometry::Line;

    let px = 1.0 / image.width() as f64;
    self.traverse(&mut |node| {
      if node.children.is_some() { return Ok(()) };

      let rect = node.rect;
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
  }
}
