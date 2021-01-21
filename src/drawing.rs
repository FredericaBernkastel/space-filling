use plotters::prelude::*;
use crate::{
  WORLD_SIZE,
  IMG_SCALE,
  quadtree::{sdf},
  lib::{Point, Result},
  quadtree::Quadtree,
};
use image::{GenericImageView, DynamicImage, ImageBuffer, Rgba};
use image::imageops::FilterType;
use crate::quadtree::sdf::SDF;

/// draw a set of circles
pub fn exec(data: Vec<sdf::Circle>, path: String) -> Result<()> {
  let img = BitMapBackend::new(
    &path,
    ((WORLD_SIZE * IMG_SCALE) as u32, (WORLD_SIZE * IMG_SCALE) as u32)
  ).into_drawing_area();

  for circle in data {
    img.draw(&Circle::new(
      ((circle.xy.x * IMG_SCALE) as i32, (circle.xy.y * IMG_SCALE) as i32),
      (circle.r * IMG_SCALE) as u32,
      Into::<ShapeStyle>::into(&RGBColor(0xff, 0xff, 0xff)).filled()
    )).ok()?;
  }
  Ok(())
}

pub fn exec_rng(data: Vec<sdf::Circle>, path: String, rng: &mut (impl rand::Rng + ?Sized)) -> Result<()> {
  let img = BitMapBackend::new(
    &path,
    ((WORLD_SIZE * IMG_SCALE) as u32, (WORLD_SIZE * IMG_SCALE) as u32)
  ).into_drawing_area();

  for circle in data {
    let color = rng.gen_range(0x90..=0xff);
    img.draw(&Circle::new(
      ((circle.xy.x * IMG_SCALE) as i32, (circle.xy.y * IMG_SCALE) as i32),
      (circle.r * IMG_SCALE) as u32,
      Into::<ShapeStyle>::into(&RGBColor(color, color, color)).filled()
    )).ok()?;
  }
  Ok(())
}

pub fn exec_img(
  data: impl Iterator<Item = (sdf::Circle, std::path::PathBuf)>,
  framebuffer: &mut ImageBuffer<Rgba<u8>, Vec<u8>>
) -> Result<()> {

  fn image_proc(circle: sdf::Circle, img_path: String) -> DynamicImage {
    let mut img =
      image::DynamicImage::ImageRgba8(
        image::open(img_path)
          .unwrap_or(
            image::DynamicImage::ImageRgba8(
              image::ImageBuffer::from_fn(1, 1, |_, _|
                image::Rgba([0xff, 0xff, 0xff, 0]))
            )
          )
          .into_rgba8());
    let (w, h) = img.dimensions();
    let size = w.min(h);
    let radii = circle.r * IMG_SCALE;
    img = img
      .crop_imm((w - size) / 2, (h - size) / 2, size, size)
      .resize_exact((radii * 2.0).ceil() as u32, (radii * 2.0).ceil() as u32, FilterType::Triangle);
    img
      .as_mut_rgba8().unwrap()
      .enumerate_pixels_mut()
      .for_each( |(x, y, pixel)| {
        let sdf = sdf::Circle {
          xy: Point {x: radii, y: radii}, r: radii
        }.sdf( Point {x: x as f32, y: y as f32}) / radii;
        let alpha = (1.0 - (1.0 - sdf.abs()).powf(radii / 2.0)) * ((sdf < 0.0) as u8 as f32);
        pixel.0[3] = (alpha * 255.0) as u8;
      });
    img
  }

  data.map(|(circle, file)| (
      circle,
      file.file_name()
        .map(|x| x.to_string_lossy())
        .unwrap_or("?".into())
        .to_string(),
      image_proc(circle, file.to_string_lossy().into()))
    )
    .enumerate()
    .for_each(|(i, (circle, filename, img))| {
      let circle = sdf::Circle {
        xy: Point { x: circle.xy.x * IMG_SCALE, y: circle.xy.y * IMG_SCALE },
        r: circle.r * IMG_SCALE
      };
      let coord = { sdf::Rect {
        center: Point { x: circle.xy.x, y: circle.xy.y },
        size: circle.r * 2.0
      }}.into(): sdf::TLBR;
      println!("#{}: {:?} -> \"{}\"", i, circle, filename);
      image::imageops::overlay(framebuffer, &img, coord.tl.x as u32, coord.tl.y as u32)
    });

  Ok(())
}

pub fn sdf_test() -> Result<()> {
  image::ImageBuffer::from_fn(512, 512, |x, y| {
    let rect = sdf::Rect { center: Point { x: 256.0, y: 256.0 }, size: 64.0 };
    let sample = Point { x: x as f32, y: y as f32 };

    /*if c1.sdf(sample) > 0.0 {
      image::Luma([0u8])
    } else {
      image::Luma([255u8])
    }*/
    image::Luma([
      rect.sdf(sample).abs() as u8
    ])
  }).save("out.png")?;
  open::that("out.png")?;
  Ok(())
}

/// draw the quadree layout
pub fn tree_test<'a>(tree: &Quadtree, path: &'a str) -> Result<BitMapBackend<'a>> {
  let mut img = BitMapBackend::new(
    path,
    ((WORLD_SIZE * IMG_SCALE) as u32 + 1, (WORLD_SIZE * IMG_SCALE) as u32 + 1)
  );

  tree.traverse(&mut |tree| {
    let (rect, depth) : (sdf::TLBR, _) = (tree.rect.into(), tree.depth);
    let color =
      if tree.data { RGBColor(255, (255.0 / 1.5f32.powf((11 - depth) as f32)) as u8, 0) } else { RGBColor(32, 32, 255) }
        .mix(if tree.data {
          0.0282475249 / 0.7f64.powf(depth as f64)
          //4.0 / 1.5f64.powf(depth as f64)
        } else {
          1.0 / 1.6f64.powf(depth as f64)
        }
    );
    img.draw_rect(
      (
        (rect.tl.x * IMG_SCALE) as i32,
        (rect.tl.y * IMG_SCALE) as i32),
      (
        (rect.br.x * IMG_SCALE) as i32,
        (rect.br.y * IMG_SCALE) as i32),
      &color,
      tree.data
      ).ok()?;
    Ok(())
  })?;

  Ok(img)
}