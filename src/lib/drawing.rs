use std::{
  collections::HashMap,
  thread
};

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba, Pixel};
use image::imageops::FilterType;
use plotters::prelude::*;

use crate::{
  error::{ErrorKind::NoneError, Result},
  geometry::{Circle, Point, Rect, TLBR},
  sdf::SDF,
  argmax2d::{ArgmaxResult, Argmax2D}
};

/// draw a set of circles
pub fn draw_circles(
  path: &str,
  circles: impl Iterator<Item = Circle>,
  resolution: Point<u32>
) -> Result<()> {
  let img = BitMapBackend::new(
    &path,
    (resolution.x, resolution.y)
  ).into_drawing_area();

  for circle in circles {
    img.draw(&plotters::element::Circle::new(
      ((circle.xy.x * resolution.x as f32) as i32, (circle.xy.y * resolution.y as f32) as i32),
      (circle.r * resolution.x as f32) as u32, //?
      Into::<ShapeStyle>::into(&RGBColor(0xff, 0xff, 0xff)).filled()
    )).ok();
  }
  Ok(())
}

/// draw a set of circles, random colors
pub fn draw_circles_rng(
  path: String,
  data: Vec<Circle>,
  resolution: Point<u32>,
  rng: &mut (impl rand::Rng + ?Sized)
) -> Result<()> {
  let img = BitMapBackend::new(
    &path,
    (resolution.x, resolution.y)
  ).into_drawing_area();

  for circle in data {
    let color = rng.gen_range(0x90..=0xff);
    img.draw(&plotters::element::Circle::new(
      ((circle.xy.x * resolution.x as f32) as i32, (circle.xy.y * resolution.y as f32) as i32),
      (circle.r * resolution.y as f32) as u32, //?
      Into::<ShapeStyle>::into(&RGBColor(color, color, color)).filled()
    )).ok();
  }
  Ok(())
}

/// draw image in each circle
pub fn draw_img(
  circle: Circle,
  file: std::path::PathBuf,
  framebuffer: &mut ImageBuffer<Rgba<u8>, Vec<u8>>
) -> Result<()> {
  let resolution = Point::<u32> { x: framebuffer.width(), y: framebuffer.height() };

  fn image_proc(circle_r: f32, img_path: String) -> DynamicImage {
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
    img = img
      .crop_imm((w - size) / 2, (h - size) / 2, size, size)
      .resize_exact((circle_r * 2.0).ceil() as u32, (circle_r * 2.0).ceil() as u32, FilterType::Triangle);
    img
      .as_mut_rgba8().unwrap()
      .enumerate_pixels_mut()
      .for_each( |(x, y, pixel)| {
        let sdf = Circle {
          xy: Point {x: circle_r, y: circle_r }, r: circle_r
        }.sdf( Point {x: x as f32, y: y as f32}) / circle_r;
        let alpha = (1.0 - (1.0 - sdf.abs()).powf(circle_r / 2.0)) * ((sdf < 0.0) as u8 as f32);
        pixel.0[3] = (alpha * 255.0) as u8;
      });
    img
  }

  // for sprite usecases
  fn _image_proc_cached(
    circle_r: f32,
    img_path: &str,
    cache: &mut HashMap::<String, DynamicImage>
  ) -> DynamicImage {
    let img = cache.entry(img_path.to_owned()).or_insert_with(||
      image::DynamicImage::ImageRgba8(
        image::open(img_path)
          .unwrap_or(
            image::DynamicImage::ImageRgba8(
              image::ImageBuffer::from_fn(1, 1, |_, _|
                image::Rgba([0xff, 0xff, 0xff, 0]))
            )
          )
          .into_rgba8())
    );
    let (w, h) = img.dimensions();
    let size = w.min(h);
    let img = img
      .crop_imm((w - size) / 2, (h - size) / 2, size, size)
      .resize_exact((circle_r * 2.0).ceil() as u32, (circle_r * 2.0).ceil() as u32, FilterType::Triangle);
    img
  }

  let img = image_proc(
    circle.r * resolution.x as f32, //?
    file.to_string_lossy().to_string()
  );
  let circle = Circle {
    xy: Point { x: circle.xy.x * resolution.x as f32, y: circle.xy.y * resolution.y as f32 },
    r: circle.r * resolution.x as f32 //?
  };
  let coord = { Rect {
    center: Point { x: circle.xy.x, y: circle.xy.y },
    size: circle.r * 2.0
  }}.into(): TLBR<f32>;

  image::imageops::overlay(framebuffer, &img, coord.tl.x as u32, coord.tl.y as u32);
  Ok(())
}

/// draw image in each circle, parallel
pub fn draw_img_parallel(
  path: &str,
  circles: impl Iterator<Item = Circle>,
  files: impl Iterator<Item = std::path::PathBuf>,
  resolution: Point<u32>,
  num_threads: usize
) -> Result<()> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let mut draw_data = circles
    .zip(files)
    .enumerate()
    .collect::<Vec<_>>();
  // will distribute the load between threads [statistically] evenly
  draw_data.shuffle(&mut rng);

  let num_threads = num_threads.min(draw_data.len());

  let draw_data_chunks = draw_data
    .chunks((draw_data.len() as f32 / num_threads as f32).ceil() as usize)
    .map(|chunk| chunk.to_vec())
    .collect::<Vec<_>>();

  if draw_data_chunks.len() != num_threads {
    error_chain::bail!("chunks are unsatisfied");
  }

  let partial_buffers = draw_data_chunks.into_iter().map(|chunk| {
    thread::spawn( move || {
      let mut framebuffer: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::new(resolution.x, resolution.y);

      chunk.into_iter()
        .for_each(|(i, (circle, file))| {
          println!("#{}: {:?} -> \"{}\"", i, Circle {
              xy: circle.xy * resolution.x as f32,
              r: circle.r * resolution.x as f32
            },
            file.file_name()
              .map(|x| x.to_string_lossy())
              .unwrap_or("?".into())
          );
          draw_img(circle, file, &mut framebuffer).ok();
        });

      framebuffer
    })
  }).collect::<Vec<_>>() // thread handles
    .into_iter()
    .map(|thread| thread.join().unwrap())
    .collect::<Vec<_>>();

  let mut final_buffer = partial_buffers.get(0).cloned().ok_or(NoneError)?;

  // merge partial buffers
  partial_buffers
    .into_iter()
    .skip(1)
    .for_each(|buffer|
      image::imageops::overlay(&mut final_buffer, &buffer, 0, 0)
    );

  final_buffer.save(path)?;

  Ok(())
}

/// draw image in each circle, parallel. faster, will crash if two circles intersect.
pub fn draw_img_parallel_unsafe(
  path: &str,
  circles: impl Iterator<Item = Circle>,
  files: impl Iterator<Item = std::path::PathBuf>,
  resolution: Point<u32>
) -> Result<()> {
  use rayon::prelude::*;

  let framebuffer: ImageBuffer<image::Rgba<u8>, _> =
    ImageBuffer::new(resolution.x, resolution.y);

  let draw_data = circles.zip(files).enumerate().collect::<Vec<_>>();
  draw_data.into_par_iter()
    .for_each(|(i, (circle, file))| {
      println!("#{}: {:?} -> \"{}\"", i, Circle {
          xy: circle.xy * resolution.x as f32,
          r: circle.r * resolution.x as f32
        },
        file.file_name()
          .map(|x| x.to_string_lossy())
          .unwrap_or("?".into())
      );

      draw_img(circle, file,
        // safe as long as no circles intersect
        unsafe { &mut *(&framebuffer as *const _ as *mut _) }
      ).ok();
  });

  framebuffer.save(path)?;
  Ok(())
}

pub fn display_argmax_debug(argmax: &Argmax2D) -> image::RgbImage {
  let mut image = image::ImageBuffer::<image::Rgb<u8>, _>::new(
    argmax.resolution as u32,
    argmax.resolution as u32
  );
  let max_dist = argmax.find_max().distance;
  argmax.pixels().for_each(|ArgmaxResult { distance, point }| {
    let color = image::Luma::from([(distance / max_dist * 255.0) as u8]);
    *image.get_pixel_mut(point.x as u32, point.y as u32) = color.to_rgb();
  });
  image
}