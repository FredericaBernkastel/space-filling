use {
  std::{collections::HashMap, thread},
  image::{
    DynamicImage, GenericImageView, ImageBuffer, Rgba, Pixel,
    imageops::FilterType
  },
  plotters::prelude::*,
  euclid::{Point2D, Box2D},

  crate::{
    error::{self, ErrorKind::NoneError, Result},
    geometry::{Circle, WorldSpace, PixelSpace},
    sdf::SDF,
    argmax2d::{ArgmaxResult, Argmax2D}
  }
};

/// draw a set of circles
pub fn draw_circles(
  circles: impl Iterator<Item = Circle<f32, WorldSpace>>,
  resolution: Point2D<u32, PixelSpace>
) -> image::RgbImage {
  let mut image = ImageBuffer::<image::Rgb<u8>, _>::new(
    resolution.x as u32,
    resolution.y as u32
  );
  let canvas = BitMapBackend::with_buffer(
    image.as_mut(), resolution.to_tuple()
  ).into_drawing_area();

  for circle in circles {
    canvas.draw(&plotters::element::Circle::new(
      (circle.xy * resolution.x as f32).to_i32().to_tuple(),
      (circle.r * resolution.x as f32) as u32, //?
      Into::<ShapeStyle>::into(&RGBColor(0xff, 0xff, 0xff)).filled()
    )).ok();
  }
  drop(canvas);
  image
}

/// draw image in a circle
pub fn draw_img(
  circle: Circle<f32, WorldSpace>,
  file: std::path::PathBuf,
  framebuffer: &mut image::RgbaImage
) -> Result<()> {
  let resolution: Point2D<_, PixelSpace> = framebuffer.dimensions().into();

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
        let sdf = Circle::<f32, WorldSpace> {
          xy: [circle_r, circle_r].into(), r: circle_r
        }.sdf( Point2D::from([x, y]).to_f32()) / circle_r;
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
  let circle = circle * resolution.x as f32;
  let coord = Box2D {
    min: (circle.xy.to_vector() - [circle.r, circle.r].into()).to_point(),
    max: Point2D::from([circle.r, circle.r]) * 2.0
  }.to_u32();

  image::imageops::overlay(framebuffer, &img, coord.min.x, coord.min.y);
  Ok(())
}

/// draw image in each circle, parallel
pub fn draw_img_parallel(
  circles: impl Iterator<Item = Circle<f32, WorldSpace>>,
  files: impl Iterator<Item = std::path::PathBuf>,
  resolution: Point2D<u32, PixelSpace>,
  num_threads: usize
) -> Result<image::RgbaImage> {
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
      let mut framebuffer: ImageBuffer<Rgba<u8>, _> =
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
          draw_img(circle, file, &mut framebuffer)
            .map_err(|e| error::display(&e)).ok();
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

  Ok(final_buffer)
}

/// draw image in each circle, parallel. faster, will crash if two circles intersect.
pub fn draw_img_parallel_unsafe(
  circles: impl Iterator<Item = Circle<f32, WorldSpace>>,
  files: impl Iterator<Item = std::path::PathBuf>,
  resolution: Point2D<u32, PixelSpace>
) -> Result<image::RgbaImage> {
  use rayon::prelude::*;

  let framebuffer: ImageBuffer<Rgba<u8>, _> =
    ImageBuffer::new(resolution.x, resolution.y);

  let draw_data = circles.zip(files).enumerate().collect::<Vec<_>>();
  draw_data.into_par_iter()
    .for_each(|(i, (circle, file))| {
      println!("#{}: {:?} -> \"{}\"", i, circle * resolution.x as f32,
        file.file_name()
          .map(|x| x.to_string_lossy())
          .unwrap_or("?".into())
      );

      draw_img(circle, file,
        // safe as long as no circles intersect
        unsafe { &mut *(&framebuffer as *const _ as *mut _) }
      ).map_err(|e| error::display(&e)).ok();
  });

  Ok(framebuffer)
}

pub fn display_argmax_debug(argmax: &Argmax2D) -> image::RgbImage {
  let mut image = ImageBuffer::<image::Rgb<u8>, _>::new(
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