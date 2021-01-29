use std::thread;
use plotters::prelude::*;
use crate::{
  sdf::SDF,
  geometry::{Point, Circle, Rect, TLBR},
  error::Result,
  quadtree::Quadtree,
  argmax::*
};
use image::{GenericImageView, DynamicImage, ImageBuffer, Rgba};
use image::imageops::FilterType;

/// draw a set of circles
pub fn exec(
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
    )).ok()?;
  }
  Ok(())
}

pub fn exec_rng(
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
    )).ok()?;
  }
  Ok(())
}

pub fn exec_img(
  data: impl Iterator<Item = (Circle, std::path::PathBuf)>,
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

  data.map(|(circle, file)| (
      circle,
      file.file_name()
        .map(|x| x.to_string_lossy())
        .unwrap_or("?".into())
        .to_string(),
      image_proc(
        circle.r * resolution.x as f32, //?
        file.to_string_lossy().into())
      )
    )
    .enumerate()
    .for_each(|(i, (circle, filename, img))| {
      let circle = Circle {
        xy: Point { x: circle.xy.x * resolution.x as f32, y: circle.xy.y * resolution.y as f32 },
        r: circle.r * resolution.x as f32 //?
      };
      let coord = { Rect {
        center: Point { x: circle.xy.x, y: circle.xy.y },
        size: circle.r * 2.0
      }}.into(): TLBR<f32>;
      println!("#{}: {:?} -> \"{}\"", i, circle, filename);
      image::imageops::overlay(framebuffer, &img, coord.tl.x as u32, coord.tl.y as u32)
    });

  Ok(())
}

pub fn exec_img_parallel(
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
    thread::spawn(move || {
      let mut framebuffer: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::new(resolution.x, resolution.y);

      exec_img(chunk.into_iter(), &mut framebuffer).ok();

      framebuffer
    })
  }).collect::<Vec<_>>() // thread handles
    .into_iter()
    .map(|thread| thread.join().unwrap())
    .collect::<Vec<_>>();

  let mut final_buffer = partial_buffers.get(0).cloned()?;

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

pub fn sdf_test() -> Result<()> {
  image::ImageBuffer::from_fn(512, 512, |x, y| {
    let rect = Rect { center: Point { x: 256.0, y: 256.0 }, size: 64.0 };
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
pub fn tree_display<'a, T: >(path: &'a str, tree: &Quadtree<T>, resolution: Point<u32>) -> Result<BitMapBackend<'a>> {
  let mut img = BitMapBackend::new(
    path,
    (resolution.x + 1, resolution.y + 1)
  );

  let scale = tree.rect.size;

  tree.traverse(&mut |tree| {
    let rect: TLBR<f32> = tree.rect.into();
    let color =
      if tree.is_inside {
        RGBColor(255, (255.0 / 1.5f32.powf((tree.max_depth - tree.depth) as f32)) as u8, 0)
      } else {
        RGBColor(32, 32, 255)
      }
        .mix(if tree.is_inside {
          0.0282475249 / 0.7f64.powf(tree.depth as f64)
          //4.0 / 1.5f64.powf(depth as f64)
        } else {
          1.0 / 1.6f64.powf(tree.depth as f64)
        }
    );
    let rect = TLBR {
      tl: rect.tl / scale * resolution.x as f32,
      br: rect.br / scale * resolution.y as f32
    };
    img.draw_rect(
      (rect.tl.x as i32, rect.tl.y as i32),
      (rect.br.x as i32, rect.br.y as i32),
      &color,
      tree.is_inside
      ).ok()?;
    Ok(())
  })?;

  Ok(img)
}

/// draw the quadree argmax values
pub fn tree_display_argmax<'a>(
  path: &'a str,
  root: &Quadtree<crate::argmax::ArgmaxResult<f32>>,
  resolution: Point<u32>
) -> Result<BitMapBackend<'a>> {
  let mut img = BitMapBackend::new(
    path,
    (resolution.x + 1, resolution.y + 1)
  );

  let scale = root.rect.size;

  root.traverse(&mut |tree| {
    // is leaf
    if tree.children.is_none() {
      let (rect, _) : (TLBR<f32>, _) = (tree.rect.into(), tree.depth);

      let distance_color = (tree.data.distance / root.data.distance * 255.0) as u8;

      let rect = TLBR {
        tl: rect.tl / scale * resolution.x as f32,
        br: rect.br / scale * resolution.y as f32
      };
      img.draw_rect(
        (rect.tl.x as i32, rect.tl.y as i32),
        (rect.br.x as i32, rect.br.y as i32),
        &RGBColor(distance_color, distance_color, distance_color),
        true
      ).ok()?;
      img.draw_circle(
        ((tree.data.point.x * resolution.x as f32) as i32, (tree.data.point.y * resolution.y as f32) as i32),
        2, //?
        &RGBColor(0xff, 0x00, 0x00).mix(distance_color as f64 / 128.0),
        true
      ).ok()?;
      img.draw_line(
        ((tree.rect.center.x * resolution.x as f32) as i32, (tree.rect.center.y * resolution.y as f32) as i32),
        ((tree.data.point.x * resolution.x as f32) as i32, (tree.data.point.y * resolution.y as f32) as i32),
        &RGBColor(0xff, 0x00, 0x00).mix(distance_color as f64 / 1.5 / 255.0)
      ).ok()?;
    }
    Ok(())
  })?;

  Ok(img)
}

pub fn display_debug_convolution(output_file: &str, argmax: &Argmax, point: Option<Point<f32>>) -> Result<()> {
  let img = BitMapBackend::new(output_file, argmax.dist_map.dimensions());
  let img = img.into_drawing_area();

  // dist map
  argmax.dist_map
    .enumerate_pixels()
    .for_each(|(x, y, pixel)| {
      let color = pixel[0] * 4.0 * 255.0;
      let color = if color > 0.0 {
        RGBColor(color.abs() as u8, color.abs() as u8, color.abs() as u8)
      } else {
        RGBColor(color.abs() as u8, 8, 8)
      };
      img.draw_pixel((x as i32, y as i32), &color).ok();
    });

  // argmax
  if let Some(point) = point {
    img.draw(&plotters::element::Circle::new(
      ((point.x * argmax.dist_map.width() as f32) as i32, (point.y * argmax.dist_map.height() as f32) as i32),
      8,
      RGBColor(8, 8, 0xff).filled(),
    )).ok()?;
  }

  // convolution vector
  let argmax_ret = argmax.find_max_convolution();
  argmax.convolution_vector.iter()
    .for_each(|row_argmax| {
      let (x, y) = (row_argmax.point.x as i32, row_argmax.point.y as i32);
      img.draw_pixel(
        (x, y),
        &RED.mix(1.0)
      ).ok();
      img.draw(
        &plotters::element::Polygon::new(
          vec![(x, y), (img.dim_in_pixel().0 as i32 - 1, y)],
          RGBColor(8, 8, 0xff)
            .mix((row_argmax.distance as f64 / argmax_ret.distance as f64).powf(8.0) * 0.5)
            .stroke_width(1)
        )
      ).ok();
    });

  Ok(())
}