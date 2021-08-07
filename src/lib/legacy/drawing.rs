use plotters::drawing::IntoDrawingArea;
use plotters::prelude::{BitMapBackend, Color, DrawingBackend, RED, RGBColor};

use crate::error;
use crate::geometry::{Point, Rect, TLBR};
use crate::legacy::quadtree::Quadtree;
use crate::legacy::argmax::{Argmax, ArgmaxResult};
use crate::sdf::SDF;

pub fn sdf_test() -> error::Result<()> {
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
pub fn tree_display<'a, T: >(path: &'a str, tree: &Quadtree<T>, resolution: Point<u32>) -> error::Result<BitMapBackend<'a>> {
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
      ).ok();
    Ok(())
  })?;

  Ok(img)
}

/// draw the quadree argmax values
pub fn tree_display_argmax<'a>(
  path: &'a str,
  root: &Quadtree<ArgmaxResult<f32>>,
  resolution: Point<u32>
) -> error::Result<BitMapBackend<'a>> {
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
      ).ok();
      img.draw_circle(
        ((tree.data.point.x * resolution.x as f32) as i32, (tree.data.point.y * resolution.y as f32) as i32),
        2, //?
        &RGBColor(0xff, 0x00, 0x00).mix(distance_color as f64 / 128.0),
        true
      ).ok();
      img.draw_line(
        ((tree.rect.center.x * resolution.x as f32) as i32, (tree.rect.center.y * resolution.y as f32) as i32),
        ((tree.data.point.x * resolution.x as f32) as i32, (tree.data.point.y * resolution.y as f32) as i32),
        &RGBColor(0xff, 0x00, 0x00).mix(distance_color as f64 / 1.5 / 255.0)
      ).ok();
    }
    Ok(())
  })?;

  Ok(img)
}

pub fn display_debug_convolution(output_file: &str, argmax: &Argmax, point: Option<Point<f32>>) -> error::Result<()> {
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
    )).ok();
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
