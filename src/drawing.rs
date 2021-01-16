use plotters::prelude::*;
use crate::{
  WORLD_SIZE,
  quadtree::{sdf},
  lib::{Point, Result},
  quadtree::Quadtree,
};

/// final image resolution is `WORLD_SIZE` * `IMG_SCALE`
const IMG_SCALE: f32 = 4.0;

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
      Into::<ShapeStyle>::into(&RGBColor(0xe0, 0xe0, 0xe0)).filled()
    )).ok()?;
  }
  Ok(())
}

fn sdf_test() -> Result<()> {
  image::ImageBuffer::from_fn(512, 512, |x, y| {
    if sdf::circle(Point { x: x as f32, y: y as f32 }, sdf::Circle {
      xy: Point { x: 256.0, y: 256.0 },
      r: 128.0,
    }) > 0.0 {
      image::Luma([0u8])
    } else {
      image::Luma([255u8])
    }
  }).save("out.png")?;
  Ok(())
}

/// draw the quadree layout
pub fn tree_test(tree: Quadtree, path: String) -> Result<()> {
  let mut img = BitMapBackend::new(
    &path,
    ((WORLD_SIZE * IMG_SCALE) as u32 + 1, (WORLD_SIZE * IMG_SCALE) as u32 + 1)
  );

  tree.traverse(&mut move |depth, tree| {
    let rect = tree.rect;
    let color =
      if tree.data { RGBColor(255, (255.0 / 1.5f32.powf((10 - depth) as f32)) as u8, 0) } else { RGBColor(32, 32, 255) }
        .mix(if tree.data {
          0.0282475249 / 0.7f64.powf(depth as f64)
          //4.0 / 1.5f64.powf(depth as f64)
        } else {
          1.0 / 1.1f64.powf(depth as f64)
        }
    );
    img.draw_rect(
      (
        ((rect.center.x - rect.size / 2.0) * IMG_SCALE) as i32,
        ((rect.center.y - rect.size / 2.0) * IMG_SCALE) as i32),
      (
        ((rect.center.x + rect.size / 2.0) * IMG_SCALE) as i32,
        ((rect.center.y + rect.size / 2.0) * IMG_SCALE) as i32),
      &color,
      tree.data
      ).ok()?;
    Ok(())
  })
}