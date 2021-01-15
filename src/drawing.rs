use plotters::prelude::*;
use crate::{
  lib::{Result},
  quadtree::Quadtree,
};
use crate::quadtree::{sdf};
use crate::lib::Point;

pub fn exec(tree: Quadtree) -> Result<()> {
  tree_test(tree)?;
  open::that("out.png")?;
  /*img.draw(&Circle::new(
    (entry.x as i32, entry.y as i32),
    entry.r as u32,
    Into::<ShapeStyle>::into(&WHITE).filled()
  ))?;*/
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

fn tree_test(tree: Quadtree) -> Result<()> {
  let mut img = BitMapBackend::new("out.png", (1080, 1080));

  tree.traverse(&mut move |depth, tree| {
    let rect = tree.boundary;
    let color =
      if tree.data { RGBColor(255, 32, 32) } else { RGBColor(32, 32, 255) }
        .mix(1.0 / 1.5f64.powf(depth as f64));
    img.draw_rect(
      ((rect.center.x - rect.size / 2.0) as i32, (rect.center.y - rect.size / 2.0) as i32),
      ((rect.center.x + rect.size / 2.0) as i32, (rect.center.y + rect.size / 2.0) as i32),
      &color,
      false
      ).ok()?;
    Ok(())
  })
}