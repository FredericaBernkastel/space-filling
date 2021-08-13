#![feature(type_ascription)]
#![feature(total_cmp)]
#![allow(dead_code)]

use lib::{
  error::Result,
  drawing,
  argmax2d::Argmax2D,
};
mod tests;

fn main() -> Result<()> {
  let path = "out.png";
  let mut argmax = Argmax2D::new(1024, 16)?;
  let circles = tests::fractal_distribution(&mut argmax);
  drawing::draw_circles(
    path,
    circles,
    (2048, 2048).into()
  )?;
  open::that(path)?;
  Ok(())
}
