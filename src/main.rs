#![feature(type_ascription)]
#![feature(total_cmp)]
#![allow(dead_code)]

use lib::{
  error::Result,
  drawing,
  profile
};
mod tests;

fn main() -> Result<()> {
  let circles = profile!("argmax", tests::sdf_argmax2d_test()?);
  profile!("draw", drawing::draw_circles(
    "out.png",
    circles.into_iter(),
    (2048, 2048).into()
  )?);
  open::that("out.png")?;
  Ok(())
}
