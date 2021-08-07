#![feature(type_ascription)]
#![feature(total_cmp)]

use lib::{
  error::Result,
  legacy
};

mod tests;

fn main() -> Result<()> {
  legacy::examples::simple()?;
  Ok(())
}
