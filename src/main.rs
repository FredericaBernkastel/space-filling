#![feature(try_trait)]
#![feature(box_syntax)]
#![allow(dead_code)]

use std::{thread};
use lib::Result;

mod quadtree;
mod drawing;
#[path = "util.rs"]
mod lib;

fn main() -> Result<()> {
  use std::time::Instant;

  thread::Builder::new()
    .spawn(||{
      let t0 = Instant::now();
      let tree = quadtree::exec()?;
      println!("tree profile: {}µs", t0.elapsed().as_micros());

      tree.print_stats();

      let t0 = Instant::now();
      drawing::exec(tree)?;
      println!("draw profile: {}µs", t0.elapsed().as_micros());
      Ok(())
    })?
    .join()
    .unwrap()
}
