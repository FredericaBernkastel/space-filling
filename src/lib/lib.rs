#![feature(box_syntax)]
#![feature(type_ascription)]
#![feature(try_trait)]
#![feature(array_map)]
#![allow(dead_code)]

pub mod error;
pub mod quadtree;
pub mod sdf;
pub mod argmax;
pub mod drawing;
pub mod geometry;

#[macro_export]
macro_rules! profile(
  ($title: literal, $stmt: stmt) => {{
    let t0 = std::time::Instant::now();
    $stmt
    println!("{} profile: {}ms", $title, t0.elapsed().as_millis());
  }}
);

pub fn find_files(path: &str) -> impl Iterator<Item = std::path::PathBuf> {
  use walkdir::{WalkDir, DirEntry};

  WalkDir::new(path)
    .sort_by(|a, b| {
      let [a, b] = [a, b].map(|x| x.file_name().to_string_lossy().to_string());
      lexical_sort::natural_cmp(&b, &a) // reversed
    })
    .into_iter()
    .filter_map(std::result::Result::ok)
    .map(|file: DirEntry| file.path().to_owned())
    .filter(|file| {
      let f = file.to_string_lossy();
      f.ends_with(".png") || f.ends_with(".jpg")
    })
}
