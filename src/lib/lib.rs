#![feature(box_syntax)]
#![feature(type_ascription)]
#![feature(associated_type_defaults)]
#![feature(total_cmp)]

pub mod error;
pub mod sdf;
pub mod legacy;
pub mod drawing;
pub mod geometry;

#[macro_export]
macro_rules! profile(
  ($title: literal, $stmt: stmt) => {{
    let t0 = std::time::Instant::now();
    let ret = {$stmt};
    println!("{} profile: {}ms", $title, t0.elapsed().as_millis());
    ret
  }}
);

pub fn find_files(
  path: &str,
  filter: impl Fn(&str) -> bool
) -> impl Iterator<Item = std::path::PathBuf> {
  use walkdir::{WalkDir, DirEntry};

  WalkDir::new(path)
    .sort_by(|a, b| {
      let [a, b] = [a, b].map(|x| x.file_name().to_string_lossy().to_string());
      lexical_sort::lexical_cmp(&a, &b)
    })
    .into_iter()
    .filter_map(std::result::Result::ok)
    .map(|file: DirEntry| file.path().to_owned())
    .filter(move |file| filter(file.file_name().unwrap().to_string_lossy().as_ref()))
}
