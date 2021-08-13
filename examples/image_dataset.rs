#![allow(dead_code)]
use space_filling::{
  error::Result,
  argmax2d::Argmax2D,
  drawing
};

use embedded::embedded;
mod embedded;

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

/// Generate a distribution, and use it to display an image dataset, up to 100'000 images.
fn main() -> Result<()> {
  let image_folder = std::env::args().nth(1)
    .map(|path| std::path::Path::new(&path).is_dir().then(|| path))
    .flatten()
    .expect("please provide a valid path in arguments");

  let mut argmax = Argmax2D::new(16384, 64)?;
  let circles = embedded(&mut argmax);

  let files = find_files(
    &image_folder, {
      let reg = regex::Regex::new("^.+\\.(jpg|png)$").unwrap();
      move |file| reg.is_match(file)
    }
  );

  drawing::draw_img_parallel(
    circles,
    files,
    (16384, 16384).into(),
    4
  )?.save("out.png")?;
  open::that("out.png")?;
  Ok(())
}