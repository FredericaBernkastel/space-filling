//! Generate a distribution, and use it to display an image dataset.

#![allow(dead_code)]
use {
  space_filling::{
    solver::Argmax2D,
    drawing::{self, Draw},
    geometry::{Shape, BoundingBox}
  },
  embedded::embedded,
  anyhow::Result,
  image::RgbaImage
};
#[path = "03_embedded.rs"]
mod embedded;

fn find_files(
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
    .filter_map(Result::ok)
    .map(|file: DirEntry| file.path().to_owned())
    .filter(move |file| filter(file.file_name().unwrap().to_string_lossy().as_ref()))
}

fn main() -> Result<()> {
  use rayon::prelude::*;

  let image_folder = std::env::args().nth(1)
    .map(|path| std::path::Path::new(&path).is_dir().then(|| path))
    .flatten()
    .expect("Please provide a valid folder path in arguments");

  let mut argmax = Argmax2D::new(16384, 64)?;
  let shapes = embedded(&mut argmax);

  let files = find_files(
    &image_folder, {
      let reg = regex::Regex::new("^.+\\.(jpg|png)$").unwrap();
      move |file| reg.is_match(file)
    }
  );

  let shapes = shapes.zip(files)
    .filter_map(|(shape, file)| {
      image::open(&file).map(|tex| {
        println!("{:?} -> {:?}", shape.bounding_box(), file);
        Box::new(shape.texture(tex)) as Box<dyn Draw<_, _> + Send + Sync>
      }).map_err(|_| println!("unable to open {:?}", file)).ok()
    })
    .par_bridge();

  drawing::draw_parallel(&mut RgbaImage::new(16384, 16384), shapes)
    .save("out.png")?;
  open::that("out.png")?;
  Ok(())
}