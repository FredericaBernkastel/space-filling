#![allow(dead_code)]
use error_chain::{error_chain};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point {
  pub x: f32,
  pub y: f32,
}

impl std::ops::Add for Point {
  type Output = Self;

  fn add(self, other: Self) -> Self {
    Self {
      x: self.x + other.x,
      y: self.y + other.y,
    }
  }
}

impl std::ops::Sub for Point {
  type Output = Self;

  fn sub(self, other: Self) -> Self {
    Self {
      x: self.x - other.x,
      y: self.y - other.y,
    }
  }
}

#[macro_export]
macro_rules! profile(
  ($title: literal, $stmt: stmt) => {{
    let t0 = std::time::Instant::now();
    $stmt
    println!("{} profile: {}ms", $title, t0.elapsed().as_millis());
  }}
);

impl Point {
  pub fn length(self) -> f32 {
    (self.x * self.x + self.y * self.y).sqrt()
  }

  pub fn translate(self, offset: Self) -> Self {
    self - offset
  }

  pub fn in_rect(self, rect: crate::quadtree::Rect) -> bool {
    let (l, t, r, b) = (
      rect.center.x - rect.size / 2.0,
      rect.center.y - rect.size / 2.0,
      rect.center.x + rect.size / 2.0,
      rect.center.y + rect.size / 2.0
    );
    self.x >= l && self.x < r &&
    self.y >= t && self.y < b
  }
}

error_chain! {
  foreign_links {
    IoError(std::io::Error);
    FromUtf8Error(std::string::FromUtf8Error);
    ImageError(image::error::ImageError);
  }

  errors {
    NoneError(e: std::option::NoneError)
  }
}

impl From<std::option::NoneError> for Error {
  fn from(e: std::option::NoneError) -> Self {
    Error::from_kind(ErrorKind::NoneError(e))
  }
}

pub fn display(error: &Error) -> String {
  let mut msg = "Error:\n".to_string();
  error
    .iter()
    .enumerate()
    .for_each(|(index, error)| msg.push_str(&format!("└> {} - {}", index, error)));

  if let Some(backtrace) = error.backtrace() {
    msg.push_str(&format!("\n\n{:?}", backtrace));
  }
  eprintln!("{}", msg);
  msg
}