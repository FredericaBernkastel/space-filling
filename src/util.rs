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

impl Point {
  pub fn length(self) -> f32 {
    (self.x * self.x + self.y * self.y).sqrt()
  }

  pub fn translate(self, offset: Self) -> Self {
    self - offset
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