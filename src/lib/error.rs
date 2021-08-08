use error_chain::{error_chain};
use std::fmt::Debug;

error_chain! {
  foreign_links {
    IoError(std::io::Error);
    FromUtf8Error(std::string::FromUtf8Error);
    ImageError(image::error::ImageError);
  }

  errors {
    NoneError
  }
}

/*impl From<std::option::NoneError> for Error {
  fn from(e: std::option::NoneError) -> Self {
    Error::from_kind(ErrorKind::NoneError(e))
  }
}

impl From<ocl::error::Error> for Error {
  fn from(e: ocl::error::Error) -> Self {
    Error::from_kind(ErrorKind::GPUError(e))
  }
}*/

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