//! .
//!
//! Following the macro expansion of [`error_chain`] to comply with cfg features and generic errors.
//! ```ignore
//! error_chain! {
//!   foreign_links {
//!     IoError(std::io::Error);
//!     FromUtf8Error(std::string::FromUtf8Error);
//!     ImageError(image::ImageError);
//!   }
//!
//!   errors {
//!     NoneError
//!   }
//! }
//! ```
use std::fmt::Debug;

#[derive(Debug)]
pub struct Error(
  pub ErrorKind,
  pub ::error_chain::State,
);

#[derive(Debug)]
pub enum ErrorKind {
  IoError(std::io::Error),
  FromUtf8Error(std::string::FromUtf8Error),
  #[cfg(feature = "image")]
  ImageError(image::ImageError),
  NoneError,

  Msg(String),
}

impl From<std::io::Error> for Error {
  fn from(e: std::io::Error) -> Self {
    Error::from_kind(
      ErrorKind::IoError(e)
    )
  }
}
impl From<std::string::FromUtf8Error> for Error {
  fn from(e: std::string::FromUtf8Error) -> Self {
    Error::from_kind(
      ErrorKind::FromUtf8Error(e)
    )
  }
}
#[cfg(feature = "image")]
impl From<image::ImageError> for Error {
  fn from(e: image::ImageError) -> Self {
    Error::from_kind(
      ErrorKind::ImageError(e)
    )
  }
}

impl ::std::fmt::Display for ErrorKind {
  fn fmt(&self, fmt: &mut ::std::fmt::Formatter)
         -> ::std::fmt::Result
  {
    use ErrorKind::*;
    match *self {
      IoError(ref err) => fmt.write_fmt(format_args!("{}", err)),
      FromUtf8Error(ref err) => fmt.write_fmt(format_args!("{}", err)),
      #[cfg(feature = "image")]
      ImageError(ref err) => fmt.write_fmt(format_args!("{}", err)),
      NoneError =>  fmt.write_fmt(format_args!("{}", self.description())),

      Msg(ref s) => fmt.write_fmt(format_args!("{}", s)),
    }
  }
}

#[allow(deprecated)]
impl ErrorKind {
  /// A string describing the error kind.
  pub fn description(&self) -> &str {
    use std::error::Error;

    use ErrorKind::*;
    match *self {
      IoError(ref err) => err.description(),
      FromUtf8Error(ref err) => err.description(),
      #[cfg(feature = "image")]
      ImageError(ref err) => err.description(),
      NoneError => stringify!(NoneError),

      ErrorKind::Msg(ref s) => &s,
    }
  }
}

impl ::error_chain::ChainedError for Error {
  type ErrorKind = ErrorKind;

  fn new(kind: ErrorKind, state: ::error_chain::State) -> Error {
    Error(kind, state)
  }

  fn from_kind(kind: Self::ErrorKind) -> Self {
    Self::from_kind(kind)
  }

  fn with_chain<E, K>(error: E, kind: K) -> Self
    where E: ::std::error::Error + Send + 'static,
          K: Into<Self::ErrorKind> {
    Self::with_chain(error, kind)
  }

  fn kind(&self) -> &Self::ErrorKind {
    self.kind()
  }

  fn iter(&self) -> ::error_chain::Iter {
    ::error_chain::Iter::new(Some(self))
  }

  fn chain_err<F, EK>(self, error: F) -> Self
    where F: FnOnce() -> EK,
          EK: Into<ErrorKind> {
    self.chain_err(error)
  }

  fn backtrace(&self) -> Option<&::error_chain::Backtrace> {
    self.backtrace()
  }

  fn extract_backtrace(e: &(dyn ::std::error::Error + Send + 'static))
    -> Option<::error_chain::InternalBacktrace> {
    if let Some(e) = e.downcast_ref::<Error>() {
      return Some(e.1.backtrace.clone());
    }
    None
  }
}
#[allow(dead_code)]
impl Error {
  /// Constructs an error from a kind, and generates a backtrace.
  pub fn from_kind(kind: ErrorKind) -> Error {
    Error(
      kind,
      ::error_chain::State::default(),
    )
  }

  /// Constructs a chained error from another error and a kind, and generates a backtrace.
  pub fn with_chain<E, K>(error: E, kind: K) -> Error
    where E: ::std::error::Error + Send + 'static,
          K: Into<ErrorKind>
  {
    Error::with_boxed_chain(Box::new(error), kind)
  }

  /// Construct a chained error from another boxed error and a kind, and generates a backtrace
  #[allow(unknown_lints, bare_trait_objects)]
  pub fn with_boxed_chain<K>(error: Box<::std::error::Error + Send>, kind: K) -> Error
    where K: Into<ErrorKind>
  {
    Error(
      kind.into(),
      ::error_chain::State::new::<Error>(error),
    )
  }

  /// Returns the kind of the error.
  pub fn kind(&self) -> &ErrorKind {
    &self.0
  }

  /// Iterates over the error chain.
  pub fn iter(&self) -> ::error_chain::Iter {
    ::error_chain::ChainedError::iter(self)
  }

  /// Returns the backtrace associated with this error.
  pub fn backtrace(&self) -> Option<&::error_chain::Backtrace> {
    self.1.backtrace()
  }

  /// Extends the error chain with a new entry.
  pub fn chain_err<F, EK>(self, error: F) -> Error
    where F: FnOnce() -> EK, EK: Into<ErrorKind> {
    Error::with_chain(self, Self::from_kind(error().into()))
  }

  /// A short description of the error.
 /// This method is identical to [`Error::description()`](https://doc.rust-lang.org/nightly/std/error/trait.Error.html#tymethod.description)
  pub fn description(&self) -> &str {
    self.0.description()
  }
}
impl ::std::error::Error for Error {
  #[cfg(not(has_error_description_deprecated))]
  fn description(&self) -> &str {
    self.description()
  }

  fn cause(&self) -> Option<&dyn ::std::error::Error> {
    match self.1.next_error {
      Some(ref c) => Some(&**c),
      None => {
        match self.0 {
          ErrorKind::IoError(ref foreign_err) => {
            foreign_err.source()
          }
          ErrorKind::FromUtf8Error(ref foreign_err) => {
            foreign_err.source()
          }
          _ => None
        }
      }
    }
  }
}
impl ::std::fmt::Display for Error {
  fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
    ::std::fmt::Display::fmt(&self.0, f)
  }
}

impl From<ErrorKind> for Error {
  fn from(e: ErrorKind) -> Self {
    Error::from_kind(e)
  }
}

impl From<Error> for ErrorKind {
  fn from(e: Error) -> Self {
    e.0
  }
}
/// Additional methods for `Result`, for easy interaction with this crate.
pub trait ResultExt<T> {
  /// If the `Result` is an `Err` then `chain_err` evaluates the closure,
 /// which returns *some type that can be converted to `ErrorKind`*, boxes
 /// the original error to store as the cause, then returns a new error
 /// containing the original error.
  fn chain_err<F, EK>(self, callback: F) -> ::std::result::Result<T, Error>
    where F: FnOnce() -> EK,
          EK: Into<ErrorKind>;
}
impl<T, E> ResultExt<T> for ::std::result::Result<T, E> where E: ::std::error::Error + Send + 'static {
  fn chain_err<F, EK>(self, callback: F) -> ::std::result::Result<T, Error>
    where F: FnOnce() -> EK,
          EK: Into<ErrorKind> {
    self.map_err(move |e| {
      let state = ::error_chain::State::new::<Error>(Box::new(e));
      ::error_chain::ChainedError::new(callback().into(), state)
    })
  }
}
impl<T> ResultExt<T> for ::std::option::Option<T> {
  fn chain_err<F, EK>(self, callback: F) -> ::std::result::Result<T, Error>
    where F: FnOnce() -> EK,
          EK: Into<ErrorKind> {
    self.ok_or_else(move || {
      ::error_chain::ChainedError::from_kind(callback().into())
    })
  }
}
impl<'a> From<&'a str> for ErrorKind {
  fn from(s: &'a str) -> Self {
    ErrorKind::Msg(s.into())
  }
}
impl From<String> for ErrorKind {
  fn from(s: String) -> Self {
    ErrorKind::Msg(s)
  }
}
impl<'a> From<&'a str> for Error {
  fn from(s: &'a str) -> Self {
    Self::from_kind(s.into())
  }
}
impl From<String> for Error {
  fn from(s: String) -> Self {
    Self::from_kind(s.into())
  }
}
/// Convenient wrapper around `std::Result`.
#[allow(unused)]
pub type Result<T> = ::std::result::Result<T, Error>;

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