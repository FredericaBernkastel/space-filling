//! Two solvers for the space-filling problem, and the optimizer they share.
//!
//! [`Argmax2D`] stores the field as a discrete bitmap and returns the exact
//! **global** maximum, at quadratic memory cost. [`ADF`] stores the field
//! adaptively as a quadtree of [`Primitive`]s — continuous, exact, and 10–100×
//! smaller — paired with [`LineSearch`], an adaptive gradient ascent that
//! converges to a **local** maximum (together, the GD-ADF method).

pub mod argmax2d;
pub use argmax2d::Argmax2D;

pub mod line_search;
pub use line_search::LineSearch;

pub mod adf;
pub use adf::{ADF, Primitive};

