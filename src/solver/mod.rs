//! Two solvers for the space-filling problem, and the optimizer they share.
//!
//! [`Argmax2D`] stores the field as a discrete bitmap and returns the exact
//! **global** maximum, at quadratic memory cost — it lives here, being inherently
//! two-dimensional. [`ADF`] stores the field adaptively as a `2^N`-tree of
//! [`Primitive`]s — continuous, exact, 10–100× smaller, and defined in any
//! dimension — paired with [`LineSearch`], an adaptive gradient ascent that
//! converges to a **local** maximum (together, the GD-ADF method).
//!
//! The ADF and the ascent are re-exported from [`adaptive_distance_field`],
//! which owns them: neither has anything to do with space filling specifically.
//! What this crate contributes on top is
//! [`util::find_max_parallel`](crate::util::find_max_parallel) — batched maxima
//! whose free balls are pairwise disjoint, so a whole batch can be placed against
//! one field snapshot without the shapes colliding.

pub mod argmax2d;
pub use argmax2d::Argmax2D;

pub use adaptive_distance_field::{
  adf, line_search,
  adf::{ADF, Primitive},
  line_search::LineSearch,
};
