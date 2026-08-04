//! Geometry vocabulary and the shape catalogue.
//!
//! <picture>
//!   <source
//!     srcset="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/_collage.avif"
//!     media="(dynamic-range: high)">
//!   <img
//!     src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/shapes/_collage_sdr.avif"
//!     alt="every shape and combinator in the catalogue, as a 7x7 sheet of its own signed distance field, each labelled with the field it draws"
//!     width="100%">
//! </picture>
//!
//! *Every field, drawn from itself — the planar ones as contour
//! plots, the rest are 3D projections. Bottom right panel provides notation
//! the sheet is written in.*
//!
//! The vocabulary — [`Point`], [`Vector`], [`Aabb`], [`BoundingBox`], the
//! [`Combinator`] algebra and the transform types it returns — is re-exported
//! from [`adaptive_distance_field::geometry`], because the ADF is defined over
//! it. What this crate adds is [`shapes`], the primitives that implement
//! [`SDF`](crate::sdf::SDF): unit spheres and boxes in one to infinite dimensions,
//! several planar families, the exceptional 4-polytopes and [gyroid](https://en.wikipedia.org/wiki/Gyroid).
//!
//! Coordinates are normalized with the origin in the minimal corner and every
//! axis growing positive (for images: top-left origin, y-axis down); the solvers
//! operate over the unit hypercube `[0, 1]^D`. Pixel coordinates are the same
//! points over an integer scalar — the scalar type alone distinguishes the two
//! spaces.
//!
//! Each primitive is a *unit* shape inscribed in the unit sphere (spanning
//! `[-1, 1]`, centred at the origin), then positioned with
//! [`Combinator::translate`] / [`scale`](Combinator::scale) /
//! [`rotate`](Combinator::rotate). Every combinator preserves or `max`-combines
//! its operands' Lipschitz bounds, so a composed shape reports an honest
//! constant to the ADF (see [`Lipschitz`](crate::sdf::Lipschitz)).

pub use adaptive_distance_field::geometry::*;

pub mod shapes;
pub use shapes::*;
