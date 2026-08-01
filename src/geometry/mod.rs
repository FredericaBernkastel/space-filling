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
//! Every field in the catalogue, drawn from itself — the planar ones as contour
//! plots, the rest meshed and rendered, and the last two cells giving the notation
//! the sheet is written in. Built by `doc/shape_gallery/collage.py`.
//!
//! There are two copies of it. The first is HDR, exactly as the sources were
//! graded, and a display that can show it gets it; the second is tone-mapped for
//! everything else. They have to be separate files: a browser without HDR support
//! reads the HLG signal as display light, with no reference-white mapping, and
//! renders the graded sheet at about half brightness — so no one set of pixels is
//! right for both, and the page picks with `(dynamic-range: high)`.
//!
//! The vocabulary — [`Point`], [`Vector`], [`Aabb`], [`BoundingBox`], the
//! [`Combinator`] algebra and the transform types it returns — is re-exported
//! from [`adaptive_distance_field::geometry`], because the ADF is defined over
//! it. What this crate adds is [`shapes`]: the primitives that implement
//! [`SDF`](crate::sdf::SDF), from unit spheres and boxes in any dimension to
//! star polygons, the exceptional 4-polytopes and a gyroid.
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
