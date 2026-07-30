//! Geometry vocabulary and the shape catalogue.
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
