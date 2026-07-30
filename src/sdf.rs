//! Signed distance functions: the [`SDF`] and [`Lipschitz`] traits, and the
//! [`Combinator`](crate::geometry::Combinator) algebra that composes them.
//!
//! A re-export of [`adaptive_distance_field::sdf`] — the traits and the
//! combinator types belong to the distance-field layer, since the ADF is defined
//! over them. This crate adds the shape primitives that implement them
//! ([`geometry::shapes`](crate::geometry::shapes)), the space-filling solvers,
//! and drawing.
pub use adaptive_distance_field::sdf::*;
