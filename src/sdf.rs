//! Signed distance functions: [`SDF`] and [`Lipschitz`] traits, and the
//! [`Combinator`](crate::geometry::Combinator) algebra that composes them.
//!
//! A re-export of [`adaptive_distance_field::sdf`] — traits and
//! combinator types belong to the distance-field layer, since the ADF is defined
//! over them.
pub use adaptive_distance_field::sdf::*;
