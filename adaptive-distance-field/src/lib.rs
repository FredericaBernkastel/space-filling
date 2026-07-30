//! Adaptively sampled distance fields in ℝᴺ, with Lipschitz-certified pruning.
//!
//! An [`ADF`](adf::ADF) stores a signed distance field as a `2^N`-tree whose
//! leaves each hold a handful of [`Primitive`](adf::Primitive)s — a field
//! closure paired with an honest upper bound on its gradient. The tree
//! represents their pointwise `min`, so a field composed of millions of
//! primitives is sampled in logarithmic time by descending to the leaf covering
//! the query point, instead of evaluating every primitive.
//!
//! What makes that sound rather than merely fast is the Lipschitz bound. Every
//! decision the structure makes — whether a new primitive can be skipped,
//! whether a stored one has become redundant, which subtrees an insertion can
//! possibly change — is settled by branch-and-bound over
//! `f - g ≥ (f - g)(c) - (L_f + L_g)·h`, which either *proves* a region clean or
//! descends to a witness. A primitive is never dropped on a heuristic, only on a
//! proof, so the stored field never deviates from the `min` over everything
//! inserted.
//!
//! ```no_run
//! use adaptive_distance_field::{
//!   adf::{ADF, Primitive},
//!   geometry::{Point, VectorExt},
//!   line_search::LineSearch,
//!   sdf::{self, SDF},
//! };
//!
//! // seed a 3D field with the walls of the unit cube, positive inside
//! let mut field = ADF::<f64, 3>::new(6, vec![Primitive::new(sdf::boundary_rect)]);
//!
//! // climb to a local maximum of the free space — the deepest point around
//! let start = Point::from([0.3, 0.6, 0.5]);
//! let peak = LineSearch::default().optimize(|p| field.sdf(p), start);
//! let clearance = field.sdf(peak);
//!
//! // drop a ball there, and let the tree work out which cells that can affect
//! let ball = move |p: Point<f64, 3>| (p - peak).length() - clearance / 2.0;
//! field.insert_at_maximum(
//!   adaptive_distance_field::geometry::DistPoint { point: peak, distance: clearance },
//!   Primitive::new(ball),
//! );
//! ```
//!
//! # Layout
//!
//! - [`geometry`] — the vocabulary: [`Point`](geometry::Point) and
//!   [`Vector`](geometry::Vector) (both [`nalgebra`]'s, re-exported so
//!   downstream code can name the same versions), the axis-aligned
//!   [`Aabb`](geometry::Aabb), [`BoundingBox`](geometry::BoundingBox), and the
//!   [`Combinator`](geometry::Combinator) trait — every way to transform or
//!   combine one field into another. Each combinator preserves or `max`-combines
//!   its operands' constants, so a composed chain still reports an honest bound.
//! - [`sdf`] — the [`SDF`](sdf::SDF) and [`Lipschitz`](sdf::Lipschitz) traits
//!   everything is written against, the field types the combinators return, and
//!   [`sdf_geq_everywhere`](adf::sdf_geq_everywhere): the branch-and-bound
//!   proof that underpins all of it. [`boundary_rect`](sdf::boundary_rect) seeds
//!   a field with the walls of the unit cube.
//! - [`adf`] — the structure itself, over the [`quadtree`](adf::quadtree) arena.
//! - [`line_search`] — adaptive gradient ascent, for locating the maxima that
//!   [`insert_at_maximum`](adf::ADF::insert_at_maximum) consumes.
//!
//! Dimension count is a compile-time constant throughout, so `ADF<f64, 2>` and
//! `ADF<f64, 3>` monomorphize separately and the 2D case costs exactly what a
//! quadtree-only implementation would.
//!
//! # Beyond this crate
//!
//! The [`space-filling`](https://docs.rs/space-filling/) crate builds on this
//! one: a catalogue of shape primitives in 2, 3, 4 and N dimensions, a discrete
//! bitmap solver for exact global maxima, batched parallel maxima search, and
//! 2D rasterization.

#![allow(clippy::type_complexity)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// Re-exported: the geometry vocabulary is nalgebra's, so downstream code can
/// name the exact same versions.
pub use nalgebra;

pub mod adf;
pub mod geometry;
pub mod line_search;
pub mod sdf;
