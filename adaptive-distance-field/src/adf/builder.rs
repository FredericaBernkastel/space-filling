//! A builder for [`ADF`], pinning one type parameter at a time.
//!
//! `ADF<Float, D, L>` takes a scalar, a dimension count and a subdivision
//! layout, none of which has a default — the layout in particular is a real
//! decision (see [`tree`](crate::adf::tree)) and reads badly as an omission. The
//! turbofish states all three at once:
//!
//! ```
//! # use adaptive_distance_field::{adf::{ADF, Primitive, tree::Kd}, sdf};
//! let field = ADF::<f64, 6, Kd>::new(3, vec![Primitive::new(sdf::boundary_rect)]);
//! ```
//!
//! The builder states them one at a time, in any order, naming each:
//!
//! ```
//! # use adaptive_distance_field::adf;
//! let field = adf::builder().f64().dims::<6>().kd().bounded(3);
//! ```
//!
//! Each slot starts as [`Unset`], which satisfies none of the bounds
//! [`AdfBuilder::build`] needs, so a forgotten refinement is a compile error at
//! the point of `build` rather than a surprising default.

use {
  super::{Bucket, Primitive, ADF},
  crate::{
    adf::tree::{Kd, Layout, Orthant, Split},
    geometry::Real,
    sdf,
  },
  std::marker::PhantomData,
};

/// A builder slot that has not been chosen yet. Deliberately implements nothing.
pub struct Unset;

/// A pinned dimension count. The count lives in a type so that "not yet chosen"
/// is a state of its own rather than `D = 0`, which would otherwise be a legal
/// and useless field.
pub struct Dims<const D: usize>;

/// Start building. See the [module documentation](self).
pub fn builder() -> AdfBuilder<Unset, Unset, Unset> {
  AdfBuilder { prune_subdiv: 8, _pin: PhantomData }
}

/// An [`ADF`] under construction: three type slots and one tuning value.
pub struct AdfBuilder<Float, Dim, L> {
  prune_subdiv: u32,
  _pin: PhantomData<fn() -> (Float, Dim, L)>,
}

impl<Float, Dim, L> AdfBuilder<Float, Dim, L> {
  /// Carry the settings across a change of type slot.
  fn refine<F2, D2, L2>(self) -> AdfBuilder<F2, D2, L2> {
    AdfBuilder { prune_subdiv: self.prune_subdiv, _pin: PhantomData }
  }

  /// Pin the scalar type. [`Self::f64`] and [`Self::f32`] are the usual choices.
  pub fn scalar<F>(self) -> AdfBuilder<F, Dim, L> {
    self.refine()
  }
  /// `scalar::<f64>()`.
  pub fn f64(self) -> AdfBuilder<f64, Dim, L> {
    self.refine()
  }
  /// `scalar::<f32>()`.
  pub fn f32(self) -> AdfBuilder<f32, Dim, L> {
    self.refine()
  }

  /// Pin the dimension count.
  pub fn dims<const D: usize>(self) -> AdfBuilder<Float, Dims<D>, L> {
    self.refine()
  }

  /// Pin the subdivision layout. [`Self::orthant`] and [`Self::kd`] name the two
  /// that ship.
  pub fn layout<M>(self) -> AdfBuilder<Float, Dim, M> {
    self.refine()
  }
  /// `layout::<Orthant>()` — `2^D` children per node, the historical tree.
  pub fn orthant(self) -> AdfBuilder<Float, Dim, Orthant> {
    self.refine()
  }
  /// `layout::<Kd>()` — one axis per level, 2 children per node.
  pub fn kd(self) -> AdfBuilder<Float, Dim, Kd> {
    self.refine()
  }

  /// Refinement budget of the redundancy proof, in full subdivisions. Defaults
  /// to 8; lower it when the proof rather than the tree dominates the clock.
  pub fn prune_subdiv(mut self, subdiv: u32) -> Self {
    self.prune_subdiv = subdiv;
    self
  }
}

/// Terminal methods, available only once all three slots are pinned to something
/// that can actually back a field.
impl<Float, const D: usize, L> AdfBuilder<Float, Dims<D>, L>
where
  Float: Real + Send + Sync,
  L: Layout<D>,
  L::Children<Split<Bucket<Float, D>, D, L>>: Send,
{
  /// Build with the given initial primitives. `max_depth` is in **full**
  /// subdivisions — halvings of every axis — so it means one resolution in either
  /// layout. It is an argument rather than a setter because a default depth
  /// silently changes memory by orders of magnitude.
  pub fn build(self, max_depth: u8, init: Vec<Primitive<Float, D>>) -> ADF<Float, D, L> {
    ADF::new(max_depth, init).with_prune_subdiv(self.prune_subdiv)
  }

  /// Build seeded with the walls of the unit hypercube — the usual first
  /// primitive, which keeps shapes from escaping the domain.
  pub fn bounded(self, max_depth: u8) -> ADF<Float, D, L> {
    self.build(max_depth, vec![Primitive::new(sdf::boundary_rect)])
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::sdf::SDF;

  #[test] fn refinements_compose_in_any_order() {
    let a = builder().f64().dims::<3>().orthant().bounded(4);
    let b = builder().orthant().dims::<3>().f64().bounded(4);
    let c = builder().dims::<3>().scalar::<f64>().layout::<Orthant>().bounded(4);
    let probe = crate::geometry::Point::from([0.25, 0.5, 0.75]);
    assert_eq!(a.sdf(probe), b.sdf(probe));
    assert_eq!(a.sdf(probe), c.sdf(probe));
    assert_eq!(a.layout_name(), "orthant");
  }

  #[test] fn layout_and_tuning_are_independent() {
    let kd = builder().f64().dims::<6>().kd().prune_subdiv(3).bounded(2);
    assert_eq!(kd.layout_name(), "k-d");
    // 2 full subdivisions of 6 axes = 12 levels of a binary tree
    assert_eq!(kd.tree.max_depth, 12);
  }
}
