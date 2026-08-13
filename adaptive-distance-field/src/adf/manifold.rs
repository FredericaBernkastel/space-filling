//! Compact manifolds: a domain whose axes carry decaying weight.
//!
//! Ambient dimension is the wrong number to reason about above `D ≈ 12`. What
//! governs cost is how many axes *matter*, and that is a property of the domain
//! rather than of the tree: §2.3 of `doc/publications/infinite_dimensions` gets
//! finite covering numbers back by replacing the ball with an ellipsoid
//! `Ωₐ = {x : Σ aᵢxᵢ² ≤ 1}`, `aᵢ → ∞`, whose semi-axis along `i` decays as
//! `1/√aᵢ`. A [`Manifold`] is that ellipsoid's bounding box, which is all the
//! tree ever needed: the weights are its extents.
//!
//! Building one by hand is possible and was how [`ADF::new_in`] was first used,
//! but it invites a silent mistake — seed the field with
//! [`boundary_rect`](crate::sdf::boundary_rect) instead of the domain's own
//! walls and the two disagree, with no error and a wrong field. [`Manifold::field`]
//! wires them together so they cannot.

use {
  crate::{
    adf::{Bucket, Layout, Primitive, tree::Split, ADF},
    geometry::{Aabb, Point, Real, Vector, VectorExt},
    sdf,
  },
  nalgebra::Scalar,
};

/// A compact manifold, given by its per-axis weights `γ`.
///
/// Weights are extents, not a separate concept: axis `i` matters in proportion
/// to `γᵢ`, and [`Widest`](super::Widest) cuts in descending `γ` for exactly
/// that reason.
#[derive(Clone, Debug)]
pub struct Manifold<Float: Scalar, const D: usize> {
  gamma: Vector<Float, D>,
}

impl<_Float: Real, const D: usize> Manifold<_Float, D> {
  /// Sobolev-type decay `γᵢ = (i+1)^(−s)`, the scaling of §2.3: `aᵢ ≍ i^(2s)`
  /// has semi-axes `i^(−s)`, and `s` is the smoothness that trades against
  /// certification cost — `log N(ε) ≍ ε^(−1/s)`.
  ///
  /// `s = 0` is the unit cube, and every claim about weights degenerates there.
  pub fn sobolev(s: _Float) -> Self {
    Self::from_extents(Vector::from_fn(|i, _| {
      let idx: _Float = num_traits::cast((i + 1) as f64).unwrap();
      idx.powf(-s)
    }))
  }

  /// Geometric decay `γᵢ = e^(−rate·i)`. Falls away faster than any Sobolev
  /// scaling, so the effective dimension is bounded however large `D` is.
  pub fn exponential(rate: _Float) -> Self {
    Self::from_extents(Vector::from_fn(|i, _| {
      let idx: _Float = num_traits::cast(i as f64).unwrap();
      (-rate * idx).exp()
    }))
  }

  /// A genuinely `k`-dimensional manifold embedded in `D`: the first `k` axes
  /// carry weight 1, the rest `tail`.
  ///
  /// `tail` must be positive. Zero is the honest description of a `k`-plane, and
  /// also a box with empty interior, which no field can be built over — pass a
  /// small positive number and the tail axes simply never get cut.
  pub fn finite_rank(k: usize, tail: _Float) -> Self {
    Self::from_extents(Vector::from_fn(|i, _| {
      if i < k { _Float::one() } else { tail }
    }))
  }

  /// Arbitrary weights, taken as given.
  pub fn from_extents(gamma: Vector<_Float, D>) -> Self {
    Self { gamma }
  }

  /// The weights themselves.
  pub fn weights(&self) -> &Vector<_Float, D> {
    &self.gamma
  }

  /// The domain box, centred on the origin.
  pub fn domain(&self) -> Aabb<_Float, D> {
    let two = _Float::one() + _Float::one();
    Aabb {
      min: Point::from(self.gamma / -two),
      max: Point::from(self.gamma / two),
    }
  }

  /// The walls of [`Self::domain`] — the seed primitive, positive inside.
  pub fn walls(&self) -> impl Fn(Point<_Float, D>) -> _Float + Copy + Send + Sync + 'static
  where
    _Float: Send + Sync + 'static,
  {
    sdf::boundary_box(self.domain())
  }

  /// The weights normalised to unit length, which is the aspect a body placed
  /// here should take.
  ///
  /// Scaled by `t`, this is a box of **circumradius** `t` — so `t = g(p)` is
  /// always free without any proof, and is where [`ADF::grow_box`] starts.
  pub fn aspect(&self) -> Vector<_Float, D> {
    let norm = self.gamma.length();
    if norm > _Float::zero() { self.gamma / norm } else { self.gamma }
  }

  /// How many axes actually matter: the participation ratio
  /// `(Σγᵢ)² / Σγᵢ²`.
  ///
  /// `D` on a cube, `1` on a rank-one manifold, and for Sobolev decay it settles
  /// to a constant independent of `D` — 2.5 at `s = 2`, which is why a
  /// hundred-dimensional field of that shape costs what a three-dimensional one
  /// does. This is the number the cost of everything downstream should track.
  pub fn effective_dimension(&self) -> _Float {
    let sum = self.gamma.iter().fold(_Float::zero(), |a, &b| a + b);
    let sq = self.gamma.iter().fold(_Float::zero(), |a, &b| a + b * b);
    if sq > _Float::zero() { sum * sum / sq } else { _Float::zero() }
  }

  /// `ln vol(domain) = Σ ln γᵢ`.
  ///
  /// In log space because the volume itself is unrepresentable: at `D = 100`
  /// with `γᵢ = (i+1)^(−2)` it is around `10^(−316)`, and one more axis
  /// underflows `f64` to zero.
  pub fn log_volume(&self) -> _Float {
    self.gamma.iter().fold(_Float::zero(), |a, &b| a + b.ln())
  }
}

impl<_Float: Real + Send + Sync + 'static, const D: usize> Manifold<_Float, D> {
  /// A field over this manifold, seeded with its own walls.
  ///
  /// The one call exists so that domain and walls cannot disagree:
  ///
  /// ```
  /// # use adaptive_distance_field::adf::{Manifold, WeightedKd, ADF};
  /// let m = Manifold::<f64, 24>::sobolev(2.0);
  /// let field: ADF<f64, 24, WeightedKd> = m.field(1);
  /// assert_eq!(field.tree.root().rect.max, m.domain().max);
  /// ```
  pub fn field<L>(&self, splits: u8) -> ADF<_Float, D, L>
  where
    L: Layout<D>,
    L::Children<Split<Bucket<_Float, D>, D, L>>: Send,
  {
    let d = self.domain();
    ADF::new_in(d, splits, vec![Primitive::enclosing(d.center(), self.walls())])
  }
}
