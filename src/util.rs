use {
  num_traits::FloatConst,
  rand::prelude::*,
  crate::{
    geometry::{Point, Vector, Aabb, DistPoint, Real, VectorExt},
    solver::LineSearch,
  }
};

/// The insertion domain for a primitive placed at the **global maximum** `p`
/// of the field.
///
/// For `S ⊆ B̄(x₀, d)`: `f(v) ≥ |v − x₀| − d`, while globality gives
/// `g(v) ≤ d` everywhere — so `f(v) < g(v)` forces `|v − x₀| < 2d`. The update
/// region is contained in the ball `B(x₀, 2d)`, whose minimal axis-aligned
/// cover is the box of side `4d`. The bound is attained (two tangent
/// maximal balls), hence the constant `4` is optimal in any dimension.
///
/// For **local** maxima no `c·d` box is sound; use
/// [`ADF::update_domain`](crate::solver::ADF::update_domain) instead.
pub fn domain_global_max<P: Real, const D: usize>(p: DistPoint<P, P, D>) -> Aabb<P, D> {
  let two = P::one() + P::one();
  let half = p.distance * two;
  Aabb {
    min: p.point.map(|x| x - half),
    max: p.point.map(|x| x + half),
  }
}

/// The historical insertion domain: a box of side `4·√2·d`, found by trial
/// and error.
///
/// It is *oversized* for global maxima (see [`domain_global_max`]: side `4d`
/// suffices and is optimal) and *unsound* for local maxima — the update region
/// of an insertion is not bounded by any multiple of `d` (see
/// [`ADF::update_domain`](crate::solver::ADF::update_domain)) — so it survives
/// only as a cheap heuristic.
#[deprecated(note = "use `domain_global_max` for global maxima, or `ADF::update_domain` for local maxima")]
pub fn domain_empirical<P: Real + FloatConst, const D: usize>(p: DistPoint<P, P, D>) -> Aabb<P, D> {
  let two = P::one() + P::one();
  let half = p.distance * two * P::SQRT_2();
  Aabb {
    min: p.point.map(|x| x - half),
    max: p.point.map(|x| x + half),
  }
}

/// Find up to `batch_size` distinct local maxima via the adaptive ascent
/// optimizer ([`LineSearch`]); the returned maxima have pairwise-disjoint free
/// balls, so shapes placed at them cannot intersect within the batch.
pub fn find_max_parallel<_Float, const D: usize>(
  f: impl Fn(Point<_Float, D>) -> _Float + Send + Sync,
  batch_size: u64,
  rng: &mut impl Rng,
  line_search: LineSearch<_Float>,
) -> Vec<DistPoint<_Float, _Float, D>>
  where _Float: Real + Send + Sync
{
  use rayon::prelude::*;

  let mut rng_buf = vec![Point::from(Vector::<_Float, D>::zeros()); batch_size as usize];
  rng_buf.iter_mut().for_each(|x| {
    *x = Point::from(std::array::from_fn(|_|
      _Float::from(rng.random_range::<f64, _>(0.0..1.0)).unwrap()
    ));
  });

  let points: Vec<_> = rng_buf.into_par_iter()
    .filter_map(|p0| {
      let p1 = line_search.optimize(&f, p0);
      let p1 = DistPoint {
        point: p1,
        distance: f(p1)
      };
      (p1.distance > line_search.Δ).then_some(p1)
    })
    .collect();
  let mut points1 = vec![];
  points.into_iter()
    .for_each(|pn| {
      // The whole batch is measured against one field snapshot, and each
      // insertion invalidates that snapshot for the rest of the batch. Keeping
      // `pn` only when its free ball is disjoint from every kept ball
      // (`|p − pn| > d_p + d_pn`) makes the batch's placements provably
      // non-intersecting regardless of insertion order — a shape placed at a
      // maximum stays inside that maximum's ball.
      points1.iter()
        .all(|p: &DistPoint<_, _, D>| (p.point - pn.point).length() > p.distance + pn.distance)
        .then(|| points1.push(pn));
    });
  points1
}

/// A convenience wrapper around [find_max_parallel], produces an infinite iterator.
pub fn local_maxima_iter<_Float, const D: usize>(
  f: impl Fn(Point<_Float, D>) -> _Float + Send + Sync,
  batch_size: u64,
  rng_seed: u64,
  line_search: LineSearch<_Float>,
) -> impl Iterator<Item = DistPoint<_Float, _Float, D>>
  where _Float: Real + Send + Sync
{
  let mut rng = rand_pcg::Lcg128Xsl64::seed_from_u64(rng_seed);

  std::iter::repeat(()).flat_map(move |_|
    find_max_parallel(&f, batch_size, &mut rng, line_search)
  )
}
