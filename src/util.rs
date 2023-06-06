use {
  num_traits::{Float, FloatConst},
  euclid::{Rect, Size2D, Vector2D as V2},
  rand::prelude::*,
  crate::{
    geometry::{P2, DistPoint, WorldSpace},
    solver::line_search::LineSearch,
  }
};

pub fn domain_empirical<P: Float + FloatConst>(p: DistPoint<P, P, WorldSpace>) -> Rect<P, WorldSpace> {
  let size = p.distance * P::from(4.0).unwrap() * P::SQRT_2();
  Rect {
    origin: (p.point.to_vector() - V2::splat(size) / (P::one() + P::one())).to_point(),
    size: Size2D::splat(size)
  }
}

/// Find up to `batch_size` distinct local maxima using GD optimizer
pub fn find_max_parallel<TFloat>(f: impl Fn(P2<TFloat>) -> TFloat + Send + Sync, batch_size: u64, rng_seed: u64, line_search: LineSearch<TFloat>)
                                 -> Vec<DistPoint<TFloat, TFloat, WorldSpace>>
  where TFloat: Float + Send + Sync
{
  use rayon::prelude::*;
  use rand_pcg::Lcg128Xsl64;
  let points: Vec<_> = (0..batch_size).into_par_iter()
    .filter_map(|j| {
      let mut rng = Lcg128Xsl64::seed_from_u64(
        rng_seed.wrapping_add(j)
      );
      let p0 = P2::new(
        TFloat::from(rng.gen_range::<f64, _>(0.0..1.0)).unwrap(),
        TFloat::from(rng.gen_range::<f64, _>(0.0..1.0)).unwrap(),
      );
      let p1 = line_search.optimize(&f, p0);
      let p1 = DistPoint {
        point: p1,
        distance: f(p1)
      };
      (p1.distance > line_search.Δ).then(|| p1)
    })
    .collect();
  let mut points1 = vec![];
  points.into_iter()
    .for_each(|pn| {
      points1.iter()
        .all(|p: &DistPoint<_, _, _>| p.point.distance_to(pn.point) / TFloat::from(2.0).unwrap() > pn.distance)
        .then(|| points1.push(pn));
    });
  points1
}

/// A convenience wrapper around [find_max_parallel], produces an infinite iterator.
pub fn local_maxima_iter<TFloat>(f: impl Fn(P2<TFloat>) -> TFloat + Send + Sync, batch_size: u64, rng_seed: u64, line_search: LineSearch<TFloat>)
    -> impl Iterator<Item = DistPoint<TFloat, TFloat, WorldSpace>>
  where TFloat: Float + Send + Sync
{
  let mut rng = rand_pcg::Lcg128Xsl64::seed_from_u64(rng_seed);

  std::iter::repeat(()).flat_map(move |_|
    find_max_parallel(&f, batch_size, rng.gen(), line_search)
  )
}