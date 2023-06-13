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
pub fn find_max_parallel<_Float>(f: impl Fn(P2<_Float>) -> _Float + Send + Sync, batch_size: u64, rng: &mut impl Rng, line_search: LineSearch<_Float>)
                                 -> Vec<DistPoint<_Float, _Float, WorldSpace>>
  where _Float: Float + Send + Sync
{
  use rayon::prelude::*;

  let mut rng_buf = vec![P2::splat(_Float::zero()); batch_size as usize];
  rng_buf.iter_mut().for_each(|x| {
    *x = P2::new(
      _Float::from(rng.gen_range::<f64, _>(0.0..1.0)).unwrap(),
      _Float::from(rng.gen_range::<f64, _>(0.0..1.0)).unwrap(),
    );
  });

  let points: Vec<_> = rng_buf.into_par_iter()
    .filter_map(|p0| {
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
        .all(|p: &DistPoint<_, _, _>| p.point.distance_to(pn.point) / _Float::from(2.0).unwrap() > pn.distance)
        .then(|| points1.push(pn));
    });
  points1
}

/// A convenience wrapper around [find_max_parallel], produces an infinite iterator.
pub fn local_maxima_iter<_Float>(f: impl Fn(P2<_Float>) -> _Float + Send + Sync, batch_size: u64, rng_seed: u64, line_search: LineSearch<_Float>)
                                 -> impl Iterator<Item = DistPoint<_Float, _Float, WorldSpace>>
  where _Float: Float + Send + Sync
{
  let mut rng = rand_pcg::Lcg128Xsl64::seed_from_u64(rng_seed);

  std::iter::repeat(()).flat_map(move |_|
    find_max_parallel(&f, batch_size, &mut rng, line_search)
  )
}