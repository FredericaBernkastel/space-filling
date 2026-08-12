//! Clearance decay as a dimension estimator.
//!
//! The covering radius of a greedy (farthest-first) point set in `[0,1]^N`
//! obeys `d_k ≍ k^(−1/N)`, so on a log–log plot the clearance sequence is a
//! straight line of slope `−1/N`. That makes the slope a *measurement* of the
//! dimension a distribution actually occupies, which need not be the dimension
//! it is embedded in.
//!
//! Two families are recorded to `decay.csv`:
//!
//! - `ff{D}` — farthest-first traversal in `D = 2, 3, 4`. A zero-radius point is
//!   inserted at the deepest spot, so `d_k` *is* the covering radius of the
//!   points together with the domain boundary. Ground truth: slope `= −1/D`.
//! - `frac{n}` — the fractal distribution of `examples/argmax2d/01`, which
//!   inserts a ball of radius `d/n` at each global maximum. Here the exponent is
//!   an empirical property of the distribution, not of the domain.
//!
//! The first family validates the estimator against a known answer; the second
//! applies it where the answer is not known in advance.
//!
//! The maximum is estimated by multistart ascent — the best of `restarts`
//! independent `LineSearch` runs — because in `D > 2` no exact global solver
//! exists, which is the whole point of the report this feeds.

use {
  anyhow::Result,
  rand::prelude::*,
  rayon::prelude::*,
  space_filling::{
    geometry::{Combinator, DistPoint, Hypersphere, Point, VectorExt},
    sdf::{self, SDF},
    solver::{ADF, Argmax2D, LineSearch, Orthant, Primitive},
    util,
  },
  std::{fmt::Write as _, fs, time::Instant},
};

/// Farthest-first traversal in a literal dimension. Written as a macro rather
/// than a generic function: the `2^D`-way tree carries its branching factor as
/// an associated type, so generic code would need a cascade of `Send`/`Sync`
/// bounds that a literal `D` resolves on the spot.
macro_rules! farthest_first {
  ($tag:literal, $D:literal, $k_max:expr, $restarts:expr, $out:expr) => {{
    const D: usize = $D;
    let t0 = Instant::now();
    let ls = LineSearch::default();
    let mut field = ADF::<f64, D, Orthant>::new(5, vec![Primitive::new(sdf::boundary_rect)])
      .with_prune_subdiv(6);

    for k in 1..=$k_max {
      let best = (0..$restarts as u64)
        .into_par_iter()
        .map(|i| {
          let mut rng = rand_pcg::Pcg64::seed_from_u64(((k as u64) << 24) ^ (i + 1));
          let p0 = Point::from(std::array::from_fn::<f64, D, _>(|_| {
            rng.random_range(0.0..1.0)
          }));
          let p = ls.optimize(|q| field.sdf(q), p0);
          DistPoint { point: p, distance: field.sdf(p) }
        })
        .max_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
      let Some(best) = best else { break };

      writeln!($out, "{},{},{:.10}", $tag, k, best.distance).unwrap();
      let c = best.point;
      // A single point is trivially contained in its free ball, so the
      // insertion-domain argument of `insert_at_maximum` applies with room to spare.
      field.insert_at_maximum(
        best,
        Primitive::new(move |p: Point<f64, D>| (p - c).length()),
      );
    }
    eprintln!("  {}: {} insertions, {} restarts, {:?}", $tag, $k_max, $restarts, t0.elapsed());
  }};
}

/// The fractal distribution of `examples/argmax2d/01`: a ball of radius
/// `d / divisor` at each *exact* global maximum, via the discrete solver.
fn fractal_2d(k_max: usize, divisor: f32, resolution: u64, out: &mut String) -> Result<()> {
  let t0 = Instant::now();
  let mut rep = Argmax2D::new(resolution, 64)?;
  rep.insert_sdf(sdf::boundary_rect);

  for k in 1..=k_max {
    let gm = rep.find_max();
    writeln!(out, "frac{},{},{:.10}", divisor, k, gm.distance)?;
    let circle = Hypersphere
      .translate(gm.point.coords)
      .scale(gm.distance / divisor);
    rep.insert_sdf_domain(util::domain_global_max(gm), |v| circle.sdf(v));
  }
  eprintln!("  frac{}: {} circles in {:?}", divisor, k_max, t0.elapsed());
  Ok(())
}

fn main() -> Result<()> {
  let mut out = String::from("series,k,d\n");

  eprintln!("farthest-first traversal (ground truth: slope = -1/D)");
  farthest_first!("ff2", 2, 3000, 96, &mut out);
  farthest_first!("ff3", 3, 1200, 96, &mut out);
  farthest_first!("ff4", 4, 800, 96, &mut out);
  // Control: same dimension and k range, four times the restarts. If the fitted
  // slope is unchanged, the shortfall against -1/D is a small-k effect rather
  // than a failure of the multistart search to find the true global maximum.
  farthest_first!("ff4hi", 4, 400, 384, &mut out);

  eprintln!("fractal distribution (exact global maxima, Argmax2D)");
  fractal_2d(3000, 4.0, 4096, &mut out)?;
  // Maximal balls exhaust the free space far faster, so this series needs a
  // finer grid before the discrete solver's pitch truncates it.
  fractal_2d(3000, 1.0, 8192, &mut out)?;

  fs::write("decay.csv", &out)?;
  eprintln!("wrote decay.csv ({} rows)", out.lines().count() - 1);
  Ok(())
}
