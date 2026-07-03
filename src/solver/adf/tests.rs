use {
  super::*,
  crate::{
    geometry::{Circle, Shape, P2},
    drawing,
    sdf,
    solver::{ADF, LineSearch},
    util
  },
  anyhow::Result,
  image::{Rgba, RgbaImage},
  euclid::{Vector2D, Size2D},
  std::cell::Cell
};
use crate::geometry::DistPoint;

#[test] fn draw_layout() -> Result<()> {
  let mut image = RgbaImage::new(512, 512);
  let mut adf = ADF::new(8, vec![Primitive::new(|_| f64::MAX / 2.0)]);
  let domain = Rect::from_size(Size2D::splat(1.0));

  let t0 = std::time::Instant::now();
  adf.insert_sdf_domain(domain, Arc::new(|p| Circle
    .scale(0.25)
    .translate(Vector2D::splat(0.5))
    .sdf(p)
  ));
  adf.insert_sdf_domain(domain, Arc::new(|p| Circle
    .scale(0.125)
    .translate(Vector2D::splat(0.125))
    .sdf(p)
  ));
  println!("{}us", t0.elapsed().as_micros());

  drawing::display_sdf(|p| adf.sdf(p), &mut image, 4.0);
  adf.tree.draw_layout(&mut image);
  image.save("test/test_adf.png")?;
  Ok(())
}

// profile: 4.85s, 100k circles, adf_subdiv = 7
#[test] #[ignore] fn gradient_adf() -> Result<()> {
  use rand::prelude::*;

  let mut image = RgbaImage::new(1024, 1024);
  let representation = ADF::<f64>::new(7, vec![Primitive::new(sdf::boundary_rect)]);
  let mut primitives = vec![];
  let trials = Cell::new(0u64);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let t0 = std::time::Instant::now();

  util::local_maxima_iter(
    Box::new(|p| representation.sdf(p)),
    32, 0, LineSearch::default()
  ).inspect(|_| trials.set(trials.get() + 1))
    .filter_map(|local_max| {
      let circle = {
        use std::f64::consts::PI;

        let angle = rng.gen_range(-PI..=PI);
        let r = (rng.gen_range(1e-6..1.0).powf(5.0) * local_max.distance)
          .min(1.0 / 6.0);
        let delta = local_max.distance - r;
        // polar to cartesian
        let offset = P2::from([angle.cos(), angle.sin()]) * delta;

        Circle.translate(local_max.point - offset)
          .scale(r)
      };
      // alternately use safe RwLock<ADF> for 1.5x slowdown
      unsafe { representation.as_mut() }.insert_at_maximum(
        local_max,
        Primitive::new(move |p| circle.sdf(p))
      ).then(|| circle)
    })
    .enumerate()
    .take(100000)
    .for_each(|(i, c)| {
      if i % 1000 == 0 { println!("#{}", i); };
      primitives.push(c);
    });

  println!("profile: {}ms", t0.elapsed().as_millis());
  // TODO: Fix sdf insertion method
  /* Here, `adf_error_margin` denotes failed attempts to instert a shape in ADF due to
     imperfect primitive elimination method. See `solver::adf::ADF::higher_all` for more details.
   */
  println!("adf_error_margin: {:+.3e}", trials.get() as f64 / primitives.len() as f64 - 1.0);
  println!("{representation:#?}");
  //drawing::display_sdf(|p| representation.sdf(p), &mut image, 3.5);
  //representation.draw_layout(&mut image);
  use {image::Pixel, drawing::Draw};
  /*primitives.into_iter()
    .for_each(|p| p.texture(image::Luma([255]).to_rgba())
    .draw(&mut image));*/
  representation
    .texture(image::Luma([255]).to_rgba())
    .draw(&mut image);

  image.save("test/test_adf.png")?;
  Ok(())
}

#[test] #[ignore] fn animation() -> Result<()> {
  use rand::prelude::*;
  use drawing::Draw;

  std::fs::create_dir("test\\anim").ok();

  let mut representation = ADF::new(11, vec![Primitive::new(sdf::boundary_rect)]);
  let mut circles = vec![];
  let mut rng = rand_pcg::Pcg64::seed_from_u64(2);

  let mut i = 0;
  'main: while i < 32 {
    let mut local_max = None;
    for _ in 0..50 {
      let p0 = P2::new(
        rng.gen_range(0.0..1.0),
        rng.gen_range(0.0..1.0),
      );
      let ret = LineSearch::default().optimize(|p| representation.sdf(p), p0);
      let ret = DistPoint { distance: representation.sdf(ret), point: ret};
      if ret.distance > 0.0 { local_max = Some(ret); break; }
    };
    let local_max = match local_max {
      Some(r) => r,
      None => {
        println!("failed to find local max, breaking");
        break 'main;
      }
    };

    let mut image = RgbaImage::new(512, 512);
    representation
      .display_sdf(&mut image, 3.5)
      .draw_bucket_weights(&mut image)
      .tree
      .draw_layout(&mut image);
    image.save(format!("test/anim/#{}_0.png", i))?;


    {
      let mut image = image.clone();
      Circle
        .translate(local_max.point.to_vector())
        .scale(local_max.distance)
        .texture(Rgba([0x45, 0x8F, 0xF5, 0x7F]))
        .draw(&mut image);
      image.save(format!("test/anim/#{}_1.png", i))?;
    }

    let circle = {
      use std::f64::consts::PI;

      let angle = rng.gen_range::<f64, _>(-PI..=PI);
      let r = (rng.gen_range::<f64, _>(0.0..1.0).powf(1.0) * local_max.distance)
        .min(1.0 / 6.0);
      let delta = local_max.distance - r;
      let offset = Point2D::from([angle.cos(), angle.sin()]) * delta;

      Circle.translate(local_max.point - offset)
        .scale(r)
    };
    let domain = representation.update_domain(local_max);

    circle.texture(Rgba([0x45, 0x8F, 0xF5, 0xFF]))
      .draw(&mut image);

    image.save(format!("test/anim/#{}_2.png", i))?;
    {
      let mut image = image.clone();
      Circle
        .translate(local_max.point.to_vector())
        .scale(local_max.distance * 4.0)
        .texture(Rgba([0xFF, 0, 0, 0x7F]))
        .draw(&mut image);
      image.save(format!("test/anim/#{}_3.png", i))?;
    }
    representation.tree.draw_bounding(domain, &mut image);
    image.save(format!("test/anim/#{}_4.png", i))?;

    representation.insert_at_maximum(
      local_max,
      Primitive::new(move |p| circle.sdf(p))
    ).then(|| {
      circles.push(circle);
      i += 1;
    });
  };

  println!("{representation:#?}");

  Ok(())
}

// Constructive proof that no constant-sized insertion domain is sound for
// local maxima. Three point-like obstacles at 90°, 210°, 330° around x0 form a
// strict local maximum (contact gaps 120° < 180°). Along the escape bisector
// w = 270°, the field grows as g(x0 + R·w) = √(R² − R·d + d²) ≈ R − d/2, while
// a primitive confined to B̄(x0, d) reaches f(v) ≥ R − d — strictly below g for
// every R (until the outer boundary caps it). Hence the update region extends
// arbitrarily many multiples of d, and the historical 4√2·d square provably
// leaves stale field behind; `ADF::update_domain` covers it exactly.
#[test] fn insertion_domain() {
  let x0 = P2::new(0.5, 0.75);
  let d = 0.05;
  let full = Rect::from_size(Size2D::splat(1.0));

  let mut adf = ADF::new(7, vec![Primitive::new(sdf::boundary_rect)]);
  for angle in [90f64, 210., 330.] {
    let (s, c) = angle.to_radians().sin_cos();
    let obstacle = Circle.scale(1e-4).translate(x0.to_vector() + Vector2D::new(c, s) * d);
    adf.insert_sdf_domain(full, Arc::new(move |p| obstacle.sdf(p)));
  }

  let local_max = DistPoint { point: x0, distance: adf.sdf(x0) };
  // pipeline-style placement: a circle inside the maximal ball, pushed toward w
  let r = 0.01;
  let circle = Circle.scale(r)
    .translate(x0.to_vector() + Vector2D::new(0.0, -1.0) * (local_max.distance - r));
  let f: Arc<dyn Fn(P2<f64>) -> f64 + Send + Sync> = Arc::new(move |p| circle.sdf(p));

  let probe = P2::new(0.48, 0.45); // R = 6·d down the escape ray
  let truth = f(probe).min(adf.sdf(probe));
  assert!(f(probe) < adf.sdf(probe), "the insertion must lower the field at the probe");

  { // the 4√2·d rectangle misses the probe's leaf → stale field
    let mut adf = adf.clone();
    #[allow(deprecated)]
    adf.insert_sdf_domain(util::domain_empirical(local_max), f.clone());
    assert!(adf.sdf(probe) - truth > 0.02,
      "expected the historical constant domain to corrupt the field");
  }

  { // the adaptive domain covers D* → exact field
    let mut adf = adf.clone();
    adf.insert_sdf_domain(adf.update_domain(local_max), f.clone());
    assert!((adf.sdf(probe) - truth).abs() < 1e-12);
  }

  { // the fused D*-pruned walk is exact as well
    let mut adf = adf.clone();
    adf.insert_at_maximum(local_max, Primitive { f: f.clone(), lipschitz: 1.0 });
    assert!((adf.sdf(probe) - truth).abs() < 1e-12);
  }
}

// Counts how often `insert_sdf_domain` returns `false` (the quadtree was not
// changed) under realistic circle-packing insertion. A `false` means the
// redundancy test judged the new primitive to be `>= g` everywhere in its
// domain, so it was discarded. With the sound `sdf_geq_everywhere` this can only
// happen when the primitive genuinely does not lower the field.
//
// `batch`  — maxima found per search round; >1 means later circles in a round
//            are placed against a stale field (an earlier insert may already
//            cover them), a genuine source of "failure".
// `subdiv` — redundancy-test subdivision budget; higher = finer proofs.
#[test] #[ignore] fn bench_insert_failures() {
  use rand::prelude::*;
  use std::f64::consts::PI;

  fn run(batch: u64, subdiv: u32, target: u64, seed: u64) -> (u64, u64) {
    let mut adf = ADF::<f64>::new(7, vec![Primitive::new(sdf::boundary_rect)])
      .with_prune_subdiv(subdiv);
    let ls = LineSearch::default();
    let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
    let (mut attempts, mut failures) = (0u64, 0u64);

    while attempts < target {
      let maxima = util::find_max_parallel(|p| adf.sdf(p), batch, &mut rng, ls);
      if maxima.is_empty() { continue; }
      for m in maxima {
        if attempts >= target { break; }
        let circle = {
          let angle = rng.gen_range(-PI..=PI);
          let r = (rng.gen_range(1e-6..1.0).powf(5.0) * m.distance).min(1.0 / 6.0);
          let delta = m.distance - r;
          let offset = P2::from([angle.cos(), angle.sin()]) * delta;
          Circle.translate(m.point - offset).scale(r)
        };
        attempts += 1;
        if !adf.insert_at_maximum(m, Primitive::new(move |p| circle.sdf(p))) {
          failures += 1;
        }
      }
    }
    (attempts, failures)
  }

  println!("{:>5} {:>6} | {:>8} {:>8} {:>8}", "batch", "subdiv", "attempts", "failures", "rate");
  for &(batch, subdiv, target) in &[
    (32u64, 8u32, 20000u64), // realistic (batch staleness present)
    (1,     8,    20000),    // fresh local max each insert
    (1,     4,    20000),    // coarser proofs
    (1,     12,   20000),    // finer proofs
  ] {
    let t = std::time::Instant::now();
    let (a, f) = run(batch, subdiv, target, 0);
    println!("{:>5} {:>6} | {:>8} {:>8} {:>7.3}%  ({}ms)",
             batch, subdiv, a, f, f as f64 / a as f64 * 100.0, t.elapsed().as_millis());
  }
}

// Measures pruning quality under the hard case: shallow tree (large nodes)
// packed with many small primitives.
//
// Ground truth `B` = the exact field of *every* successfully-inserted primitive
// (no pruning). The pruned tree `A` stores a subset of them per leaf, so
// `g_A(v) >= g_B(v)` always; any point where `g_A(v) > g_B(v)` proves pruning
// discarded a primitive that actually defined the field there — a false
// positive that corrupts the output. Also reports node crowding and the count
// of provably-redundant primitives left unpruned.
#[test] #[ignore] fn bench_pruning_audit() {
  use rand::prelude::*;
  use rayon::prelude::*;
  use std::f64::consts::PI;

  let subdiv = 8u32;
  let mut adf = ADF::<f64>::new(6, vec![Primitive::new(sdf::boundary_rect)])
    .with_prune_subdiv(subdiv);
  let ls = LineSearch::default();
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  // Every primitive whose insert succeeded, plus the initial boundary = ground
  // truth field with no pruning.
  let mut all_prims: Vec<Arc<dyn Fn(P2<f64>) -> f64 + Send + Sync>> =
    vec![Arc::new(sdf::boundary_rect)];

  let (mut attempts, mut failures) = (0u64, 0u64);
  while attempts < 50000 {
    let maxima = util::find_max_parallel(|p| adf.sdf(p), 32, &mut rng, ls);
    if maxima.is_empty() { continue; }
    for m in maxima {
      if attempts >= 50000 { break; }
      let circle = {
        let angle = rng.gen_range(-PI..=PI);
        let r = (rng.gen_range(0f64..1.0).powf(1.0) * m.distance).min(1.0 / 6.0);
        let delta = m.distance - r;
        let offset = P2::from([angle.cos(), angle.sin()]) * delta;
        Circle.translate(m.point - offset).scale(r)
      };
      let f: Arc<dyn Fn(P2<f64>) -> f64 + Send + Sync> = Arc::new(move |p| circle.sdf(p));
      attempts += 1;
      if adf.insert_at_maximum(m, Primitive { f: f.clone(), lipschitz: 1.0 }) {
        all_prims.push(f);
      } else {
        failures += 1;
      }
    }
  }

  // --- collect leaves (rect + bucket) ---
  let mut leaves_v: Vec<(Rect<f64, WorldSpace>, Vec<Primitive<f64>>)> = vec![];
  adf.tree.traverse(&mut |n| { if n.is_leaf() { leaves_v.push((n.rect, n.data.clone())); } Ok(()) }).ok();

  let sizes: Vec<usize> = leaves_v.iter().map(|(_, b)| b.len()).collect();
  let leaves = sizes.len();
  let sum: usize = sizes.iter().sum();
  let max_bucket = *sizes.iter().max().unwrap_or(&0);
  let mut hist = std::collections::BTreeMap::<usize, u64>::new();
  for &s in &sizes { *hist.entry(s / 10 * 10).or_default() += 1; } // 10-wide bins

  println!("=== build: max_depth=6, {} attempts, uniform radius, subdiv={} ===", attempts, subdiv);
  println!("inserted={}  failed={} ({:.3}%)  leaves={}  mean_bucket={:.2}  max_bucket={}",
    all_prims.len() - 1, failures, failures as f64 / attempts as f64 * 100.0,
    leaves, sum as f64 / leaves as f64, max_bucket);
  println!("bucket size bin -> #leaves:");
  for (s, c) in &hist {
    println!("  {:>3}..{:<3}: {:>6}", s, s + 9, c);
  }

  // --- redundant-but-kept (FN): primitives the pruning failed to remove ---
  // Sound (conservative) via 1-Lipschitz SDFs: within a grid cell each `f` moves
  // by <= s, so `f_k - min(others)` (2-Lipschitz) moves by <= s*sqrt2. If it stays
  // >= that margin at every grid node, `f_k >= others` everywhere => f_k never
  // defines the field => provably redundant, so pruning should have dropped it.
  const GL: usize = 96;
  let redundant: u64 = leaves_v.par_iter().map(|(rect, bucket)| {
    let n = bucket.len();
    if n < 2 { return 0u64; }
    let s = rect.size.width / (GL as f64 - 1.0);
    let margin = s * std::f64::consts::SQRT_2;
    let mut min_diff = vec![f64::MAX; n];
    let mut vals = vec![0.0f64; n];
    for iy in 0..GL {
      for ix in 0..GL {
        let v = P2::new(rect.origin.x + ix as f64 * s, rect.origin.y + iy as f64 * s);
        for (i, p) in bucket.iter().enumerate() { vals[i] = (p.f)(v); }
        let (mut min1, mut min2, mut arg1) = (f64::MAX, f64::MAX, 0usize);
        for (i, &val) in vals.iter().enumerate() {
          if val < min1 { min2 = min1; min1 = val; arg1 = i; }
          else if val < min2 { min2 = val; }
        }
        for k in 0..n {
          let others = if k == arg1 { min2 } else { min1 };
          min_diff[k] = min_diff[k].min(vals[k] - others);
        }
      }
    }
    (0..n).filter(|&k| min_diff[k] >= margin).count() as u64
  }).sum();

  println!("=== redundant-but-kept (provable, Lipschitz grid {}^2 / leaf) ===", GL);
  println!("provably-redundant primitives still stored: {} / {} ({:.1}% of stored are dead weight)",
    redundant, sum, redundant as f64 / sum as f64 * 100.0);

  // --- pruning corruption: g_pruned(A) vs g_all(B) on a grid ---
  const G: usize = 256;
  let tol = 1e-6;
  let (corrupt_pts, max_err, sum_err) = (0..G).into_par_iter().map(|iy| {
    let y = (iy as f64 + 0.5) / G as f64;
    let (mut n, mut maxd, mut sumd) = (0u64, 0.0f64, 0.0f64);
    for ix in 0..G {
      let v = P2::new((ix as f64 + 0.5) / G as f64, y);
      let ga = adf.sdf(v);
      let gb = all_prims.iter().map(|f| f(v)).fold(f64::MAX, f64::min);
      let d = ga - gb; // >= 0; > 0 == pruning dropped a relevant primitive
      if d > tol { n += 1; maxd = maxd.max(d); sumd += d; }
    }
    (n, maxd, sumd)
  }).reduce(|| (0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1.max(b.1), a.2 + b.2));

  println!("=== pruning corruption (field g_pruned vs g_all, {0}x{0} grid) ===", G);
  println!("corrupted points: {}/{} ({:.4}%)", corrupt_pts, G * G, corrupt_pts as f64 / (G * G) as f64 * 100.0);
  println!("max field error: {:.4e}   mean error over corrupted pts: {:.4e}",
    max_err, if corrupt_pts > 0 { sum_err / corrupt_pts as f64 } else { 0.0 });
}