//! Step 4: contrapuntal capacity, on the main crate's field.
//!
//! §7.3 mapped the legal region by re-running an independent branch and bound in
//! every cell, at 425 000 evaluations each and sharing nothing. That is what an
//! `ADF` is for, so this hands the placement search over.
//!
//! **One reformulation makes the field ADF-shaped.** Steps 1–3 thresholded the
//! *total* roughness of the texture, `θ − Σ_pairs r`. A sum is not a `min`, and an
//! `ADF` represents a `min` of primitives, so a sum cannot be carried by one. Move
//! the threshold onto the **pair** — no two voices may exceed `θ_pair` at any
//! instant — and the field becomes
//!
//! ```text
//! g(p) = min over committed e of [ min over t ( θ_pair − r(p, e; t) ) ]
//! ```
//!
//! which is exactly a `min` of one primitive per committed entry. That is also the
//! more musical rule: counterpoint constrains the interval between *two* voices,
//! not the aggregate roughness of the texture.
//!
//! **The coordinates are normalised so the constants vanish.** Cents and seconds
//! carry Lipschitz constants three orders of magnitude apart, and the `ADF`'s
//! certificate charges one scalar against a Euclidean half-diagonal. Scaling each
//! axis by its own constant makes every primitive exactly 1-Lipschitz, which is
//! the same trick as scaling a space-time box by the asteroid speed in the
//! motion-planning plan.

use {
  crate::voice,
  adaptive_distance_field::{
    adf::{Orthant, Primitive, ADF},
    geometry::{Aabb, P2},
    line_search::LineSearch,
    sdf::SDF,
  },
  crate::{
    certify,
    roughness::Timbre,
    voice::{Note, Voice},
  },
};

/// A committed statement of the subject: where it enters, and at what pitch.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
  pub onset: f64,
  pub cents: f64,
}

/// Cents and seconds per normalised unit, chosen so a primitive is 1-Lipschitz.
#[derive(Clone, Copy)]
pub struct Scale {
  pub cents_per_unit: f64,
  pub secs_per_unit: f64,
}

impl Scale {
  pub fn from_constants(l_cents: f64, l_onset: f64) -> Self {
    Self { cents_per_unit: 1.0 / l_cents, secs_per_unit: 1.0 / l_onset }
  }
  pub fn to_entry(&self, u: P2<f64>) -> Entry {
    Entry { cents: u.x * self.cents_per_unit, onset: u.y * self.secs_per_unit }
  }
  pub fn to_unit(&self, e: Entry) -> P2<f64> {
    P2::new(e.cents / self.cents_per_unit, e.onset / self.secs_per_unit)
  }
}

/// Owned, because a `Primitive` closure must be `'static`: it outlives any
/// borrow of the caller's setup.
#[derive(Clone)]
pub struct Setup {
  /// `(offset from entry, cents, duration)` — an explicit offset rather than
  /// notes laid end to end, because BWV 867's subject contains a rest and the
  /// rest is the point: it is the "rhetorical pause" every description names.
  pub subject: Vec<(f64, f64, f64)>,
  pub theta_pair: f64,
  pub attack: f64,
  pub timbre: Timbre,
  pub window: (f64, f64),
  pub l_t: f64,
}

impl Setup {
  pub fn voice(&self, e: Entry) -> Voice {
    Voice {
      notes: self.subject.iter().map(|&(off, c, d)| Note {
        onset: e.onset + off,
        duration: d,
        cents: c + e.cents,
      }).collect(),
    }
  }

  /// A *guide* for the search: the sampled minimum, which is an over-estimate.
  ///
  /// The search must not use the certified bound. That bound is a lower one, and
  /// a loose lower bound is negative everywhere — which is exactly what happened
  /// when BWV 867's subject lengthened the window from 3.2 s to 7 s and the
  /// search went from two placements to none while a grid scan still found a
  /// legal fifth. A guide may be optimistic; the verification behind it is what
  /// has to be sound.
  pub fn pair_guide(&self, a: Entry, b: Entry) -> f64 {
    let voices = [self.voice(a), self.voice(b)];
    voice::field(&voices, self.theta_pair, self.attack, &self.timbre, self.window, 1200)
  }

  /// `min over t (θ_pair − r(a, b; t))`, certified by the step-2 branch and
  /// bound. Used to *verify* a placement the guide proposed, never to find one.
  pub fn pair_clearance(&self, a: Entry, b: Entry, depth: u32) -> f64 {
    let voices = [self.voice(a), self.voice(b)];
    certify::certified_min(
      &voices, self.theta_pair, self.attack, &self.timbre, self.window, self.l_t, depth).0
  }
}

/// Greedy capacity: ascend to the placement with the most room, commit an entry
/// there, and repeat until nothing legal is left.
///
/// Returns the committed entries paired with the clearance each was placed at —
/// the sequence `d_k` whose decay *is* the subject's contrapuntal capacity.
pub fn capacity(
  setup: &Setup,
  domain_cents: (f64, f64),
  domain_onset: (f64, f64),
  scale: &Scale,
  max_entries: usize,
  depth_verify: u32,
) -> Vec<(Entry, f64)> {
  let lo = scale.to_unit(Entry { cents: domain_cents.0, onset: domain_onset.0 });
  let hi = scale.to_unit(Entry { cents: domain_cents.1, onset: domain_onset.1 });
  let domain = Aabb::new(lo, hi);

  // The subject itself, always present, is the first committed entry.
  let mut committed = vec![Entry { onset: 0.0, cents: 0.0 }];
  let mut placed: Vec<(Entry, f64)> = vec![];

  for _ in 0..max_entries {
    // A field over placement space: one primitive per committed entry, plus the
    // domain's own walls. Rebuilt each round — with a handful of entries that is
    // cheaper than the bookkeeping to update it in place.
    let walls = adaptive_distance_field::sdf::boundary_box(domain);
    let mut field = ADF::<f64, 2, Orthant>::new_in(
      domain, 4, vec![Primitive::new(walls)]);

    for &e in &committed {
      let (s, sc) = (setup.clone(), *scale);
      field.insert_primitive_domain(domain, Primitive::new(move |u: P2<f64>| {
        s.pair_guide(sc.to_entry(u), e)
      }));
    }

    // Ascend from a few restarts; the field is not convex and one start is luck.
    let ls = LineSearch { step_limit: Some(60), ..LineSearch::default() };
    let mut best = (P2::new(0.0, 0.0), f64::NEG_INFINITY);
    for k in 0..5 {
      let f = (k as f64 + 0.5) / 5.0;
      let start = P2::new(
        lo.x + (hi.x - lo.x) * f,
        lo.y + (hi.y - lo.y) * (1.0 - f));
      let p = ls.optimize(|q| field.sdf(q), start);
      let v = field.sdf(p);
      if v > best.1 {
        best = (p, v);
      }
    }

    let entry = scale.to_entry(best.0);
    // re-certify the chosen placement properly before believing it
    let verified = committed.iter()
      .map(|&e| setup.pair_clearance(entry, e, depth_verify))
      .fold(f64::INFINITY, f64::min);
    if !(verified > 0.0) {
      break; // the texture is full: no legal entry remains
    }
    placed.push((entry, verified));
    committed.push(entry);
  }
  placed
}

/// Independent check on "the texture is full".
///
/// The greedy loop searches a field built from a *lower* bound, which is sound
/// but conservative: a loose bound understates clearance everywhere and can hide
/// placements that are in fact legal, so termination might be the search giving
/// up rather than the texture being saturated. This scans a coarse grid at the
/// verification depth and reports the best clearance it finds — if that is
/// positive, the greedy loop stopped early and the capacity figure is a floor
/// rather than a measurement.
pub fn best_remaining(
  setup: &Setup,
  committed: &[Entry],
  domain_cents: (f64, f64),
  domain_onset: (f64, f64),
  cols: usize,
  rows: usize,
  depth: u32,
) -> (Entry, f64) {
  let mut best = (Entry { onset: 0.0, cents: 0.0 }, f64::NEG_INFINITY);
  for r in 0..rows {
    for c in 0..cols {
      let e = Entry {
        cents: domain_cents.0
          + (domain_cents.1 - domain_cents.0) * (c as f64 + 0.5) / cols as f64,
        onset: domain_onset.0
          + (domain_onset.1 - domain_onset.0) * (r as f64 + 0.5) / rows as f64,
      };
      let v = committed.iter()
        .map(|&k| setup.pair_clearance(e, k, depth))
        .fold(f64::INFINITY, f64::min);
      if v > best.1 {
        best = (e, v);
      }
    }
  }
  best
}
