//! Roadmap step 1: does the roughness field have a usable Lipschitz constant?
//!
//! The certificate clears a cell when the centre margin covers `L·h(R)`. So the
//! whole ricercar idea rests on `L` being small enough that cells of a musically
//! meaningful size get cleared — if resolving a margin needs cells narrower than
//! a cent or shorter than a millisecond, the proof settles at the root and
//! proves nothing.
//!
//! Three measurements, in order of what they can rule out:
//!
//! 1. **The consonance curve.** Sanity: the minima must land on the simple
//!    ratios. If they do not, the model is wrong and the constants are noise.
//! 2. **`L` in pitch.** How steep the field is as a voice is transposed.
//! 3. **`L` in onset, against attack time.** The one predicted to hurt, since a
//!    note's envelope is what makes the field continuous in time at all and the
//!    constant scales as `1/attack`.
//!
//! Both constants are measured two ways: a central difference (the local slope)
//! and the largest secant over random pairs (a lower bound on the true constant,
//! which also catches jumps a derivative would step over).

mod roughness;
mod voice;

use {
  rand::prelude::*,
  roughness::Timbre,
  voice::{Note, Voice, F_REF},
};

/// Threshold: the texture is legal while `R(t) < THETA`. Set from the measured
/// scale of the curve rather than chosen — see `consonance_curve`.
const THETA: f64 = 1.0;
const WINDOW: (f64, f64) = (0.0, 2.5);
const SAMPLES: usize = 5000;

/// Two sustained tones, the second swept over an octave.
fn two_tones(cents: f64) -> Vec<Voice> {
  vec![
    Voice { notes: vec![Note { onset: 0.0, duration: 2.0, cents: 0.0 }] },
    Voice { notes: vec![Note { onset: 0.0, duration: 2.0, cents }] },
  ]
}

/// A four-note motif — the shortest thing that behaves like a subject.
const SUBJECT: [(f64, f64); 4] = [(0.0, 0.25), (700.0, 0.25), (500.0, 0.25), (300.0, 0.25)];

fn subject_at(onset: f64, transpose: f64) -> Voice {
  let mut t = onset;
  Voice {
    notes: SUBJECT
      .iter()
      .map(|&(c, d)| {
        let n = Note { onset: t, duration: d, cents: c + transpose };
        t += d;
        n
      })
      .collect(),
  }
}

/// Stretto: the same subject answered at the fifth, entering `offset` later.
///
/// Unlike two sustained tones this actually *depends* on the offset — shifting
/// the entry changes which note of the answer sounds against which note of the
/// subject, which is the whole reason onset is a hard axis.
fn stretto(offset: f64) -> Vec<Voice> {
  vec![subject_at(0.0, 0.0), subject_at(offset, 700.0)]
}

fn field_n(voices: &[Voice], attack: f64, timbre: &Timbre, samples: usize) -> f64 {
  voice::field(voices, THETA, attack, timbre, WINDOW, samples)
}

/// The model is only worth measuring if its minima are the consonances.
fn consonance_curve(timbre: &Timbre) {
  println!("\n1. consonance curve — roughness against interval, two harmonic tones\n");
  let named = [
    (0.0, "unison"), (386.3, "major 3rd"), (498.0, "perfect 4th"),
    (701.9, "perfect 5th"), (884.4, "major 6th"), (1200.0, "octave"),
    (100.0, "minor 2nd"), (600.0, "tritone"),
  ];

  // dense sweep, so local minima can be located rather than assumed
  let n = 2400;
  let curve: Vec<(f64, f64)> = (0..=n)
    .map(|i| {
      let c = 1200.0 * i as f64 / n as f64;
      let f2 = F_REF * roughness::cents_to_ratio(c);
      (c, roughness::between(timbre, F_REF, 1.0, f2, 1.0))
    })
    .collect();

  let peak = curve.iter().fold(0.0f64, |m, &(_, r)| m.max(r));
  println!("   peak roughness {peak:.3} at {:.0} cents",
    curve.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0);

  let minima: Vec<f64> = (1..curve.len() - 1)
    .filter(|&i| curve[i].1 < curve[i - 1].1 && curve[i].1 <= curve[i + 1].1)
    .map(|i| curve[i].0)
    .collect();
  println!("   interior minima at: {}",
    minima.iter().map(|c| format!("{c:.0}")).collect::<Vec<_>>().join(", "));

  println!("\n   {:>12} {:>8} {:>10}", "interval", "cents", "roughness");
  println!("   {}", "-".repeat(32));
  for (c, name) in named {
    let f2 = F_REF * roughness::cents_to_ratio(c);
    println!("   {:>12} {:>8.0} {:>10.4}", name, c, roughness::between(timbre, F_REF, 1.0, f2, 1.0));
  }
}

/// Central difference and worst secant, over a swept axis.
fn lipschitz(
  label: &str,
  attack: f64,
  timbre: &Timbre,
  lo: f64,
  hi: f64,
  step: f64,
  build: impl Fn(f64) -> Vec<Voice>,
  rng: &mut impl Rng,
  samples: usize,
) -> (f64, f64) {
  let field = |v: &[Voice], a: f64, t: &Timbre| field_n(v, a, t, samples);
  let mut worst_slope: f64 = 0.0;
  let mut at = lo;
  let mut x = lo;
  while x <= hi {
    let a = field(&build(x - step), attack, timbre);
    let b = field(&build(x + step), attack, timbre);
    let slope = (b - a).abs() / (2.0 * step);
    if slope > worst_slope {
      worst_slope = slope;
      at = x;
    }
    x += step;
  }

  // secants over random pairs: a lower bound on the true constant, and unlike
  // the derivative it cannot step over a jump
  let mut worst_secant: f64 = 0.0;
  for _ in 0..400 {
    let u = lo + (hi - lo) * rng.random::<f64>();
    let v = lo + (hi - lo) * rng.random::<f64>();
    if (u - v).abs() < 1e-9 {
      continue;
    }
    let s = (field(&build(u), attack, timbre) - field(&build(v), attack, timbre)).abs()
      / (u - v).abs();
    worst_secant = worst_secant.max(s);
  }
  println!("   {label:>22}  slope {worst_slope:>9.4}  (at {at:>7.1})   secant {worst_secant:>9.4}");
  (worst_slope, worst_secant)
}

/// Levels of subdivision to resolve a margin of `eps` over a domain of
/// half-extent `h0`, given a constant `l`.
fn levels(h0: f64, l: f64, eps: f64) -> f64 {
  (h0 * l / eps).max(1.0).log2()
}

fn main() {
  let timbre = Timbre::harmonic(6, 0.88);
  let mut rng = rand_pcg::Pcg64::seed_from_u64(0x21C_E12A);

  consonance_curve(&timbre);

  println!("\n2. Lipschitz constant in pitch — cents, over an octave\n");
  // finely near the unison, where the curve is steepest — the concern the plan
  // named explicitly
  let (near_slope, near_secant) = lipschitz(
    "0..120 cents", 0.010, &timbre, 0.4, 120.0, 0.2,
    two_tones, &mut rng, SAMPLES);
  let (far_slope, far_secant) = lipschitz(
    "0..1200 cents", 0.010, &timbre, 1.0, 1199.0, 1.0,
    two_tones, &mut rng, SAMPLES);
  let pitch_l = near_slope.max(near_secant).max(far_slope).max(far_secant);

  println!("\n3. Lipschitz constant in onset — seconds, against attack time\n");
  println!("   {:>22}  {:>15}  {:>7}  {:>16}",
    "attack", "slope (per s)", "", "secant (per s)");
  let mut onset_rows = vec![];
  for attack_ms in [1.0f64, 5.0, 20.0, 50.0, 100.0] {
    let attack = attack_ms / 1000.0;
    let (s, sec) = lipschitz(
      &format!("attack {attack_ms:>5.0} ms"), attack, &timbre, 0.02, 1.20, 0.002,
      stretto, &mut rng, SAMPLES);
    onset_rows.push((attack_ms, s, sec));
  }

  // `min over t` is sampled, so a coarse grid can invent jumps of its own and
  // inflate every constant above. At a 1 ms attack the envelope spans about two
  // samples, which is exactly where that would happen — so refine and watch.
  println!("\n3b. is the time sampling fine enough, or inventing jumps?\n");
  for attack_ms in [1.0f64, 20.0] {
    for mult in [1usize, 2, 4] {
      let n = SAMPLES * mult;
      lipschitz(
        &format!("{attack_ms:>3.0} ms, {n:>6} samples"), attack_ms / 1000.0, &timbre,
        0.02, 1.20, 0.002, stretto, &mut rng, n);
    }
  }

  println!("\n4. verdict — subdivisions to resolve a margin of 0.05·θ\n");
  println!("   pitch domain ±600 cents, onset domain ±0.5 s, θ = {THETA}\n");
  println!("   {:>22} {:>10} {:>10}", "axis", "L", "levels");
  println!("   {}", "-".repeat(46));
  let eps = 0.05 * THETA;
  println!("   {:>22} {:>10.3} {:>10.1}", "pitch (cents)", pitch_l,
    levels(600.0, pitch_l, eps));
  for (ms, s, sec) in &onset_rows {
    let l = s.max(*sec);
    println!("   {:>22} {:>10.3} {:>10.1}", format!("onset, {ms:.0} ms attack"), l,
      levels(0.5, l, eps));
  }

  println!("\n   A cell is cleared when the centre margin covers L·h. Levels above ~20");
  println!("   mean the branch-and-bound is resolving below a cent or below a");
  println!("   millisecond, and the certificate is not buying anything musical.");
}
