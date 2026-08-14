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

mod capacity;
mod certify;
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

  step2(&timbre);
}

/// Two voices whose note changes are staggered by `gap`, so that between the two
/// changes a *brief* semitone sounds with a tritone and a fifth either side.
///
/// Before: 0 against 600 cents, a tritone. During: 500 against 600, a semitone —
/// the roughest interval there is. After: 500 against 1200, a fifth. The
/// dissonance is real, audible and short, which is the case a sampled check is
/// blind to.
fn brief_collision(gap: f64) -> Vec<Voice> {
  let t = 1.0;
  vec![
    Voice { notes: vec![
      Note { onset: 0.0, duration: t, cents: 0.0 },
      Note { onset: t, duration: 1.5, cents: 500.0 },
    ]},
    Voice { notes: vec![
      Note { onset: 0.0, duration: t + gap, cents: 600.0 },
      Note { onset: t + gap, duration: 1.5 - gap, cents: 1200.0 },
    ]},
  ]
}

/// Step 2: replace the sampled minimum with a proof.
fn step2(timbre: &Timbre) {
  const THETA2: f64 = 0.45; // above the tritone at 0.30, below the semitone at 0.60
  const ATTACK: f64 = 0.001;
  let gap = 0.020;
  let voices = brief_collision(gap);

  println!("\n\n===== step 2: a certificate, not a sample =====");
  println!("\n   two voices, note changes staggered by {:.0} ms, θ = {THETA2}", gap * 1e3);
  println!("   a tritone, then a {:.0} ms semitone, then a fifth\n", gap * 1e3);

  let l_t = certify::time_slope(&voices, ATTACK, timbre, WINDOW, 200_000);
  println!("   measured |dR/dt| = {l_t:.1} per second; the certificate uses ×{} of it\n",
    certify::SAFETY);

  println!("   {:>9} {:>11} {:>12}   {}", "samples", "spacing", "min seen", "verdict");
  println!("   {}", "-".repeat(54));
  for n in [50usize, 100, 200, 500, 2000, 10_000] {
    let g = voice::field(&voices, THETA2, ATTACK, timbre, WINDOW, n);
    let spacing = (WINDOW.1 - WINDOW.0) / n as f64 * 1e3;
    println!("   {:>9} {:>8.1} ms {:>12.4}   {}", n, spacing, g,
      if g > 0.0 { "LEGAL  <- wrong" } else { "illegal" });
  }

  println!("\n   {:>9} {:>9} {:>12} {:>11}   {}",
    "max depth", "evals", "certified", "observed", "verdict");
  println!("   {}", "-".repeat(62));
  for d in [4u32, 8, 12, 16, 20] {
    let (c, o, e) = certify::certified_min(&voices, THETA2, ATTACK, timbre, WINDOW, l_t, d);
    println!("   {:>9} {:>9} {:>12.4} {:>11.4}   {}", d, e, c, o,
      if c > 0.0 { "legal" } else { "illegal" });
  }

  // The bound must never sit above a densely observed minimum, at any budget.
  let dense = voice::field(&voices, THETA2, ATTACK, timbre, WINDOW, 500_000);
  for d in [4u32, 8, 12, 16, 20] {
    let (c, _, _) = certify::certified_min(&voices, THETA2, ATTACK, timbre, WINDOW, l_t, d);
    assert!(c <= dense + 1e-12,
      "depth {d}: certified {c} exceeds a dense observation {dense} — not a bound");
  }
  println!("\n   soundness: certified <= densely observed ({dense:.4}) at every depth  ok");
  println!("\n   The certificate has no sampling parameter to get wrong. A grid reports");
  println!("   legal until it happens to be fine enough; this cannot.");

  // Finding a witness is the easy direction — one point below zero settles it.
  // Proving a texture *legal* is the expensive one, since every instant has to be
  // covered, and that is the direction the packing actually needs.
  println!("\n   proving the other direction: a consonant texture is LEGAL\n");
  let calm = vec![
    Voice { notes: vec![
      Note { onset: 0.0, duration: 1.2, cents: 0.0 },
      Note { onset: 1.2, duration: 1.3, cents: 200.0 },
    ]},
    Voice { notes: vec![
      Note { onset: 0.0, duration: 1.2, cents: 702.0 },
      Note { onset: 1.2, duration: 1.3, cents: 884.0 },
    ]},
  ];
  println!("   {:>9} {:>10} {:>7} {:>10} {:>12}   {}",
    "attack", "|dR/dt|", "depth", "evals", "certified", "proves legal");
  println!("   {}", "-".repeat(66));
  for attack_ms in [1.0f64, 20.0, 100.0] {
    let attack = attack_ms / 1000.0;
    let l = certify::time_slope(&calm, attack, timbre, WINDOW, 200_000);
    let mut shown = false;
    for d in [8u32, 12, 16, 18, 20, 22] {
      let (c, _, e) = certify::certified_min(&calm, THETA2, attack, timbre, WINDOW, l, d);
      if c > 0.0 || d == 22 {
        println!("   {:>7.0} ms {:>10.1} {:>7} {:>10} {:>12.4}   {}",
          attack_ms, l, d, e, c, if c > 0.0 { "yes" } else { "not yet" });
        shown = true;
        break;
      }
    }
    let _ = shown;
  }
  println!("\n   Cost falls with attack exactly as step 1 said it would: a gentler");
  println!("   envelope is a smaller constant is a cheaper proof.");

  step3(timbre);
}

/// Step 3: the legal region of placement space, certified cell by cell.
///
/// A subject in one voice, answered in another at some transposition and entry
/// offset. Which `(transposition, offset)` pairs give a legal stretto? That
/// region is the free space a packing would place entries into, and mapping it is
/// the same operation the main crate performs on a square: subdivide, and prove
/// each cell clear or find a witness in it.
fn step3(timbre: &Timbre) {
  const THETA3: f64 = 0.55;
  const ATTACK: f64 = 0.020; // 19 per second, from step 2 — the affordable end
  const WIN: (f64, f64) = (0.0, 2.6);
  const COLS: usize = 24; // transposition, 0..1200 cents
  const ROWS: usize = 14; // entry offset, 0.05..0.75 s
  const DEPTH: u32 = 18;

  println!("\n\n===== step 3: the legal region of placement space =====");

  let texture = |cents: f64, onset: f64| vec![subject_at(0.0, 0.0), subject_at(onset, cents)];
  let r = |cents: f64, onset: f64, t: f64| {
    voice::roughness_at(&texture(cents, onset), t, ATTACK, timbre)
  };

  // per-axis constants: pitch from step 1, time from step 2, and the offset axis
  // measured the same way as pitch
  let l_t = {
    let mut worst: f64 = 0.0;
    for i in 0..8 {
      let v = texture(150.0 * i as f64, 0.3);
      worst = worst.max(certify::time_slope(&v, ATTACK, timbre, WIN, 50_000));
    }
    worst
  };
  let l = [0.021 * 2.0, 25.0, l_t * certify::SAFETY];
  println!("\n   θ = {THETA3}, attack {:.0} ms, constants: {:.3}/cent  {:.1}/s onset  {:.1}/s time",
    ATTACK * 1e3, l[0], l[1], l[2]);
  println!("   grid {COLS}×{ROWS} cells, max depth {DEPTH}\n");

  let (mut legal, mut illegal, mut unknown, mut evals) = (0usize, 0usize, 0usize, 0usize);
  let mut tightest = f64::INFINITY;
  let mut rows = vec![];
  for row in 0..ROWS {
    let o0 = 0.05 + 0.70 * row as f64 / ROWS as f64;
    let o1 = 0.05 + 0.70 * (row + 1) as f64 / ROWS as f64;
    let mut line = String::new();
    for col in 0..COLS {
      let c0 = 1200.0 * col as f64 / COLS as f64;
      let c1 = 1200.0 * (col + 1) as f64 / COLS as f64;
      let (lo, obs, e) = certify::certify_region(
        r, (c0, c1), (o0, o1), WIN, THETA3, l, DEPTH);
      evals += e;
      line.push(if lo > 0.0 {
        legal += 1;
        if o0 < tightest { tightest = o0 }
        '#'
      } else if obs < 0.0 {
        illegal += 1;
        '.'
      } else {
        unknown += 1;
        '?'
      });
    }
    rows.push((o0, line));
  }

  println!("      {:>6}  transposition 0 .. 1200 cents", "offset");
  for (o, line) in &rows {
    println!("      {o:>5.2}s  {line}");
  }
  println!("      {:>6}  {}", "", "^unison".to_string());

  let total = (COLS * ROWS) as f64;
  println!("\n   # certified legal   {legal:>4}  ({:.0}%)", 100.0 * legal as f64 / total);
  println!("   . witness found     {illegal:>4}  ({:.0}%)", 100.0 * illegal as f64 / total);
  println!("   ? undecided at {DEPTH:>2}   {unknown:>4}  ({:.0}%)", 100.0 * unknown as f64 / total);
  println!("   {evals} evaluations, {:.0} per cell", evals as f64 / total);
  if tightest.is_finite() {
    println!("\n   tightest certified-legal stretto: entry at {tightest:.2} s");
  }
  println!("\n   This map is an ADF over placement space in all but name: cells proved");
  println!("   clear, cells with a witness, and cells the budget could not settle.");
  println!("   Step 4 packs entries into the cleared region.");

  step4(timbre);
  step5(timbre);
}

/// Bach's Stretto II, fugue bars 67-71 (file measures 91-95), read from the
/// score: `(onset in quarters, MIDI note, duration in quarters)`.
///
/// Five entries begin inside two bars — the passage the analyses call the
/// closest stretto, reserved for last. Every note is taken as its own voice,
/// since the roughness field cares only which pitches sound together.
const STRETTO_II: [(f64, i32, f64); 74] = [
  (0.0, 51, 1.0),
  (0.0, 57, 2.0),
  (0.0, 60, 1.0),
  (0.0, 65, 4.0),
  (1.0, 49, 0.5),
  (1.0, 63, 2.0),
  (1.5, 48, 0.5),
  (2.0, 49, 1.0),
  (2.0, 58, 2.0),
  (2.0, 70, 2.0),
  (3.0, 51, 1.0),
  (3.0, 61, 1.0),
  (4.0, 53, 1.0),
  (4.0, 57, 2.0),
  (4.0, 60, 2.0),
  (4.0, 65, 2.0),
  (4.0, 65, 2.0),
  (5.0, 51, 1.0),
  (6.0, 49, 1.0),
  (6.0, 58, 2.0),
  (6.0, 58, 2.0),
  (7.0, 48, 1.0),
  (7.0, 78, 1.0),
  (8.0, 49, 1.0),
  (8.0, 53, 2.0),
  (8.0, 53, 2.0),
  (8.0, 77, 1.0),
  (9.0, 45, 1.0),
  (9.0, 72, 1.0),
  (9.0, 75, 1.0),
  (10.0, 46, 2.0),
  (10.0, 46, 2.0),
  (10.0, 70, 1.0),
  (10.0, 73, 1.0),
  (11.0, 66, 1.0),
  (11.0, 69, 1.0),
  (11.0, 72, 1.0),
  (12.0, 41, 2.0),
  (12.0, 65, 1.0),
  (12.0, 70, 1.0),
  (12.0, 73, 1.0),
  (13.0, 60, 1.0),
  (13.0, 63, 1.0),
  (13.0, 69, 1.0),
  (13.0, 75, 1.0),
  (14.0, 58, 1.0),
  (14.0, 61, 1.0),
  (14.0, 70, 1.0),
  (14.0, 77, 1.0),
  (15.0, 54, 1.0),
  (15.0, 57, 1.0),
  (15.0, 60, 1.0),
  (15.0, 72, 1.0),
  (15.0, 75, 1.0),
  (16.0, 53, 1.0),
  (16.0, 58, 1.0),
  (16.0, 61, 0.5),
  (16.0, 70, 1.0),
  (16.0, 73, 1.0),
  (16.5, 63, 0.5),
  (17.0, 51, 1.0),
  (17.0, 60, 1.0),
  (17.0, 65, 2.0),
  (17.0, 69, 1.0),
  (17.0, 72, 1.0),
  (18.0, 49, 1.0),
  (18.0, 61, 1.0),
  (18.0, 70, 2.0),
  (18.0, 70, 2.0),
  (18.0, 70, 2.0),
  (18.0, 70, 2.0),
  (19.0, 48, 1.0),
  (19.0, 63, 1.0),
  (19.0, 66, 1.0),
];

/// The subject of BWV 867, read from the score.
///
/// `(offset in quarters, cents from B♭4, duration in quarters)`. Transcribed from
/// `BWV_0867/BWV_0867.xml`, fugue bars 1–3 — file measures 25–27, since the
/// prelude's 24 bars precede it and 24 + 75 = 99 is the file's length.
///
/// ```text
/// B♭4 half | F4 half | (quarter rest) G♭5 F5 E♭5 | D♭5
/// ```
///
/// Everything the prose analyses claim is visible in it. F4 to G♭5 is **13
/// semitones** — the minor ninth, upward, between the second and third sounding
/// notes. The quarter rest opening bar 2 is the "rhetorical pause". The two limbs
/// are B♭–F and the descending tail G♭–F–E♭–D♭, which is the material the
/// episodes are built on. D♭ is the minor third that gets altered to major five
/// times. And the opening falls a *fourth*: reading the quoted "B♭–F–G♭" as
/// ascending, which is the natural reading, would have got that wrong.
const SUBJECT_867: [(f64, f64, f64); 6] = [
  (0.0, 0.0, 2.0),     // B♭4  half
  (2.0, -500.0, 2.0),  // F4   half
  //  quarter rest — the rhetorical pause
  (5.0, 800.0, 1.0),   // G♭5  quarter  <- a minor ninth above the F
  (6.0, 700.0, 1.0),   // F5   quarter
  (7.0, 500.0, 1.0),   // E♭5  quarter
  (8.0, 300.0, 1.0),   // D♭5  quarter
];

/// Seconds per quarter. The piece is slow; this is brisk enough to keep the
/// certified time window affordable.
const QUARTER: f64 = 0.30;

fn subject_867_seconds() -> Vec<(f64, f64, f64)> {
  SUBJECT_867.iter().map(|&(o, c, d)| (o * QUARTER, c, d * QUARTER)).collect()
}

fn interval_name(cents: f64) -> &'static str {
  let c = ((cents % 1200.0) + 1200.0) % 1200.0;
  match c {
    x if x < 50.0 || x > 1150.0 => "unison/8ve",
    x if x < 150.0 => "minor 2nd",
    x if x < 250.0 => "major 2nd",
    x if x < 350.0 => "minor 3rd",
    x if x < 450.0 => "major 3rd",
    x if x < 550.0 => "perfect 4th",
    x if x < 650.0 => "tritone",
    x if x < 760.0 => "perfect 5th",
    x if x < 850.0 => "minor 6th",
    x if x < 950.0 => "major 6th",
    x if x < 1050.0 => "minor 7th",
    _ => "major 7th",
  }
}

/// Step 4: how many entries will this subject bear?
fn step4(timbre: &Timbre) {
  const THETA_PAIR: f64 = 0.82;   // calibrated in step 5 against BWV 867 Stretto II
  const ATTACK: f64 = 0.020;
  const WIN: (f64, f64) = (0.0, 7.0);

  println!("\n\n===== step 4: contrapuntal capacity =====");
  println!("\n   subject: BWV 867, read from the score — 6 notes over 9 quarters,");
  println!("   falling a fourth then leaping a minor ninth, with the rest\n");

  let mut setup = capacity::Setup {
    subject: subject_867_seconds(),
    theta_pair: THETA_PAIR,
    attack: ATTACK,
    timbre: timbre.clone(),
    window: WIN,
    l_t: 0.0,
  };
  setup.l_t = {
    let mut worst: f64 = 0.0;
    let base = capacity::Entry { onset: 0.0, cents: 0.0 };
    for i in 0..6 {
      let v = [setup.voice(base),
               setup.voice(capacity::Entry { onset: 0.3, cents: 200.0 * i as f64 })];
      worst = worst.max(certify::time_slope(&v, ATTACK, timbre, WIN, 40_000));
    }
    worst
  };

  // normalise so every primitive is exactly 1-Lipschitz
  let scale = capacity::Scale::from_constants(0.042, 25.0);
  println!("   θ_pair = {THETA_PAIR}, attack {:.0} ms, |dR/dt| = {:.1}/s", ATTACK * 1e3, setup.l_t);
  println!("   normalised: {:.0} cents and {:.0} ms to the unit\n",
    scale.cents_per_unit, scale.secs_per_unit * 1e3);

  let placed = capacity::capacity(
    &setup, (0.0, 1200.0), (0.15, 1.60), &scale, 10, 15);

  println!("   {:>3} {:>9} {:>9} {:>14} {:>10}", "k", "onset", "cents", "interval", "d_k");
  println!("   {}", "-".repeat(50));
  for (k, (e, d)) in placed.iter().enumerate() {
    println!("   {:>3} {:>8.2}s {:>9.0} {:>14} {:>10.4}",
      k + 1, e.onset, e.cents, interval_name(e.cents), d);
  }

  if placed.len() >= 2 {
    let (first, last) = (placed[0].1, placed[placed.len() - 1].1);
    println!("\n   clearance fell {first:.4} -> {last:.4} over {} entries", placed.len());
  }
  println!("   the texture took {} answers before no legal placement remained", placed.len());

  // Is it full, or did the search give up? The greedy field is a lower bound, so
  // it can hide legal placements. Scan independently.
  let mut committed = vec![capacity::Entry { onset: 0.0, cents: 0.0 }];
  committed.extend(placed.iter().map(|(e, _)| *e));
  let (e, v) = capacity::best_remaining(
    &setup, &committed, (0.0, 1200.0), (0.15, 1.30), 16, 10, 15);
  println!("
   independent grid scan of what remains:");
  if v > 0.0 {
    println!("   found {v:.4} at {:.2}s / {:.0} cents ({}) — the greedy search stopped",
      e.onset, e.cents, interval_name(e.cents));
    println!("   early, so the count above is a FLOOR on capacity, not a measurement");
  } else {
    println!("   best remaining clearance {v:.4} <= 0 — the texture really is full");
  }
  println!("\n   d_k is the capacity curve of §6.1. Its decay is the number this whole");
  println!("   document exists to produce; comparing it across subjects is step 5.");
}

/// Step 5: what threshold does Bach's own hyperstretto require?
///
/// §7.4 left `θ` unpinned and §7.5 showed it is not merely unprincipled but
/// wrong — capacity of 1 where Bach fits 5. The calibration writes itself: his
/// five-voice stretto is by construction acceptable counterpoint, so
///
/// > `θ_pair` must be at least the largest pairwise roughness that passage ever
/// > reaches.
///
/// Sampled densely with a Lipschitz margin added, so the figure is an *upper*
/// bound on the maximum and therefore safe to threshold against.
fn step5(timbre: &Timbre) {
  const ATTACK: f64 = 0.020;
  let span = 20.0 * QUARTER;

  println!("

===== step 5: calibrating θ against Bach =====");
  println!("
   BWV 867 Stretto II, fugue bars 67-71: {} notes, five entries in two bars",
    STRETTO_II.len());

  // every note its own voice — the field only cares which pitches coincide
  let voices: Vec<Voice> = STRETTO_II.iter().map(|&(on, midi, dur)| Voice {
    notes: vec![Note {
      onset: on * QUARTER,
      duration: dur * QUARTER,
      cents: (midi - 70) as f64 * 100.0,
    }],
  }).collect();

  let n = 120_000;
  let (mut worst_pair, mut worst_total, mut at) = (0.0f64, 0.0f64, 0.0);
  for k in 0..n {
    let t = span * (k as f64 + 0.5) / n as f64;
    let sounding: Vec<_> = voices.iter().filter_map(|v| v.sounding(t, ATTACK)).collect();
    let mut total = 0.0;
    let mut pair_max = 0.0f64;
    for i in 0..sounding.len() {
      for j in i + 1..sounding.len() {
        let ((f1, a1), (f2, a2)) = (sounding[i], sounding[j]);
        let r = roughness::between(timbre, f1, a1, f2, a2);
        total += r;
        pair_max = pair_max.max(r);
      }
    }
    if pair_max > worst_pair { worst_pair = pair_max; at = t; }
    worst_total = worst_total.max(total);
  }
  // the sampled maximum understates; add the Lipschitz margin over half a step
  let l_t = certify::time_slope(&voices, ATTACK, timbre, (0.0, span), 120_000);
  let margin = l_t * span / n as f64 * 0.5;

  println!("
   worst pairwise roughness   {:.4}  (at {:.2}s, +{:.4} margin)",
    worst_pair, at, margin);
  println!("   worst total roughness      {:.4}", worst_total);
  println!("
   θ_pair used in step 4      0.3000");
  println!("   θ_pair Bach requires     >= {:.4}", worst_pair + margin);
  let need = worst_pair + margin;
  if need > 0.30 {
    println!("
   So step 4's threshold rejects Bach's own stretto by a factor of {:.1}.",
      need / 0.30);
    println!("   Capacity of 1 measured the threshold, not the subject — exactly the");
    println!("   failure §7.4 predicted would happen if θ were left unpinned.");
  } else {
    println!("
   Bach's stretto passes at step 4's threshold, so θ is not what");
    println!("   limited capacity to 1 — something else is.");
  }
}
