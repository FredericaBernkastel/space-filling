//! Plomp–Levelt sensory dissonance, in Sethares' parametrisation.
//!
//! Two pure tones a small interval apart beat against each other, and the
//! sensation peaks at roughly a quarter of the critical bandwidth before falling
//! away again. Sethares fits that curve as a difference of exponentials in the
//! frequency gap, with the bandwidth scaling taken from the lower tone:
//!
//! ```text
//! d(f₁, f₂, a₁, a₂) = a₁·a₂·( e^(−b₁·s·Δf) − e^(−b₂·s·Δf) ),   s = d* / (s₁·f_min + s₂)
//! ```
//!
//! What makes this musical rather than merely psychoacoustic is *complex* tones.
//! Summed over the partial pairs of two harmonic spectra, the curve grows minima
//! exactly at the simple frequency ratios — the octave, fifth, fourth, thirds —
//! which is where consonance comes from in the first place. `consonance_curve` in
//! `main.rs` checks that those minima land where they should before anything else
//! is believed.

/// Sethares' constants: the peak sits at `ln(b₂/b₁)/((b₂−b₁)·s)`, about a quarter
/// of the critical bandwidth.
const D_STAR: f64 = 0.24;
const S1: f64 = 0.0207;
const S2: f64 = 18.96;
const B1: f64 = 3.5;
const B2: f64 = 5.75;

/// Dissonance between two *pure* tones, in arbitrary units proportional to the
/// product of their amplitudes.
///
/// The product rather than `min(a₁, a₂)`, which some presentations use: the
/// product is smooth, and a kink in the field would be a kink in every Lipschitz
/// estimate downstream.
pub fn partial(f1: f64, f2: f64, a1: f64, a2: f64) -> f64 {
  let (lo, hi) = if f1 < f2 { (f1, f2) } else { (f2, f1) };
  let s = D_STAR / (S1 * lo + S2);
  let d = hi - lo;
  a1 * a2 * ((-B1 * s * d).exp() - (-B2 * s * d).exp())
}

/// A harmonic spectrum: partial `i` at `i·f₀` with amplitude `rolloff^(i−1)`.
#[derive(Clone, Debug)]
pub struct Timbre {
  pub partials: Vec<(f64, f64)>,
}

impl Timbre {
  pub fn harmonic(count: usize, rolloff: f64) -> Self {
    Self {
      partials: (1..=count)
        .map(|i| (i as f64, rolloff.powi(i as i32 - 1)))
        .collect(),
    }
  }
}

/// Roughness between two complex tones at fundamentals `f1`, `f2`, scaled by the
/// notes' amplitudes.
///
/// Cross-pairs only. A tone's partials also beat against each other, but that
/// contribution is the same whatever the interval, so it is a constant offset
/// that would only obscure the shape being measured.
pub fn between(timbre: &Timbre, f1: f64, a1: f64, f2: f64, a2: f64) -> f64 {
  let mut sum = 0.0;
  for &(m1, p1) in &timbre.partials {
    for &(m2, p2) in &timbre.partials {
      sum += partial(f1 * m1, f2 * m2, a1 * p1, a2 * p2);
    }
  }
  sum
}

/// Cents above a reference to a frequency ratio.
pub fn cents_to_ratio(cents: f64) -> f64 {
  (cents / 1200.0).exp2()
}
