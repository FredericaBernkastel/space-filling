//! Voices as pitch-time paths, and the roughness field over a texture.
//!
//! A note is a pitch held over a window, with a raised-cosine attack and release
//! so that it enters and leaves the texture *continuously*. That envelope is not
//! decoration: without it, moving an onset by less than a note changes which
//! notes coincide discretely, the field jumps, and there is no Lipschitz constant
//! to certify with. The price is that the constant scales as `1/attack`, which is
//! what `main.rs` measures.

use crate::roughness::{self, Timbre};

/// Reference pitch: cents are measured from here.
pub const F_REF: f64 = 261.626; // middle C

#[derive(Clone, Copy, Debug)]
pub struct Note {
  pub onset: f64,
  pub duration: f64,
  /// Above [`F_REF`].
  pub cents: f64,
}

#[derive(Clone, Debug)]
pub struct Voice {
  pub notes: Vec<Note>,
}

impl Voice {
  /// The fundamental and amplitude sounding at `t`, if any.
  ///
  /// Raised cosine in and out over `attack`, clamped so a note shorter than two
  /// attacks still rises and falls rather than inverting.
  pub fn sounding(&self, t: f64, attack: f64) -> Option<(f64, f64)> {
    for n in &self.notes {
      let end = n.onset + n.duration;
      if t < n.onset || t >= end {
        continue;
      }
      let a = attack.min(n.duration / 2.0);
      let env = if a <= 0.0 {
        1.0
      } else if t < n.onset + a {
        0.5 * (1.0 - (std::f64::consts::PI * (t - n.onset) / a).cos())
      } else if t > end - a {
        0.5 * (1.0 - (std::f64::consts::PI * (end - t) / a).cos())
      } else {
        1.0
      };
      return Some((F_REF * roughness::cents_to_ratio(n.cents), env));
    }
    None
  }
}

/// Total roughness across every pair of voices sounding at `t`.
pub fn roughness_at(voices: &[Voice], t: f64, attack: f64, timbre: &Timbre) -> f64 {
  let sounding: Vec<_> = voices.iter().filter_map(|v| v.sounding(t, attack)).collect();
  let mut sum = 0.0;
  for i in 0..sounding.len() {
    for j in i + 1..sounding.len() {
      let ((f1, a1), (f2, a2)) = (sounding[i], sounding[j]);
      sum += roughness::between(timbre, f1, a1, f2, a2);
    }
  }
  sum
}

/// The field: `g = min over t of (θ − R(t))`, positive where the texture stays
/// under the threshold.
///
/// Sampled in `t`. That is unsound as a *certificate* — sampling can step over
/// the worst instant — and the fix is the chord-bounded `min_over_curve` of the
/// motion-planning plan. For *measuring the Lipschitz constant* it is fine, and
/// the sampling has to be fine enough not to invent jumps of its own, which
/// `main.rs` checks by refining it.
pub fn field(
  voices: &[Voice],
  theta: f64,
  attack: f64,
  timbre: &Timbre,
  window: (f64, f64),
  samples: usize,
) -> f64 {
  let (t0, t1) = window;
  let mut worst = f64::INFINITY;
  for k in 0..samples {
    let t = t0 + (t1 - t0) * (k as f64 + 0.5) / samples as f64;
    let g = theta - roughness_at(voices, t, attack, timbre);
    if g < worst {
      worst = g;
    }
  }
  worst
}
