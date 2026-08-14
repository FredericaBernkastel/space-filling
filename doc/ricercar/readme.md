# Ricercar — counterpoint as a space-filling problem

*Theory, implementation roadmap, and results as they land.*

The name is the argument in miniature. A *ricercar* is the fugue's ancestor, and *ricercare* is Italian for **to
search** — the form was named after the activity this crate performs. Bach's *Ricercar a 6* is the six-voice fugue
of the *Musical Offering*, written on a subject he was handed and asked to develop on the spot.

## Where this stands

Steps 1–4 of §7 are implemented and measured. **Step 5 is blocked twice over**, once on principle and once on cost.

**Established.**

- The roughness field is certifiable — 5 to 9 levels of subdivision resolve a usable margin, against the order of
  279 that made the main crate's anisotropic-body attempt intractable (§7.1).
- A certificate beats any fixed grid: 50 ms sampling pronounces a texture holding a 20 ms dissonance legal, and
  the branch and bound catches it at depth 4 in 31 evaluations (§7.2).
- The legal placement region is **piecewise constant in entry offset, at the note grid** — measured, not argued,
  and it is why fugal onsets are quantized in practice (§7.3).
- The field becomes `ADF`-shaped the moment the threshold moves from the texture to the **pair** (§7.4).
- **BWV 867's subject, from the score** — and its opening falls a fourth, which memory would have inverted (§7.5).
- **`θ_pair ≥ 0.82`, calibrated against Bach's own Stretto II** rather than chosen (§7.6).

**Not established.**

- **No capacity figure survives.** Every number before §7.6 was taken at `θ = 0.30`, which rejects Bach's own
  hyperstretto by a factor of 2.7. The re-run at the calibrated threshold was killed at thirty minutes without
  reaching a single placement.
- No MIDI, and so nothing audible.
- No corpus comparison, and so nothing yet about which subjects stretto well — which was the point (§6.1).

**Three defects the work found in itself**, each surfaced by an independent check rather than by reading the code:

- searching on a *lower* bound blinds the search — caught by a grid scan that found a legal fifth where the greedy
  loop found nothing (§7.5);
- step 1's constant was swept at middle C and understates the bass, where close intervals are far rougher — caught
  by Bach's five-octave texture (§7.6);
- `capacity()` rebuilds the whole field every round, which was cheap at two entries and intractable at ten —
  caught by a timeout (§7.6).

---

## 0. Why this is not merely an analogy

Fugal writing is the one musical discipline that is explicitly a **constraint-satisfaction problem over
transformations of a single shape**. Its literature is a rulebook, its devices are geometric operations, and its
masterworks are studied precisely for how densely they pack material. That is unusually close to what this crate
already does — close enough that the interesting question is not whether the mapping exists but *where it breaks*,
which is §3 and §5 below.

---

## 1. The dictionary

Draw the subject as a curve in **(time, log-pitch)**. The fugal devices are then an affine subgroup acting on that
plane, and one this crate already has combinators for:

| fugal device | affine map | crate |
|---|---|---|
| entry at bar *n* | translation in time | `Translation` |
| transposition | translation in log-pitch | `Translation` |
| inversion | reflection about a pitch axis | `Rotation` by π |
| retrograde | reflection about a time axis | `Rotation` by π |
| augmentation / diminution | scaling in time | `Scale` |

So "state the subject in the alto, up a fifth, inverted, at double length" is
`Scale(Rotation(Translation(subject)))` — a **pose**, exactly the lift of §6.1 of
[`infinite_dimensions`](../publications/infinite_dimensions/readme.md), except that the group is small and mostly
discrete rather than `SO(N)`.

Two consequences fall out immediately.

**Different subjects are different shapes.** A double or triple fugue is two or three shapes that must tile
together, and "which subjects combine well" is "which shapes pack well" — the shape catalogue, not a metaphor for
one.

**Augmentation is literal self-similarity.** Entries at 1×, 2×, 4× duration are the same shape on a geometric
scale ladder — which is the radius ladder of [`examples/argmax2d/01`](../../examples/argmax2d/01_fractal_distribution.rs),
whose measured effective dimension is 1.89. The self-similarity of a fugue's entries and that of a fractal circle
packing are the same structure, and §2.1's diagnostic measures both the same way.

---

## 2. The inversion that has to be fixed first

In packing, shapes **must not** overlap. In counterpoint, voices **must** — simultaneity is the entire art, and
stretto is deliberately overlapping entries. So notes are not what is being packed.

**What is being packed is legality.**

| space filling | counterpoint |
|---|---|
| obstacle | a forbidden configuration — but see §7.2: a roughness field sees *sensory* dissonance, not the stylistic rules |
| clearance `g(x)` | **contrapuntal margin** — how far this placement is from breaking a rule |
| a placed shape lowers the field | each committed voice constrains every future one |
| `max g ≤ 0` | no further legal entry exists: the texture is *full* |

That last row is the one to keep. The clearance sequence `d_k` measures how much room a texture has left, and it
falls monotonically as entries are committed — a genuine space-filling problem with a musical reading.

---

## 3. Where the Lipschitz property lives, and where it does not

This section is the go/no-go, and it is more awkward than §1 and §2 suggest.

### The rulebook is not Lipschitz

Shift an entry's onset by less than a beat and *which notes coincide* changes discretely. "Parallel fifths
present" is therefore piecewise constant in onset time, with jumps wherever a note boundary in one voice crosses
one in another. No Lipschitz constant exists, so no certificate does either.

### A roughness functional is

Replace the predicate with a continuous psychoacoustic functional. Over the sounding pitches at time `t`,

```
R(t) = Σ_{i<j} A_i(t)·A_j(t)·roughness( pitch_i(t) − pitch_j(t) )
g    = min_t ( θ − R(t) )
```

with `roughness` the Plomp–Levelt curve over log-frequency — smooth, bounded, with minima at the simple ratios,
which is where consonance comes from in the first place. Then:

- **in pitch**: Lipschitz, with constant `(V−1)·L_R` for `V` voices, where `L_R` is the roughness curve's maximum
  slope in cents. **`L_R` is register-dependent and this was measured wrongly at first** — §7.1 swept it at middle
  C, and §7.6 found Bach's bass register reaching 0.809 where that sweep peaked at 0.608. The critical bandwidth
  is narrower in hertz but *wider in cents* lower down, so `L_R` should be a function of register rather than a
  scalar;
- **in onset**: Lipschitz *only if notes have amplitude envelopes*. With `A_i(t)` a raised cosine over a few
  milliseconds, a note fades into and out of the texture continuously and the jumps disappear. The constant scales
  as `1/attack`, so **sharp attacks make the certificate expensive** — a real trade, and one to measure rather than
  assume.

### The honest problem: most fugal choices are discrete

Onsets land on a rhythmic grid. Transpositions land on scale degrees. Voice, inversion and retrograde are
categorical. Quantize all of that and there is no continuum left for the ADF to work on, and the machinery has
nothing to do.

**So the continuum has to come from somewhere real.** Three candidates, in increasing order of interest:

1. **Tuning and timing tolerance.** Certify legality not at exact equal temperament but over a ball of tunings and
   micro-timings around it — *robust* counterpoint, legal under any reasonable performance. A genuine continuous
   certificate, but a modest result.
2. **Free canon.** Drop the grid: delay and interval both continuous. This is the classical mathematical object
   and it is genuinely a self-similar packing, but it is no longer a fugue.
3. **The subject itself.** *Do not fix the subject — design it.* Its pitch contour is `N` continuous
   coordinates, and the objective is how much fugal treatment it will bear.

**(3) is the version worth building**, and §6 is about it.

---

## 4. This is the motion-planning problem with a different metric

A voice is a piecewise path in pitch-time — a **trajectory**. Voices must stay apart. The forbidden set is a
separation condition.

> **Counterpoint is collision avoidance in pitch space.**

Voice crossing is two agents passing through each other. Parallel fifths are two agents holding a fixed offset — a
formation constraint. Contrary motion, which every counterpoint text pushes you toward, is the avoidance
manoeuvre. So this shares level 1 and `min_over_curve` with
[the motion-planning plan](../multi_agent_motion_planning/readme.md) and differs only in the metric on the
ambient space. **That is a practical argument for doing them in that order**, not merely a pleasing observation:
the certified path query is the expensive part and it is the same code.

---

## 5. The objection, and what it is actually an objection to

**Maximum clearance is not the goal, and stating it as one was an error.**

The first version of this document said maximin is anti-correlated with sounding good: maximum distance from every
rule boundary is the safest counterpoint, hence the blandest, while real music lives *near* the boundary. That
much is true. The mistake was concluding that the maximin step is therefore the wrong instrument.

It is not the objective. **It is the search.** In circle packing, maximising clearance is how the algorithm
*finds room*; it then places a circle **of that clearance size**, consuming the margin, and repeats until nothing
fits. The result is a densely packed disc, not a maximally cautious one. Transposed:

> maximin locates where an entry can legally go; the entry is then placed as tightly as that margin allows,
> using it up.

So the objective is **densest legal stretto** — minimise entry spacing subject to `g > 0` — not *safest
counterpoint*. That is much closer to what BWV 867 actually does, and it rescues more of the method than the
first draft admitted.

### What remains a real limit

Two things survive the correction, and they are the honest boundary.

**Roughness is instantaneous; counterpoint is preparation and resolution.** A suspension is dissonant *now* and
consonant *next*, and the tension-then-release is the whole effect. A field scoring the sounding interval at each
instant sees a dissonance and marks it down; it cannot see that the dissonance was prepared and resolves. The
fugues most worth imitating are largely made of that device, so this is not a corner case — it is the centre of
the repertoire, and the model is blind to it.

**Chromaticism sits near the threshold on purpose.** A search that respects `g > 0` will still prefer subjects
whose strettos land on thirds and sixths over ones that grind. The austerity of BWV 849 and the friction of BWV
857 come from a willingness to stay close to the boundary, and the closer a target sits to it the less the margin
formulation has to say about why it works.

The division of labour therefore stands, with the roles named more precisely:

- **The ADF certifies and searches.** It proves a texture legal, reports the room remaining, and locates where the
  next entry fits. All three are well defined, and the third is what circle packing has always used maximin for.
- **Something else judges.** "Nice" belongs in the primitive's definition — what counts as roughness, and whether
  it can represent resolution — or in an outer model. Never in the objective.

---

## 6. What to build: densest legal stretto, measured, then designed

### 6.0 The benchmark: BWV 867

**WTC I No. 22 in B flat minor** is the target to aim at, chosen because it is hard rather than because it is
convenient. It is famous precisely for stretto density — including inversion and augmentation, so the whole
transformation group of §1 earns its keep — and if the method cannot say something true about *its* stretto it
cannot say anything about fugue.

It is also, mechanically, near the cheap end:

| property | effect on this machinery |
|---|---|
| slow subject, long note values | `L_t ~ 1/(note duration)` — **the certificate is cheap**, the same inequality that made fast asteroids expensive |
| short, stepwise subject | few continuous coordinates when the contour becomes the variable |
| stretto-saturated, few episodes | a high fraction of the piece is the part that *is* packing |
| **five voices** | 10 simultaneous pairs against 6 for four, and a `(V−1)` factor in the constant — the expensive end |

Two nearby pieces are deliberately *not* the first target. **BWV 849** is a triple fugue, which is the multi-shape
tiling problem and strictly harder. **BWV 857** is the case the roughness model handles worst, its chromatic
subject living exactly where §5 says the model goes blind.

A note worth recording, because it is counter-intuitive: the *Art of Fugue* is the **easier** target for this
machinery, though it is the harder music. It is an explicit, systematic exploration of transformations of one
subject, so its aesthetic content and its combinatorial content very nearly coincide — which is the ideal case for
a method whose entire vocabulary is affine maps of a shape. The Well-Tempered Clavier fugues use counterpoint in
service of affect, and affect is the part that does not transfer.

### 6.1 The measurement

> Given a subject, how many entries can be packed into `V` voices over `T` bars with no pair of voices ever
> crossing the dissonance threshold?

Entries are enumerated discretely (voice × grid onset × scale-degree transposition × transformation); the
**certificate** runs over continuous time and continuous pitch. The output is the clearance sequence `d_k` — the
**contrapuntal capacity** of that subject.

This is worth doing on its own, because it is a number musicians have opinions about but no measure for: **some
subjects stretto densely and some will not, and that difference is precisely what makes a subject good for fugal
treatment.** Bach chose subjects that stretto well; the *Art of Fugue* is the extended demonstration. Ranking a
corpus by measured capacity is falsifiable against what musicians already believe, which makes it a real result
rather than a demo.

> **Blocked.** §7.6 gives the threshold this needs and §7.4 the pipeline, but the two together do not yet run —
> see *Where this stands*. Nothing below is measured.

### 6.2 The design problem

Then invert it. The subject's pitch contour is `N` continuous coordinates; maximise the capacity of §6.1 over
them. `N = 8..16` notes puts this in the `[4, 12]` and `[12, ∞)` bands, and the weights are natural: **the head of
a subject matters more than its tail** (it is what the ear recognises on re-entry), so `γ` decays with note index
and `Manifold::from_extents` takes it directly.

The deliverable is a comparison a musician can judge:

| | measured capacity | notes |
|---|---|---|
| *Art of Fugue* subject | ? | designed for stretto by hand |
| *Musical Offering* royal theme | ? | handed to Bach, not chosen by him |
| a random contour | ? | the control |
| the optimized contour | ? | designed for stretto by geometry |

If the optimized subject scores below Bach's, that is the more interesting result and should be reported as such.

---

## 7. Roadmap

1. ~~**Roughness field and its constant.**~~ **Done — see §7.1. The answer is go.**
2. ~~**Two voices, certified over continuous time.**~~ **Done — see §7.2.** The test as originally written was
   impossible: it asked for a parallel fifth to be reported illegal, and a perfect fifth is one of the *least*
   rough intervals there is (0.089, measured in §7.1). Substituted a brief semitone collision, which the model can
   actually see.
3. ~~**Certify over the placement parameters, not only time.**~~ **Done — see §7.3.**
4. ~~**Capacity for a fixed subject.**~~ **Partly done — see §7.4.** The pipeline runs and the capacity is
   verified, but on a *stand-in* subject rather than BWV 867's, and the MIDI render is not written.
5. **Corpus measurement.** Capacity for a dozen historical subjects against random contours. **Blocked on
   pinning `θ`** — see §7.4: capacity is acutely sensitive to it, so a comparison run at an arbitrary threshold
   measures the threshold rather than the subject. Needs the real BWV 867 subject entered, and MIDI, first.
6. **Subject design (§6.2).** Continuous optimization over the contour, with `Manifold` weights decaying from the
   head. Compare against the corpus.
7. Optional: **double fugue** — two shapes that must tile, which is where the shape-catalogue reading earns its
   keep.

### 7.1 Step 1 result: go

`cargo run --release` in this directory. Sethares' parametrisation of Plomp–Levelt over harmonic spectra of six
partials, `rolloff = 0.88`, raised-cosine envelopes.

**The model is behaving.** Interior minima of the two-tone curve land at **316, 386, 498, 702 and 884 cents** —
the minor third, major third, perfect fourth, perfect fifth and major sixth — with the peak at **81 cents** and
the unison and octave lowest of all. Consonance falls out of the summed partial pairs rather than being put in by
hand, which is the only reason the constants below are worth anything.

**Pitch is cheap.** `L ≈ 0.021` per cent. The steepest point is right against the unison, which is exactly the
failure the plan feared — but the absolute number is small, and the fear was misplaced.

**Onset scales as `1/attack`, as predicted.**

| attack | 1 ms | 5 ms | 20 ms | 50 ms | 100 ms |
|---|---:|---:|---:|---:|---:|
| `L` (per second) | 65.2 | 45.2 | 17.5 | 7.14 | 3.58 |

Doubling the attack from 50 ms to 100 ms halves the constant exactly (7.14 → 3.58), and the relation holds down to
about 20 ms before saturating — below that the note's *pitch* change rather than its envelope sets the slope.

**Verdict: 5 to 9 levels of subdivision** resolve a margin of `0.05·θ` over ±600 cents and ±0.5 s. For comparison,
the anisotropic-body attempt in the main crate needed on the order of **279** and was abandoned. The certificate
is affordable here by two orders of magnitude, and that is the go/no-go answered.

**The numbers are not sampling artefacts.** `min over t` is sampled, so a coarse grid could invent jumps and
inflate every constant. Quadrupling the sample count moves the measured slope from 65.1566 to 65.1566 at a 1 ms
attack and from 17.5282 to 17.5362 at 20 ms — five significant figures and 0.05%. (The secant column bounces,
because it redraws random pairs each run; the swept slope is the reliable statistic.)

**Two things this does not license.** It measures the *two-parameter* case, one interval and one entry offset;
the subject-design problem of §6.2 has its own dimension and needs its own measurement, though pitch being the
cheap axis and onsets being discrete there suggests it will be cheaper rather than dearer. And `min over t` is
**sampled, so it is not yet a certificate** — that is `min_over_curve` from the motion-planning plan, and step 3.

### 7.2 Step 2 result: a proof, not a sample

Two voices, a tritone then a **20 ms semitone** then a fifth — a real dissonance, audible, and short.

**A grid can be defeated, and is.**

| samples | spacing | min seen | verdict |
|---:|---:|---:|---|
| 50 | 50.0 ms | +0.1477 | **LEGAL — wrong** |
| 100 | 25.0 ms | −0.1172 | illegal |
| 10 000 | 0.2 ms | −0.1172 | illegal |

At 50 ms spacing the collision falls between grid points and the texture is pronounced clean. Any fixed density
can be beaten by a short enough dissonance; the only question is who picks the constant last.

**The branch and bound cannot be.** `min over [a,b] of (θ − R) ≥ (θ − R(mid)) − L_t·(b−a)/2` — the crate's own
`sdf_geq_everywhere`, in one dimension over time. It reports *illegal* at every budget, from depth 4 and 31
evaluations upward, and its bound converges from below: −139.3, −8.8, −0.66, −0.151, −0.119 against a true
−0.1172. Sound at every depth, checked by assertion.

**The expensive direction is proving a texture legal**, since a witness settles illegality in one point while
legality needs every instant covered. On a consonant texture:

| attack | measured slope of `R` in `t` | depth | evaluations | certified |
|---:|---:|---:|---:|---:|
| 1 ms | 385.2 | 12 | 8 191 | +0.0261 |
| 20 ms | 19.3 | 8 | 511 | +0.0730 |
| 100 ms | 3.9 | 8 | 361 | +0.2236 |

Cost falls with attack exactly as §7.1 predicted — a gentler envelope is a smaller constant is a cheaper proof —
and the whole range is affordable.

**What step 2 also established, by failing to test what it meant to.** The written test asked for a *parallel
fifth* to be reported illegal. It cannot be: a perfect fifth measured 0.089 in §7.1, among the least rough
intervals there is. Parallel fifths are forbidden for **voice independence**, not for roughness — two voices in
parallel fifths sound like one voice, which is a perceptual objection a pointwise dissonance functional has no
access to. The same holds for unresolved suspensions, and for voice crossing, whose unison is perfectly smooth.

So the roughness field detects **sensory dissonance and nothing else**. Every other rule in §2's table needs its
own primitive, and several of them — parallel motion, preparation, resolution — are not pointwise functions of
the sounding pitches at all. They are functions of *motion*, and a field over instantaneous pitch cannot express
them. That is a larger limitation than this document originally claimed, and §8 now says so.

### 7.3 Step 3 result: the legal region, and what its shape gives away

Step 2 proved one texture legal. A packing needs the *set* of legal placements, so the bound extends over the
product space — a subject in one voice, its answer in another at some transposition and entry offset, certified
over `(cents × onset × time)` at once:

```text
min over the box of (θ − R) ≥ (θ − R(centre)) − Σᵢ Lᵢ·halfᵢ
```

The axes are wildly unlike — 0.042 per cent against 92.5 per second — so the cut goes to `argmax Lᵢ·halfᵢ`. That
is the main crate's `Widest` policy, in the metric the constants define, arrived at for the same reason: halving
the axis that dominates the slack is the only cut that buys anything.

`θ = 0.55`, 20 ms attack, `#` proved legal, `.` a witness found, `?` undecided at depth 18:

```text
   offset  transposition 0 .. 1200 cents
    0.05s  ...???###??......?####?#
    0.10s  ...???###??......?####?#
    0.15s  ...???###??......?####?#
    0.20s  ...???###??......?####?#
    0.25s  ?####??..........?######
    0.30s  ?####??..........?######
    0.35s  ?####??..........?######
    0.40s  ?####??..........?######
    0.45s  ?####??..........?######
    0.50s  ???..........??#########
    0.55s  ???..........??#########
    0.60s  ???..........??#########
    0.65s  ???..........??#########
    0.70s  ???..........??#########
```

38% certified legal, 40% with a witness, 22% undecided at that budget.

**The rows come in bands, and the bands break at 0.25 s.** The subject's notes are quarter-second; offsets inside
one note give identical verdicts, and the pattern changes only when the entry crosses a note boundary. Nothing in
the code knows about the note grid — the bands fall out of the certificate.

That is §3's claim arriving as a measurement rather than an argument. **The legal region really is piecewise
constant in the entry offset, at the note grid**, which is why fugal onsets are quantized in practice, and why a
continuous search over onset has almost nothing to find. It also sharpens §3's conclusion: of the three
candidate continua, the one worth having is not onset but the *subject's own contour*.

Read musically the map is a stretto table. Tight entries admit thirds and the sixth-to-octave band; entries at one
note admit the narrow intervals and the upper band; entries at two notes admit everything from the fifth up. That
is the shape of a real one.

**The cost is the finding to act on.** 143 million evaluations, 425 000 per cell — because each cell re-runs an
independent branch and bound over the full time window, sharing nothing with its neighbours. A single field over
placement space would prune once and reuse it, which is exactly what `ADF` does and exactly why the next step
should hand this to the main crate rather than hand-roll a fourth branch and bound.

`θ` is a free parameter here, not a measured one: it sets how strict the counterpoint is, and the map's shape
moves with it. Nothing in this document derives a principled value, and nothing should pretend to.

### 7.4 Step 4 result: the pipeline closes, on two answers

§7.3 ended by saying to stop hand-rolling branch and bounds and hand the placement search to the main crate. Doing
that needed one reformulation.

**A sum is not a `min`.** Steps 1–3 thresholded the *total* roughness, `θ − Σ_pairs r`, and an `ADF` represents a
minimum of primitives, so no single field can carry a sum. Moving the threshold onto the **pair** — no two voices
exceed `θ_pair` at any instant — gives

```text
g(p) = min over committed e of [ min over t ( θ_pair − r(p, e; t) ) ]
```

which is exactly one primitive per committed entry. It is also the more musical rule: counterpoint constrains the
interval between two voices, not the aggregate roughness of a texture.

**Normalised coordinates retire the constants.** Cents and seconds differ by three orders of magnitude and the
`ADF` charges one scalar against a Euclidean half-diagonal, so each axis is scaled by its own constant — 24 cents
and 40 ms to the unit — and every primitive is then exactly 1-Lipschitz. The same trick as scaling a space-time box
by the asteroid speed.

With that, the whole main-crate pipeline applies unchanged: `ADF` over placement space, `LineSearch` to the
emptiest point, insert, repeat.

| k | onset | cents | interval | `d_k` |
|---:|---:|---:|---|---:|
| 1 | 0.72 s | 825 | minor 6th | 0.0489 |
| 2 | 0.27 s | 1026 | minor 7th | 0.0354 |

**Two answers, then saturation** — and the saturation is checked rather than assumed. The greedy search runs on a
*lower* bound, which is sound but conservative: a loose bound understates clearance everywhere and could hide
legal placements, so termination might be the search giving up rather than the texture being full. An independent
grid scan at verification depth finds a best remaining clearance of **−0.0104**, so it really is full.

**Three things this does not deliver.**

*The subject was not Bach's, and now is.* See §7.5.

*Two points are not a decay curve.* §6.1 promised a capacity *exponent*, and a sequence of length two cannot give
one. Whatever else changes, the texture has to admit enough entries for `d_k` to have a slope.

*And `θ` is doing far too much of the work.* A sustained major third measures 0.263 against a threshold of 0.30, so
this run forbids very nearly everything a real fugue does. Capacity is acutely sensitive to a number §7.3 already
flagged as unprincipled. **Step 5 is blocked on that**: a corpus comparison run at an arbitrary threshold measures
the threshold, not the subjects, and would produce a ranking that looks like a result and is not one.

### 7.5 The real subject, and two things it broke

A score of BWV 867 in MusicXML settled the transcription that no prose analysis would. The prelude is 24 bars and
the fugue 75, which is exactly the file's 99, so fugue bar 1 is file measure 25:

| # | pitch | cents from B♭4 | duration |
|---:|---|---:|---|
| 1 | B♭4 | 0 | half |
| 2 | F4 | **−500** | half |
| — | *rest* | | quarter |
| 3 | G♭5 | **+800** | quarter |
| 4 | F5 | +700 | quarter |
| 5 | E♭5 | +500 | quarter |
| 6 | D♭5 | +300 | quarter |

Every prose claim checks against it. F4 to G♭5 is 13 semitones — the minor ninth, upward, between the second and
third sounding notes. The quarter rest opening bar 2 is the rhetorical pause. The limbs are B♭–F and the
descending tail the episodes are built on. D♭ is the minor third altered to major five times.

**And the opening falls a fourth.** Reading the quoted "B♭–F–G♭" as ascending — the natural reading, and the one
memory supplies — inverts the subject's whole shape. That is why it was worth refusing to guess.

**The real subject broke the search, and the safety net caught it.** The longer subject took the window from 3.2 s
to 7 s, and the greedy search placed *zero* entries while the independent grid scan still found a legal placement
at 712 cents and 0.21 s. The design was wrong: the search ran on the *certified* field, which is a **lower** bound,
and a loose lower bound is negative everywhere. A search wants an optimistic guide with a sound verification
behind it, not a sound bound in front of it. Fixed by guiding with the sampled minimum and certifying only the
placement chosen.

Worth recording what the grid found in that broken run: **a perfect fifth**, which is what a fugal answer is. The
model has no notion of tonality and picked the interval Bach picks.

**Capacity is 1 answer at `θ = 0.30`, and Bach fits five.** Stretto II puts five entries inside two bars. So the
threshold is not merely unprincipled, it is *wrong*, and §7.4's blocker now has its calibration:

> `θ` must be at least large enough that BWV 867's own Stretto II certifies as legal.

That is principled rather than tuned — Bach's five-voice hyperstretto is by construction acceptable counterpoint,
so a threshold rejecting it is wrong about music rather than about the passage. And it falsifies in both
directions: if no plausible `θ` admits that texture under a roughness model, the *model* is inadequate, which
would itself be the finding.

### 7.6 Step 5: θ, calibrated against Bach

§7.5 left the calibration stated but not run:

> `θ_pair` must be at least the largest pairwise roughness Bach's own stretto reaches.

Stretto II is read straight from the score — fugue bars 67–71, file measures 91–95, **74 notes with five entries
beginning inside two bars**, up to seven sounding at once, spanning MIDI 41 to 78. Every note is taken as its own
voice, since the field cares only which pitches coincide. Sampled at 120 000 points with a Lipschitz margin added,
so the figure is an *upper* bound on the maximum and safe to threshold against.

| | |
|---|---:|
| worst **pairwise** roughness | **0.809** (+0.012 margin) |
| worst **total** roughness | 4.754 |
| `θ_pair` used through step 4 | 0.300 |
| `θ_pair` Bach requires | **≥ 0.821** |

**Step 4's threshold rejects Bach's own hyperstretto by a factor of 2.7.** So the capacity of 1 measured in §7.5
was measuring the threshold, which is exactly the failure §7.4 predicted for an unpinned `θ` — now demonstrated
rather than feared.

**And 0.809 exceeds the peak of §7.1's two-tone curve, which was 0.608.** That is not a contradiction, it is a
gap in the earlier measurement: §7.1 swept intervals at middle C, and roughness is register-dependent — the
critical bandwidth is narrower in hertz but wider in *cents* lower down, so a close interval in the bass is far
rougher than the same interval in the treble. Bach's stretto puts voices down to MIDI 41. Any threshold derived
from a single register is wrong for a texture that spans five octaves, and the eventual model should make `θ`
register-dependent rather than scalar.

**Capacity at the calibrated threshold could not be computed.** `θ_pair` is set to 0.82 in the code, and the
re-run was **killed at thirty minutes** without reaching a single placement — where the same loop at `θ = 0.30`
finished in about two. The measurement became intractable at the moment the threshold stopped being wrong.

The bottleneck is identified and it is mine, not the problem's. `capacity()` **rebuilds the whole field every
round**, calling `insert_primitive_domain` over the entire domain for every committed entry, and each primitive
evaluation is a 1200-sample roughness computation. The comment justifying that rebuild says it is "cheaper than
the bookkeeping to update it in place" — which was true at two entries and is false at ten. The `ADF` is built for
incremental insertion and this code throws that away.

So step 5's corpus comparison is blocked a second time, on cost rather than on principle: it runs this loop once
per subject. Fixing the rebuild is the prerequisite, and it is ordinary work rather than a modelling question.

---

## 8. What this will not do

- **Form.** Exposition, episode, middle entries, stretto, pedal, coda. A fugue is narrative; a packing has none.
  Expect a stretto or an invertible-counterpoint puzzle, not a fugue.
- **The interesting combinatorics.** Which transformation, which voice, which key — discrete, and an outer loop
  around everything above.
- **Harmony.** Roughness is not tonality. A texture can be perfectly smooth and harmonically incoherent, and
  nothing here knows the difference between a cadence and a stop.
- **The stylistic rules.** Measured in §7.2 rather than supposed: a roughness field sees sensory dissonance only.
  Parallel fifths are *consonant* by this measure and would never be flagged; voice crossing produces a smooth
  unison; a suspension is indistinguishable from an accident. These are objections about voice independence and
  about motion, and a field over instantaneous pitch cannot express any of them. Counterpoint's rulebook needs
  primitives this model does not have.
- **The 2-approximation guarantee, meaningfully.** It bounds `k`-center in a metric space. Applied to entry
  *diversity* (§6.4's reading) it says something real; applied to musical quality it says nothing at all.
- **Resolution.** The roughness field is instantaneous. It cannot represent a dissonance that is *prepared and
  resolves*, which is the device most of the repertoire worth imitating is built from. Extending the primitive to
  see a short window of time rather than an instant is the obvious repair and is not planned here.
- **Anything about whether the result is good.** §5 is not a caveat to be managed. It is the boundary of the
  method, and the honest claim is a certificate of legality, a search for where material fits, and a measurement
  of capacity — not a composition.
