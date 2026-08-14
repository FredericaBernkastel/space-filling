# Ricercar — counterpoint as a space-filling problem

*Theory and implementation roadmap. Nothing here is built yet.*

The name is the argument in miniature. A *ricercar* is the fugue's ancestor, and *ricercare* is Italian for **to
search** — the form was named after the activity this crate performs. Bach's *Ricercar a 6* is the six-voice fugue
of the *Musical Offering*, written on a subject he was handed and asked to develop on the spot.

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
| obstacle | a forbidden configuration: parallel fifths, unresolved dissonance, voice crossing, out of range |
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
  slope in cents;
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

1. **Roughness field and its constant.** Implement Plomp–Levelt over log-frequency with amplitude envelopes;
   measure the realised Lipschitz constant of `g = min_t(θ − R(t))` in pitch and in onset, as a function of attack
   time. **This is the go/no-go.** If the constant is enormous — because roughness spikes near unisons — the
   certificate proves nothing below the root and the metric needs reconsidering before any music is written.
2. **Two voices, one subject, fixed placements.** Certify legality over continuous time. Test: a texture known to
   contain a parallel fifth is reported illegal, and no sampling density changes the verdict.
3. **`min_over_curve` reuse.** The pitch-time path query is the motion-planning one; if that plan is built first,
   this step is an import.
4. **Capacity for a fixed subject (§6.1), on BWV 867's subject.** Enumerate entries, use maximin to locate room,
   place each entry as tightly as the margin allows, record `d_k`. First audible artifact: render the packed
   stretto to MIDI and listen to it. The honest comparison is against Bach's own stretto in the same piece.
5. **Corpus measurement.** Capacity for a dozen historical subjects against random contours. The validation, and
   the first result worth showing anyone.
6. **Subject design (§6.2).** Continuous optimization over the contour, with `Manifold` weights decaying from the
   head. Compare against the corpus.
7. Optional: **double fugue** — two shapes that must tile, which is where the shape-catalogue reading earns its
   keep.

---

## 8. What this will not do

- **Form.** Exposition, episode, middle entries, stretto, pedal, coda. A fugue is narrative; a packing has none.
  Expect a stretto or an invertible-counterpoint puzzle, not a fugue.
- **The interesting combinatorics.** Which transformation, which voice, which key — discrete, and an outer loop
  around everything above.
- **Harmony.** Roughness is not tonality. A texture can be perfectly smooth and harmonically incoherent, and
  nothing here knows the difference between a cadence and a stop.
- **The 2-approximation guarantee, meaningfully.** It bounds `k`-center in a metric space. Applied to entry
  *diversity* (§6.4's reading) it says something real; applied to musical quality it says nothing at all.
- **Resolution.** The roughness field is instantaneous. It cannot represent a dissonance that is *prepared and
  resolves*, which is the device most of the repertoire worth imitating is built from. Extending the primitive to
  see a short window of time rather than an instant is the obvious repair and is not planned here.
- **Anything about whether the result is good.** §5 is not a caveat to be managed. It is the boundary of the
  method, and the honest claim is a certificate of legality, a search for where material fits, and a measurement
  of capacity — not a composition.
