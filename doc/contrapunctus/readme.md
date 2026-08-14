# Contrapunctus — counterpoint on the lattice

*Design document. Nothing here is built.*

The alternative to [`doc/ricercar`](../ricercar/readme.md), written after that approach's §8 was traced back to its
causes. It is not a repair of ricercar. It starts from a different category and reaches most of the same questions
with less machinery, and it is written because ricercar's own measurements point at it.

**The claim in one line.** Fugue is a word problem over a finite alphabet, subject to a constraint of bounded
memory — so the right tools are automata, dynamic programming and exact combinatorial search, and the continuum was
never there to begin with.

---

## The name is the argument in miniature

*Ricercare*, to search: that document named its method. *Contrapunctus* — **punctus contra punctum**, point against
point — names its objects. Note against note. The material is discrete, countable, and set against itself, and the
word says so in the first syllable. It is also Bach's own heading for each movement of the *Art of Fugue*, which is
the one place in the repertoire where the combinatorial content and the aesthetic content very nearly coincide.

---

## 0. Where this comes from

Ricercar reached step 5 and stopped, blocked twice: once on principle (the threshold `θ` was unpinned, so a corpus
comparison would measure the threshold rather than the subject) and once on cost (the calibrated run was killed at
thirty minutes without a single placement). §8 of that document lists seven things the method will not do.

The question this answers: *starting over, without space-filling, is there an approach free of most of §8 — and
elegant rather than fitted?*

Yes. And the first evidence for it is in ricercar's own results.

---

## 1. Diagnosis: §8 is two causes, not seven items

| §8 item | cause |
|---|---|
| the stylistic rules — parallel fifths, voice crossing | **the state is a point, not a transition** |
| resolution, suspension | same |
| harmony, "the difference between a cadence and a stop" | **the surrogate is not the thing** — roughness is psychoacoustics, not tonality |
| form | neither; it was simply never modelled |
| the 2-approximation guarantee | an artefact of the geometric framing |
| whether the result is good | irreducible — see §5 |

### 1.1 The state is a point, not a transition

**A parallel fifth is not a property of an instant.** Neither is contrary motion, voice crossing, a suspension, a
cadence, or voice independence in any form. They are properties of *consecutive configurations*. A field over
instantaneous pitch content is not underpowered here — it is structurally incapable of expressing any of them, and
that single fact generates most of §8. Ricercar says as much: *"a field over instantaneous pitch cannot express any
of them."*

The repair is cheap once the diagnosis is stated:

> **Every rule of strict counterpoint is a condition on at most three consecutive events.**

That is not an approximation of the rulebook. It is what the rulebook says, and §2.2 takes it literally.

### 1.2 Ricercar's own evidence against the continuum

Two measurements, neither of them arguments:

- **§7.3.** *"The legal placement region is piecewise constant in entry offset, at the note grid — measured, not
  argued, and it is why fugal onsets are quantized in practice."* That is the continuum announcing it does no work.
  Every certificate in the crate exists to bound a function over a domain whose answer is constant on a lattice.
- **§7.4 against §7.1**, which is arithmetic on two published numbers rather than a finding either section records.
  A certified placement has to be rounded onto the semitone grid before it can be notated. Worst-case rounding is
  25 cents; the pitch constant is `0.021` per cent; so rounding can move the roughness by `≈ 0.5`. The clearances
  actually certified were `0.0489` and `0.0354`. **The rounding is an order of magnitude larger than the margin it
  destroys.**

The second one is the tell. If the answer must be rounded onto a grid at the end, and the rounding is worth ten
times the margin the proof establishes, then the proof was over the wrong set.

---

## 2. The reformulation

### 2.1 Exact arithmetic, and therefore no certificates

Pitch is an integer — semitones, or a scale degree with a chromatic inflection. Time is a rational on a sixteenth
grid. The transformation group of ricercar §1 becomes exact integer and rational arithmetic:

| transformation | operation |
|---|---|
| transposition by `k` | `x + k` |
| inversion about axis `a` | `a − x` |
| retrograde | reverse the word |
| augmentation by `r` | multiply durations by `r ∈ ℚ` |

**No floating point anywhere.** Nothing to quantise, nothing to certify, no Lipschitz constant to measure, no
safety factor to assume. Roadmap steps 1 through 3 of ricercar — the roughness constant, the branch and bound over
time, the branch and bound over placement — do not have counterparts here. They were the cost of a continuum that
§1.2 says was not there.

This also disposes of the register problem §7.6 found the hard way. There is no `L_R` to be register-dependent
about.

### 2.2 Counterpoint is a finite automaton

Read a pair of voices tick by tick. The alphabet at each tick:

```
symbol = ( interval class, motion type, articulation, metric weight )

interval class   (p_upper − p_lower) mod 12, tagged perfect / imperfect / dissonant,
                 plus unison-vs-compound where it matters
motion type      parallel | similar | contrary | oblique — from the signs of the two melodic steps
articulation     which voices strike at this tick, and which are tied over
metric weight    strong | weak — a function of the tick, not of the music
```

Now the rulebook, transcribed:

| rule | as an automaton condition | order |
|---|---|---|
| parallel fifths, octaves | forbidden edge `5 → 5`, `8 → 8` under parallel motion | 2 |
| hidden / direct fifths | edge `* → 5` under similar motion with a leap above | 2 |
| passing dissonance | dissonant tick approached and left by step in one direction | 3 |
| **suspension** | consonant-and-tied → dissonant-on-strong → step down to consonance | 3 |
| neighbour tone | step away and back | 3 |
| voice crossing, overlap | `p_upper(t) ≥ p_lower(t)`, `p_upper(t) ≥ p_lower(t−1)` | 2 |
| leap recovery | a leap beyond a fourth is followed by a step against it | 2–3 |

The state is best described in one phrase: **the interval, plus what you owe.** A leap incurs an obligation to
recover; a dissonance incurs an obligation to resolve; a suspension is an obligation created and discharged. The
obligation set is finite and small precisely because counterpoint requires debts to be settled on the very next
event — that is what "strict" means.

Three things follow.

**The suspension is repaired.** Ricercar's §8 calls preparation-and-resolution *"the device most of the repertoire
worth imitating is built from"*, and the model blind to it. In an automaton a prepared dissonance and an accidental
one are **different paths spelling the same instantaneous interval**. The distinction is free, because the state
remembers where it came from. This is the single largest gain, and it is not a patch — it falls out of using the
right category.

**The parallel fifth is repaired.** §7.2 had to *substitute its own test* because the roughness field rates a
perfect fifth at `0.089`, among the least rough intervals there are, and would never flag a parallel one. Here it
is the canonical forbidden edge — the first thing the automaton knows.

**The rulebook is smaller than the model it replaces.** The crude product of the state components is on the order
of `10³` before minimisation, and DFA minimisation is a solved algorithm. The reachable count after minimisation
should be **measured and reported** rather than asserted — that is step 1 of §8 below, and it is the sort of number
this project prefers to have measured.

### 2.3 Harmony is a second automaton

A functional automaton over `(key, scale degree, inversion)`, with edges for the standard progressions, and
modulation via pivot chords to closely related keys. A cadence is then a **labelled accepting path** — `ii⁶ → V → I`
with its voice leading — and ricercar's *"nothing here knows the difference between a cadence and a stop"* is
answered by construction.

The two automata compose by intersection. This is the classical reading — harmony as a regular language, voice
leading as a transduction over it — and it is worth saying that it is classical, because the components being
known-good is the point.

### 2.4 Form is a grammar

```
Fugue       → Exposition Middle+ Final
Exposition  → Entry (Countersubject Entry){V−1}
Middle      → Episode Entry+
Final       → Stretto? Pedal? Cadence
Episode     → Sequence(motive, transposition pattern, n)
```

with the key plan a bounded walk on the circle of fifths. Ten lines, and §8's first item — *"a fugue is narrative;
a packing has none"* — is repaired by making the narrative the top-level object and letting counterpoint fill the
blocks in.

A packing cannot do this and never could, because a packing has no distinguished order. A grammar is nothing but
order. The mismatch was in the choice of formalism, not in the effort spent on it.

### 2.5 The search is a shortest path

Because the constraints have **bounded memory — order ≤ 3 ticks** — filling free voices against fixed entries is a
shortest path in a layered DAG. Plain Viterbi. Exact, no backtracking, no heuristics, no tuning, no restarts, and
no `LineSearch` that might have been climbing a lower bound.

State size at a tick is the tuple of sounding pitches with their obligations. For two or three voices this is
outright small; at four it wants pruning by the harmonic automaton, which cuts it hard because a chord constrains
every voice at once; at five it wants a beam or a constraint solver. In the solver direction the relevant tool is
Pesant's `regular` global constraint (CP 2004) — a domain-consistent propagator for *"this sequence is accepted by
this DFA"*, which is exactly this problem's shape, and it exists because this shape is common.

**And the cost profile is the right way round.** In a fugue most voices are not free; they are stating the subject.
If `e` entries sound, only `V − e` voices need filling, and in a dense stretto that is zero or one. **The method is
cheapest exactly where the counterpoint is densest** — the opposite of ricercar, where the calibrated stretto was
the run that had to be killed.

One honest boundary. Melodic *shape* rules are not order-3: a single melodic climax, tessitura over a whole phrase,
no repeating a figure. Those are genuinely long-range, and they are what pushes a problem from dynamic programming
to constraint programming. That the boundary falls exactly between the harmonic-contrapuntal rules and the shape
rules is a real finding about counterpoint, and it is falsifiable — name a strict-counterpoint rule with longer
memory and the claim is wrong.

---

## 3. The measurement §6.1 wanted, computed exactly

This is the part to build first, because it is small and it settles the thing that has been blocked twice.

Parameterise an entry as `(τ, d, k)` — transformation, offset in ticks, transposition — with `τ` acting relative to
the entry point. Then for any two entries the sounding interval sequence depends only on

```
( τᵢ , τⱼ , Δd , pitch offset )
```

where `Δd = dⱼ − dᵢ` and the pitch offset is `±kᵢ ± kⱼ` according to which entries are inverted. **Shifting both
entries together changes nothing**, and this holds for the whole transformation group, including retrograde and
augmentation, because every transformation is applied relative to its own entry point.

So the compatibility relation is a **precomputed table**, filled exhaustively by running the §2.2 automaton over
each overlap. Order of magnitude: 36 transformation pairs × 128 offsets × 49 pitch offsets ≈ 2·10⁵ entries, each an
`O(n)` automaton run over a subject of a few dozen ticks. **Milliseconds, once, per subject.**

Then:

> **Densest stretto = maximum clique in the compatibility graph.**

Within a single transformation class the graph is a Cayley graph on the shift group, and a legal stretto is a set
of offsets whose pairwise differences all avoid the bad set — the same object as a Sidon set or a difference
family, with the structure that implies. Across classes it is still one small explicit graph.

Three consequences.

**Infeasibility becomes a proof.** Ricercar's `best_remaining()` exists only because the greedy loop cannot tell "no
legal entry remains" from "the search gave up on a loose lower bound" — a defect the project caught by grid scan
and recorded. A complete search over a finite graph does not have the distinction to make. `best_remaining()`
deletes.

**Clique size is bounded by the voice count**, `V ≤ 5`, so this is a depth-5 search with strong pruning over a few
hundred vertices once the plausible transposition set is fixed. Maximum clique is NP-hard in general and this
instance is not the hard case; if it ever becomes one, that is an ordinary engineering problem with a large
literature behind it, not a modelling question.

**Pairwise legality is necessary, not sufficient** — dissonance treatment and harmony read the whole sonority. So
the clique is an *upper bound*, and each candidate clique is then verified against the full `V`-voice automaton.
That is exact branch and bound with an admissible bound, and it is the same shape as the packing argument it
replaces, only finite.

### 3.1 The calibration disappears

Ricercar §7.6 had to pin `θ` against Bach's own hyperstretto, found `θ_pair ≥ 0.821` against the `0.300` used
throughout step 4, and concluded that *"the measurement became intractable at the moment the threshold stopped
being wrong."*

Here there is no threshold. The calibration becomes a **yes-or-no test**:

> Does BWV 867's Stretto II appear as a clique in its own compatibility graph?

If yes, the automaton is calibrated — by construction, since Bach's five-voice hyperstretto is acceptable
counterpoint. If no, the automaton is too strict and *that is the finding*, exactly as ricercar argued for its own
falsification, but without a constant to fit. Nothing is tuned, and the corpus ranking of §6.1 becomes a loop over
subjects at milliseconds each.

### 3.2 And §6.2 gets easier, not harder

Ricercar's design problem — optimise the subject's contour for capacity — was posed as continuous optimisation over
`N = 8..16` coordinates with `Manifold` weights decaying from the head of the subject. Here a contour is a word,
capacity costs microseconds, and the search is exhaustive or branch-and-bound over words with the head fixed. The
weighting intuition survives intact: fix the head, vary the tail, because the head is what the ear recognises on
re-entry. It is just a search order now rather than a metric.

---

## 4. Space filling, in the right category

The instinct behind ricercar was not wrong. Counterpoint really is a tiling problem. The error was the category:
not packing in `ℝᵈ`, but **factorisation of a finite abelian group**.

A tiling rhythmic canon is a partition of `ℤₙ` into translates of a rhythmic motif — every beat covered exactly
once, no gaps, no overlaps, which is space filling in the strictest sense available. The mathematics is Vuza's
canons, the Coven–Meyerowitz conditions, and Hajós groups, and it has been pursued in a music-theoretic setting by
Andreatta, Amiot and Agon.

It is discrete, it is elegant, it is not fitted to anything, and it is deep. If the aesthetic pull of this project
is *counterpoint as tiling*, that literature is where the pull is actually satisfied — and it is adjacent to §3,
since a difference-set condition on entry offsets is the same kind of object.

---

## 5. What this will not do

Written in ricercar's §8 form, because the point of that section is that it exists.

- **Whether the result is good.** Unchanged and irreducible. A legal fugue is not a beautiful one, and no formalism
  fixes that.
- **But the failure mode inverts, and this is worth stating plainly.** A complete solver does not fail by finding
  nothing; it fails by finding *far too much*. Completeness is not selectivity, and the standard reply — soften the
  constraints, optimise a weighted sum — is where taste re-enters through the back door. That is ricercar §5's
  boundary arrived at from the other side, and it is the same boundary.
- **The rules are stipulated, not derived.** This is the real methodological cost, and it is a genuine loss against
  ricercar. Plomp–Levelt *derives* consonance: §7.1 found interior minima at 316, 386, 498, 702 and 884 cents —
  the minor third, major third, fourth, fifth and major sixth — falling out of summed partial pairs rather than
  being put in by hand. An automaton transcribed from Fux has consonance **stipulated in its alphabet**. It is
  transcription of an explicit theory rather than fitting to data, which is what "elegant, not fitted" asks for,
  but it is not derivation and should not be described as such.
- **A style, and a caricature of one.** Fux is not Bach, and Bach breaks Fux constantly. Whose rulebook goes into
  the automaton is an arguable, inspectable modelling choice — which is better than an unarguable one, but it is
  still a choice, and the output is bounded by it.
- **Melodic invention.** The subject is input. §3.2 makes designing one cheaper, but designing for *capacity* is
  not designing for interest.
- **Robustness.** See §6.
- **Performance.** Expressive timing, dynamics, ornamentation, articulation. The output is a score, not a
  performance.

---

## 6. What ricercar still owns

Not superseded — pointed at a different question, which is the thing the project conflated.

- **Robustness under continuous perturbation.** *"This texture is legal under any tuning within ±20 cents and any
  micro-timing within ±15 ms"* is irreducibly a continuous statement, it is candidate (1) of ricercar §3, and no
  lattice method can produce it. The Lipschitz certificate is the right instrument and this document has nothing
  to say about it.
- **Free canon.** Continuous delay and continuous interval — ricercar §3's candidate (2). Genuinely a continuum,
  genuinely self-similar, and genuinely not a fugue.
- **A derived model of consonance**, per §5 above.

The honest summary is that ricercar answers the robustness question well and the fugue question badly, and that
the two were not distinguished when the domain was chosen.

---

## 7. Prior art

None of this is novel, and that is a feature — the components are known-good and the risk sits in the composition
rather than in the parts. Details should be checked before they are relied on; the lineage is not in doubt.

| | |
|---|---|
| Hiller & Isaacson, *Illiac Suite* (1957) | rule-based counterpoint by generate-and-reject |
| Schottstaedt, *Automatic Counterpoint* (1984) | Fux's species rules, backtracking with penalty weights |
| Ebcioğlu, CHORAL (CMJ 1988) | ~350 rules for Bach chorale harmonisation — the cautionary tale, and the argument for factoring a rulebook into automata rather than listing it |
| Pesant, `regular` constraint (CP 2004) | the domain-consistent DFA-membership propagator of §2.5 |
| Anders, Strasheela; Laurson, PWConstraints | constraint systems built for music |
| Anders & Miranda, ACM Comput. Surv. 43(4), 2011 | the survey to read first |
| Boenn, Brain, De Vos, Fitzgerald, ANTON | the same programme in answer-set programming, which may be the most elegant surface syntax available for it |
| Vuza; Coven–Meyerowitz; Andreatta, Amiot, Agon | tiling rhythmic canons — §4 |

Deliberately excluded: Cope's EMI and everything downstream of it. Recombinant methods are fitted to a corpus by
construction, which is the constraint this document was written under.

---

## 8. Roadmap

Each step is decidable, and each produces a number or a verdict rather than a demo.

1. **The two-voice automaton.** Build it, minimise it, and **report the reachable state count** — measured, in this
   project's habit. Verdict test: reproduce textbook judgements on textbook examples, including the three the
   roughness field got wrong. Parallel fifths flagged; a bare fifth consonant; a suspension distinguished from an
   accidental dissonance of the same interval. §7.2 had to substitute its own test because the field could not do
   the first of those; this is the direct answer to it.
2. **The compatibility table and the clique**, on BWV 867's subject — already entered from the score in
   [`ricercar/src/main.rs`](../ricercar/src/main.rs). Verdict test, per §3.1: *does Bach's Stretto II come out as a
   clique?* Pass calibrates the automaton; fail falsifies it. No constant is fitted either way.
3. **The corpus ranking.** Ricercar §6.1, blocked twice, at milliseconds per subject. The deliverable is the table
   in §6.2 of that document, filled in.
4. **Subject design**, per §3.2 — search over contours with the head fixed.
5. **Realisation.** Viterbi fill of the free voices against the harmonic automaton, then MIDI. The first audible
   output of either document.
6. **Form**, per §2.4. A whole fugue, with the packing question living inside the stretto block where it belongs.

Steps 1 to 3 are the ones that pay for themselves. They are perhaps a few hundred lines and they close a question
that has now been open across two blocked attempts.
