# Multi-agent motion planning as space filling

*Implementation plan. Nothing here is built yet.*

Roadmap step 5 of [`doc/publications/infinite_dimensions`](../publications/infinite_dimensions/readme.md), §6.2 —
recast as a concrete demonstration.

---

## 0. The scenario

Two ships cross a dense field of **moving** asteroids: ship 1 from A to B, ship 2 from B to A. The asteroids are
given data, not decision variables — they may collide with each other freely. The ships must

- reach their endpoints,
- **provably** never touch an asteroid or each other,
- and take the most separated paths the field allows.

Then the same code runs at `d = 2`, `3` and `4` spatial dimensions, with only `const D: usize` changing.

**Moving obstacles are free here**, which is the first thing worth showing. In configuration-space planning they
force a time-augmented state and re-planning; in trajectory space time is already inside the coordinate, so an
asteroid's trajectory is data captured in a closure. The primitive

$$
\mathrm{dist}(x, \text{asteroid}) = \min_t \left( \lVert x(t) - a(t) \rVert - \rho \right)
$$

is 1-Lipschitz in the sup-norm because $\lvert \min_t u - \min_t v \rvert \le \max_t \lvert u - v \rvert$.
Nothing in the field knows the rock is moving.

---

## 1. Architecture: two levels

The single most important design decision, and the one that took a wrong turn first. Putting the whole problem in
trajectory space leaves the tree idle: an asteroid's primitive is *not local* in coefficient space — it is small
for every path passing near that rock at the right moment, a fat curved slab through $\mathbb{R}^{64}$ — so no
cell can prune it and the arena flattens.

The asteroid field *is* local in **space-time**. So:

| level | space | dimension | what works there |
|---|---|---:|---|
| 1 | space-time $(x, t)$ | `d + 1` = 3, 4, 5 | the **tree**: localized tubes, low `D`, millions of queries |
| 2 | spline coefficients | `2·d·K` = 32, 48, 64 | the **certificate and ascent** |

Level 2 queries level 1. Since `ADF` implements `SDF`, a level-2 objective is a closure holding an `Arc<ADF>` —
the two compose with no new machinery.

With only two ships, level 2 needs no ADF of its own: it is `RandomSearch::optimize` over an objective. A
coefficient-space ADF becomes worthwhile only when scaling to many ships, where committed trajectories accumulate
as primitives.

---

## 2. Which basis: spectral or hierarchical?

A path must be a finite coefficient vector `c`, and the choice of basis decides four things at once: the
Lipschitz constant of `c ↦ x_c`, whether endpoint constraints are free, whether the coefficients carry a decaying
importance (so the compact-manifold machinery applies at all), and whether a coefficient's influence is local in
time.

### The three candidates

**Uniform clamped B-splines** — the obvious baseline, and the one to reject.
The basis is non-negative and a partition of unity, so

```
|x_c(t) − x_c'(t)| = |Σⱼ (cⱼ − c'ⱼ)Bⱼ(t)| ≤ maxⱼ|cⱼ − c'ⱼ| · ΣⱼBⱼ(t) = ‖c − c'‖∞
```

giving a Lipschitz constant of **exactly 1** — the best available — and endpoints that are literally the first and
last coefficients. But all coefficients are equivalent by symmetry, so there is **no importance decay**: the domain
is an isotropic cube, `effective_dimension` returns `M`, and every weighted mechanism built in steps 2–4 does
nothing. At `M = 64` that is the regime the stress test showed does not work.

**Spectral** — a straight line carrying the endpoints, plus sine modes that vanish at both:

$$
x_c(t) = A + \tfrac{t}{T}(B - A) + \sum_{k=1}^{K} c_k \sin\!\left(\tfrac{k \pi t}{T}\right)
$$

- **Endpoints are automatic.** Every mode vanishes at $t = 0, T$, so *no coefficient is constrained* and the whole
  box is free. For A→B and B→A this is a real simplification.
- **The weights are exactly §2.3's ellipsoid.** Paths of bounded curvature satisfy $\sum_k k^{2s} c_k^2 \le 1$,
  whose semi-axes are $\gamma_k = k^{-s}$ — literally $\Omega_a$ with $a_k = k^{2s}$. `Manifold` applies with no
  adaptation, and `s` reads physically: `s = 2` is bounded acceleration.
- **The Lipschitz constant is the cost.** $\sup_t \lvert \sum_k c_k \sin(k\pi t/T)\rvert$ is $\Theta(\sqrt{M})$
  over an isotropic ball — about `0.64√M`, so ≈ 5 at `M = 64`.

**Hierarchical B-splines** — coarse wide bumps, then successively narrower ones.

- **Endpoints** carried by the coarsest level; finer levels vanish there. Clean.
- **Weights** decay geometrically by level, $\gamma_\ell \sim 2^{-\ell s}$ with $2^\ell$ coefficients per level —
  also a genuine compact manifold, and levels read as *manoeuvre scale*: one sweeping turn versus a small
  correction.
- **Lipschitz constant ≈ the number of levels**, since only `O(1)` functions per level overlap any `t`. That is
  `≈ log₂ M`, which beats `√M` asymptotically (8 against 16 at `M = 256`) but is a wash at our sizes (≈ 5 versus
  ≈ 5 at `M = 64`).
- **Time locality is retained** — a coefficient affects a bounded window, which spectral does not.

### The comparison that actually decides it

| | uniform B-spline | spectral | hierarchical |
|---|---|---|---|
| Lipschitz `c ↦ x_c` | **1** | `0.64√M` (≈5 at 64) | `≈ log₂M` (≈5 at 64) |
| …in weighted coordinates | n/a | **`‖γ‖₁/‖γ‖₂ ≈ 1.6`** | similar |
| endpoint constraints | fix 2 coefficients | **none needed** | fix coarse level |
| importance decay | **none** | `k^(−s)`, textbook | `2^(−ℓs)`, by level |
| time locality | yes | **no** | yes |
| implementation | trivial | **trivial** | knot bookkeeping |

The `√M` figure overstates the spectral penalty, and this is the point that settles the choice. The certificate
charges `L·h(R)` with `h` the *Euclidean* half-diagonal, while the true bound is the *ℓ¹* half-extent. For cells
whose extents are proportional to `γ` — which is what the domain box is, and what `Widest` approximately
maintains while cutting — the required constant is

```
L ≥ ‖γ‖₁ / ‖γ‖₂ = 1.645 / 1.04 ≈ 1.58        for γₖ = k⁻²
```

**independent of `M`**. The `√M` worst case only applies to isotropic cells, which is not the shape the weighted
domain produces at the depths that matter.

### Recommendation: spectral first, hierarchical held in reserve

Spectral wins on the two things that cost real implementation time — unconstrained endpoints and a `Manifold` that
needs no adaptation — and its Lipschitz penalty is ≈1.6 rather than ≈5 once the weighted geometry is accounted
for. It is also two lines to evaluate.

Hierarchical is the fallback if the constant proves lossier in practice than the analysis suggests, and it is
strictly better on two axes (asymptotic constant, time locality) that would matter for longer horizons or many
more modes.

**The experiment that settles it is cheap and should be run first**: sample coefficient pairs within the weighted
domain, measure `‖x_c − x_c'‖∞ / ‖c − c'‖₂` empirically, and compare against the declared constant. If the
realised ratio tracks 1.6 rather than 5, spectral is settled. That measurement costs an afternoon and removes the
only real uncertainty in this plan.

**Effective dimension either way.** Weights depend on the mode index only, repeated across the `d` spatial axes,
so `Manifold::from_extents` gets `γ_(k,a) = k^(−s)`. The participation ratio is then `d ×` the per-axis figure —
about `2.5d`, so **5 at `d = 2` and 10 at `d = 4`**, against an ambient `M` of 32 and 64.

---

## 3. Level 1 — the space-time field

```rust
// (x, t) with the time axis scaled so speeds are commensurate
let dom = Aabb::new(Point::origin(), Point::from([1.0, 1.0, T * v_max]));
let mut field = ADF::<f64, 3, Orthant>::new_in(dom, 6, vec![walls]);
for rock in &asteroids {
  field.insert_primitive_domain(dom, Primitive::new(rock.sdf()).with_lipschitz(l));
}
```

**The Lipschitz constant is genuinely non-trivial here**, which is the first place in this crate where
`with_lipschitz` stops being decoration. For `f(p, t) = ‖p − a(t)‖ − ρ`:

```
∂f/∂p  has norm 1        ∂f/∂t  has magnitude ≤ v        ⟹  L = √(1 + v²)
```

**Scale the time axis to keep that near 1.** A fast asteroid field means a large `L` and weak pruning; choosing
the time extent as `T·v_max` makes space and time units commensurate and brings `L` back to ≈ `√2`. This is
`new_in` doing something pointed rather than merely enabling high `D`.

**Layout by dimension**, straight from the readme's bands: `Orthant` at `d = 2` (`D = 3`), `Kd` at `d = 3, 4`
(`D = 4, 5`). The layout choice flips across the progression at exactly the documented boundary.

**Inclusion functions.** A moving sphere is monotone in the componentwise distance from its centre only when the
centre is fixed, which it is not. So `Primitive::centred` does not apply directly; the closed form for the minimum
of `‖p − a(t)‖ − ρ` over a space-time box needs deriving — for a linearly-moving rock over a box it reduces to a
point-to-segment distance and is exact. Worth doing, since level 2 hammers this query.

---

## 4. Level 2 — the certified path query

### 4.1 `min_over_curve`, from the start

The naive route samples `t` and takes the minimum, which is **unsound** — sampling steps over the closest approach
and would certify a separation the ships do not have, falsifying the demo's headline claim. Building the certified
query from the beginning, rather than retrofitting it:

```
min over the path ≥ minᵢ lower_bound_over(bbox of segment i) − L · maxᵢ δᵢ
```

where `δᵢ` bounds the spline's deviation from its chord on segment `i`. Two properties make this the right shape:

- **`δ` shrinks as `Δt²`** (chord deviation is second order), where a naive `t`-sampling margin shrinks only as
  `Δt`. Few segments therefore suffice.
- **It uses the tree.** `lower_bound_over` already walks the leaves meeting a box and takes each primitive's own
  bound, so the estimate tightens as level 1 refines — subdivision is buying *certificate quality*, not only query
  speed.

**v1 needs no new traversal code**: split the path into `S` segments and call the existing
`ADF::lower_bound_over` on each segment's bounding box. Sound immediately.

**v2** clips segments against cells properly (a DDA walk through the arena) instead of bounding each whole
segment by its box, which is tighter where a segment crosses several cells. Deferred until v1 is measured.

### 4.2 Sequential formulation

Prioritized planning with a certificate:

1. maximize ship 1's clearance to the asteroids over its coefficients;
2. commit it;
3. maximize ship 2's `min(clearance to asteroids, separation from ship 1)`.

Two optimizations in `ℝ^M`. Cheap, and order-dependent — ship 1 gets the better corridor.

### 4.3 Joint formulation

One optimization in `ℝ^(2M)`:

```
maximize  min( clearance₁, clearance₂, separation₁↔₂ )
```

which is a `min` of primitives, exactly what the representation is for. Symmetric, no priority, and it optimizes
the objective the demo actually claims.

### 4.4 What the comparison should report

| metric | why |
|---|---|
| achieved minimum separation | the objective; joint should win |
| per-ship clearance to asteroids | reveals sequential's asymmetry |
| field evaluations, wall clock | joint pays for `2M`; `RandomSearch` costs `m` per step either way |
| variance across restarts | `2M` dimensions should be more prone to local optima |
| whether joint escapes a configuration sequential cannot | the interesting question |

**Stated prediction, to be falsified rather than confirmed**: joint achieves better worst-case separation and
costs more; sequential is cheaper and systematically unfair to ship 2. If joint does *not* win on separation, the
ascent is finding poor local optima in `2M` dimensions and that is the finding.

---

## 5. The dimension progression

| stage | `d` | level-1 `D` | layout | level-2 `M = 2·d·K` | eff. dim | renderable |
|---|---:|---:|---|---:|---:|---|
| plane | 2 | 3 | `Orthant` | 32 | ≈5 | yes — `(x, y, t)` braid |
| space | 3 | 4 | `Kd` | 48 | ≈7.5 | `(x,y,z)`, `t` animated |
| four | 4 | 5 | `Kd` | 64 | ≈10 | projections and numbers only |

The 4-dimensional stage cannot be drawn, and that is worth leaning into: **the diff between stages is
`const D: usize`**. Nothing to look at is a stronger demonstration of generality than a picture.

---

## 6. Order of work

1. **Measure the basis constant.** Empirical `‖x_c − x_c'‖∞ / ‖c − c'‖₂` over the weighted domain, spectral versus
   hierarchical. Settles §2 before anything depends on it.
2. **Level 1 at `d = 2`.** Space-time `ADF<f64, 3, Orthant>`, a few thousand moving rocks, time axis scaled.
   Report build time, nodes, leaf occupancy — and the query cost against evaluating all primitives directly, which
   is the number that says what the tree contributed.
3. **`ADF::lower_bound_over`-based `min_over_curve` (v1)** plus the chord-deviation bound. Test: a path known to
   graze a rock is reported as grazing, and no sampling density changes the verdict.
4. **Sequential**, two ships, `d = 2`. First braid picture.
5. **Joint**, same scenario. The comparison table of §4.4.
6. **`d = 3`, then `d = 4`.** Only `const D` moves; anything else that has to change is a generality bug.
7. Optional: **`min_over_curve` v2**, and scaling to ~20 ships, where the 2-approximation guarantee stops being
   vacuous and the `dₖ` decay curve starts measuring corridor capacity.

---

## 7. Risks, stated up front

- **"Maximal separation" means a certified local maximum.** GD-ADF is a local method; the global solver is 2-D
  only. The claim is *no collision, proved* plus *locally optimal clearance* — not the best pair of paths.
- **At `k = 2` the 2-approximation guarantee is nearly vacuous.** Gonzalez's bound is about `k`-center over many
  points. The headline must be the certificate, not the guarantee. Twenty ships is where the packing character
  appears.
- **The objective is separation, not efficiency.** Maximin insertion produces maximally *spread* paths, which can
  mean wasteful detours. If short paths that merely avoid collision are wanted, this optimizes the wrong thing.
- **Tubes are spheres for all time.** Fine for ships, wrong for anything with orientation — that is §6.1, deferred.
- **No time-shifting.** The sup-norm correctly compares agents at the same instant, but the schedule is not a free
  variable, so "wait two seconds" is not in the search space.
- **The level-1 inclusion function needs deriving.** `Primitive::centred` does not apply to a moving centre. Until
  the closed form exists, level 1 falls back to the Lipschitz bound and level 2 will be slower than the estimates
  above.
