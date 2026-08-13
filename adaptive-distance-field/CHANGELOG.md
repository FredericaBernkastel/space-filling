# Changelog

## Unreleased — selectable tree layout

The backing tree gained a **compile-time subdivision layout**. `Orthant` is the
historical `2^N`-way split; `Kd` splits one axis per level, cycling `depth % D`.
The layout has **no default**: it is a decision with a measurable cost either way,
and reads badly as an omission.

```rust
let octree = ADF::<f64, 3, Orthant>::new(6, init);   // all three axes at once
let kd     = ADF::<f64, 6, Kd>::new(3, init);        // one axis per level
let wide   = ADF::<f64, 8, Kd>::new(2, init);        // D > 6, previously impossible

// or name the parameters one at a time, in any order
let field = adf::builder().f64().dims::<6>().kd().prune_subdiv(3).bounded(2);
```

### Breaking

| before | after |
|---|---|
| `ADF<Float, D>` | `ADF<Float, D, L>` — the layout must be named |
| `Quadtree<Data, Float, DIMS>` (struct) | `Tree<Data, Float, DIMS, L>`; `Quadtree` survives as an alias fixing `L = Orthant` |
| `Refine<Data, DIMS>` | `Refine<Data, DIMS, L>` |
| `ADF::new(max_depth, ..)` counted tree levels | counts **full subdivisions** — halvings of every axis — so one budget means one resolution in either layout (`× D` levels for `Kd`) |
| `ADF::tree: Quadtree<..>` | `ADF::tree: Tree<.., L>` |
| node payload `Vec<Primitive<..>>` | `Bucket<..>`; read leaves with `node.data.primitives()` |
| `Refine::Subdivide(children)` | `Refine::Subdivide { parent, children }`, children being [`Split`]s |

Every in-repo call site is migrated: 4 examples, 3 integration tests, 2 doctests,
`doc/clearance_decay`, `doc/video2`. Nothing else in the public surface moved —
`Node`, `Primitive`, `sdf_geq_everywhere`, `Dim`, `Branching`, `Children`,
`child_rect`, `child_rects` and every insertion and query method are untouched,
and bounds simply moved from `Dim<D>: Branching` to `L: Layout<D>`.

### Added

| item | notes |
|---|---|
| `adf::tree` | the tree module's new name; `adf::quadtree` re-exports it |
| `adf::tree::Layout<const DIMS>` | the strategy trait: `CHILDREN`, `LEVELS_PER_SPLIT`, `NAME`, `Children<T>`, `child_rect`, `child_index` |
| `adf::Orthant`, `adf::Kd` | the two zero-sized layout markers, re-exported next to `ADF` |
| `adf::tree::Tree<Data, Float, DIMS, L>` | the arena, generic over layout |
| `adf::tree::KdTree<Data, Float, DIMS>` | alias for `Tree<.., Kd>` |
| `adf::tree::cut_axis::<DIMS>(depth)` | which axis a `Kd` node at `depth` halves; rejects `DIMS = 0` at compile time |
| `Tree::leaf_count`, `Tree::arena_bytes` | were only reachable through `Debug` before |
| `adf::Bucket<Float, D>` | what a node holds: a leaf's primitives, or the bound a divided node froze |
| `adf::tree::Split<Data, DIMS, L>` | a subtree to graft onto one leaf, so a single `Refine::Subdivide` can be several levels deep |
| `ADF::with_split_round` | levels one overflow may divide through; 1 by default |
| `ADF::with_bucket_size` | primitives a leaf holds before dividing; 3 by default |
| `adf::tree::Layout::CARRY_CELL` | whether a descent carries the cell instead of loading each node's rect |
| `adf::builder()` → `AdfBuilder` | type-state builder: `scalar`/`f64`/`f32`, `dims::<D>`, `layout`/`orthant`/`kd`, `prune_subdiv`, then `build`/`bounded`. Each call pins **one** parameter, in any order; an unpinned slot stays `Unset`, which satisfies none of `build`'s bounds, so a forgotten refinement is a compile error at `build` rather than a default |
| `ADF::memory_bytes`, `ADF::layout_name` | the accounting `Debug` already did, now callable |
| `adf::sdf_geq_everywhere_in::<L, ..>` | the redundancy proof, refining boxes in layout `L` |
| `QuadtreeDraw`, `AdfDraw` (`space-filling`) | impls now generic over `L`, so a 2-D k-d tree renders too |

### Unchanged behaviour

`Orthant` is bit-for-bit the previous implementation: same child order, same
descent, same node size (64 bytes at `DIMS = 2`), same pruning decisions. The
layout marker is zero-sized, so `size_of::<Tree<..>>()` is equal for both.

## Measurements

`cargo test -p adaptive-distance-field --release --test layout -- --ignored --nocapture`
— same insertion sequence, same resolution, `prune_subdiv = 3`; roughly 8 s of
measurement, sized so nothing is timer-bound. Both layouts are asserted to answer
**bit-identically** on every probe before any timing is reported, so the comparison
is like with like. Queries are the fastest of five sweeps: a single sweep is
memory-bandwidth bound and swung by 2× between runs, where the minimum reproduces
to within 5%.

| D | layout | children | build | query | nodes | leaves | mean bucket | memory |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | orthant | 4 | 250 ms | 39 ms | 5 345 | 4 009 | 6 | 937 KiB |
| 2 | k-d | 2 | 261 ms | 70 ms | 7 543 | 3 772 | 8 | 1.2 MiB |
| 3 | orthant | 8 | 627 ms | 73 ms | 35 849 | 31 368 | 5 | 6.5 MiB |
| 3 | k-d | 2 | **549 ms** | 159 ms | 52 095 | 26 048 | 8 | 9.1 MiB |
| 6 | orthant | 64 | 2 881 ms | 57 ms | 4 161 | 4 096 | 14 | 1.9 MiB |
| 6 | k-d | 2 | **1 267 ms** | 117 ms | 8 159 | 4 080 | 18 | 2.7 MiB |

6 000 balls at `D = 2`, 1 500 at `D = 3`, 150 at `D = 6`; 300 000 queries each.
Ratios reproduce across runs: build ×1.03, ×0.90, ×0.45. The query column is
superseded further down — "Closing the k-d query gap" takes `Kd` to ×1.67, ×1.50,
×1.93 without moving `Orthant`.

- **The build advantage appears with dimension, and only with dimension.** Parity
  at `D = 2` (×1.03 — within noise), ~10% at `D = 3`, **2.2× at `D = 6`**. This is
  the branching factor showing up where it should: nothing to win in the plane,
  everything to win at 64-way fan-out. An earlier, ten-times-smaller run of this
  benchmark showed a uniform 8–18% and a *win* at `D = 2`; it was under-loaded and
  the D = 2 figure was noise. Size the workload before believing a layout
  comparison.
- **Queries are 1.7–2.2× slower.** Descent is `D` times deeper (12 levels against
  2 at `D = 6`), and leaves hold more primitives: pruning into two half-cells
  proves less than pruning into `2^D` small ones, and the depth budget expires
  before `Kd` catches up.
- **Memory is 1.32–1.48× higher.** `Kd` holds fewer *leaves* at every dimension,
  but internal nodes keep their pre-subdivision bucket — the upper bound
  `insert_within` and `update_domain` rely on — and a binary tree of equal
  resolution has far more of them.

One hypothesis is ruled out by the numbers rather than by argument: the build
win is **not** cheaper subdivision. The proofs charged to subdivision go *up*
under `Kd` (8 158 against 4 160 at `D = 6`), because a split that halves one axis
leaves a straddling ball in both children, so the bucket does not shrink and the
node splits again — 4 079 split events against the orthant's 65. The remaining
candidate, unverified, is the insertion walk: a binary tree offers `D` times as
many chances to discard half of a shrinking cell, where a 64-way node applies the
same `D*` test to 64 large cells at once, so the orthant walk should visit more
leaves per insertion. Instrumenting `decide` would settle it.

The structural argument stands regardless of throughput: **one subdivision
allocates 2 nodes instead of `2^D`** — 1 024 per split at `D = 10`, a million at
`D = 20` — and `Kd` carries no `Branching` ceiling, so it reaches dimensions the
orthant layout cannot express at all. Choose it for reach and for build cost in
high dimension; keep `Orthant` for query-heavy work in two or three.

## Internal nodes hold a bound, not a bucket

A node that has divided kept its whole frozen `Vec<Primitive>`, but after dividing
it was only ever asked one question — an upper bound of the field over its cell,
for the insertion walk's `keep` predicate. It now stores that one number:

```rust
pub enum Bucket<Float, const D: usize> {
  Leaf(Vec<Primitive<Float, D>>),   // a leaf: the primitives, for queries and pruning
  Bound(Float),                     // divided: ĝ(c_R) + L_B·h(R), frozen
}
```

Sound for the same reason the frozen bucket was: insertions only ever *lower* the
field, so a bound taken at subdivision stays an upper bound forever. The
staleness is identical, so no pruning power is lost — and the walk now reads one
`f64` where it used to make up to four indirect closure calls per internal node.

**The root is the exception**: `sdf` answers from the root's own bucket for points
outside the domain, where there is no leaf to descend to. Losing that turns an
out-of-domain query into the empty-bucket sentinel `+MAX/2`, whereupon the ascent
climbs *out* of the cube toward it and inserts a ball of radius `MAX/4`. Every
in-domain test still passed; `sphere_packing_3d` caught it by saturating.
`out_of_domain_reads_the_seed_field` now pins it.

Measured against the layout benchmark above, tree structure bit-identical before
and after (same nodes, same leaves), so this is purely a change of representation:

| | build | query | memory |
|---|---:|---:|---:|
| `D = 2` orthant | −18% | −13% | −10% |
| `D = 2` k-d | −24% | −11% | −22% |
| `D = 3` orthant | −37% | −22% | −5% |
| `D = 3` k-d | −29% | −21% | −24% |
| `D = 6` orthant | −28% | −26% | −5% |
| `D = 6` k-d | −21% | −22% | −15% |

Faster to build *and* to query, and smaller, in every cell — the memory saving
landing where predicted, on the layout that is half internal nodes. Queries were
not expected to move at all, since leaf buckets are untouched; the likely cause is
one fewer heap allocation per internal node leaving the leaf buckets packed more
tightly, but that is a guess, not a measurement.

**Correction.** This change also invalidates a claim made above, and the reason is
worth recording: the earlier report that `Kd` "leaves hold more primitives" was an
artefact of the metric. Mean bucket was computed over *all* nodes, and `Kd` has
roughly seven times as many internal ones, each carrying a frozen snapshot.
Counting leaves only, `Kd` holds ~4.0 primitives per leaf against `Orthant`'s
~4.4 — slightly *fewer*. The benchmark column is now leaf-only.

## A split round, measured and left off

That correction removes the premise for the other change suggested above.
"Prune against the eventual child cell" is unsound as stated — a primitive may be
dropped from a child only once it is proved redundant over the *whole* child cell,
and a smaller box proves nothing about the rest. The sound way to get the same
effect is to *reach* the smaller cells: let one overflow divide through a full
round of axis cuts, `LEVELS_PER_SPLIT` levels, pruning against each level's actual
cell. `Orthant` is unaffected by construction, its round being one level.

Implemented, measured, and **off by default** — [`ADF::with_split_round`]. Against
the scalar-bound baseline it enlarges the `Kd` tree by 3–10% and costs build time,
while leaf occupancy and query time do not move:

| | nodes | build | query |
|---|---:|---:|---:|
| `D = 2` k-d | +2.8% | +8% | −8% |
| `D = 3` k-d | +10.5% | +24% | ±0 |
| `D = 6` k-d | +0.1% | −12% | ±0 |

Two things to read from that. Reaching orthant-sized cells sooner does not improve
what pruning achieves, because `Kd` was never the layout with fuller leaves. And
the `D = 6` and `D = 3` build figures point in opposite directions by more than the
run-to-run spread on this benchmark (±10% on build), so the one place it might pay
is exactly where the measurement is weakest. The knob stays so the negative result
can be reproduced rather than taken on trust.

Still worth trying: `Box<[Primitive]>` in place of `Vec` for leaves, which saves 8
bytes on every node and costs nothing, since `insert_where` already rebuilds the
whole bucket on append rather than pushing in place.

## Closing the k-d query gap

`Kd` was 1.7–2.3× slower to query. Descent was the reason, and not for the
arithmetic: a node is `2·DIMS` floats wide — 136 bytes at `DIMS = 6` — so walking
one touched up to three cache lines, and a layout needing `DIMS` levels per split
paid that `DIMS` times over.

Two changes, both in `pt_to_node`:

- **`Tree::links`**, a dense mirror of `nodes[i].children`. A descent needs
  nothing else, and four bytes per node keeps a `DIMS = 6` tree's navigation
  inside tens of kilobytes where its nodes span a megabyte.
- **[`Layout::CARRY_CELL`]**: the walk halves a local copy of the cell instead of
  reading each node's stored rect, since the root's cell is the unit hypercube and
  every child's follows from its parent's. That trades one `child_rect` per level
  against a wide load, which only pays when levels are many — so it is a per-layout
  const, `true` for `Kd` and `false` for `Orthant`.

Getting the second one wrong was instructive: carrying the cell unconditionally
took 25% off `Kd`'s query at `D = 3` and *added* 30% to `Orthant`'s, which the
ratio hid entirely — it looked like the gap had closed to ×1.18 when half of that
was `Orthant` getting worse. Query, at equal settings:

| | before | after |
|---|---:|---:|
| `D = 2` k-d | 61 ms | 59 ms |
| `D = 3` k-d | 123 ms | 90 ms |
| `D = 6` k-d | 101 ms | 86 ms |
| all orthant | — | unchanged |

Ratios: ×1.67, ×1.50, ×1.93, from ×1.72, ×2.07, ×2.28.

## Bucket capacity is a knob now

[`ADF::with_bucket_size`] — three by default, as it was hardcoded. A pure
performance knob: the field is bit-identical at every capacity, which
`tests/tuning.rs` asserts across its whole sweep, since neither pruning nor the
redundancy proof consults it. Larger buckets mean fewer, fatter leaves: a
shallower tree and less memory, paid for in primitives per query.

It matters more than the layout does. At `D = 3`, moving from 3 to 14 buys
**both** layouts more than the layout choice is worth in either direction:

| bucket | orthant query | orthant memory | k-d query | k-d memory |
|---:|---:|---:|---:|---:|
| 3 | 22.3 ms | 5.9 MiB | 34.4 ms | 6.2 MiB |
| 8 | 19.4 ms | 4.1 MiB | 25.6 ms | 2.6 MiB |
| 14 | **16.2 ms** | 2.1 MiB | **25.0 ms** | 1.0 MiB |
| 24 | 17.1 ms | 0.8 MiB | 26.6 ms | 0.5 MiB |

The default stays at 3, because there is no single right answer — at `D = 6` a
capacity of 24 costs 2.5× the build time of 3, since a fatter bucket means more
primitives per redundancy proof at max depth — and because the witness in
`insertion_domain` is tree-shape-dependent, so changing it means pinning that test.
For mixed-steepness fields the rule to reach for is capacity `⌊β / max(L_bucket,
L_prim)⌋` rather than a constant: the proof resolves margins at scale
`(L_f + L_g)·h(node)`, so a steep bucket needs proportionally deeper
branch-and-bound while dividing the node halves `h` once and for all.

## Tuned against tuned

`tests/tuning.rs` sweeps capacity and depth per layout. Since both are pure
performance knobs, the layouts need not be compared at one shared setting — each
can be tuned on its own terms, which is the only comparison that says anything
about the layouts rather than about a default:

| | orthant | k-d | ratio |
|---|---:|---:|---:|
| `D = 3` best query | 15.8 ms (b14, s5) | 24.4 ms (b14, s5) | ×1.55 |
| `D = 3` best build | 94.6 ms (b5, s3) | 93.6 ms (b5, s3) | ×0.99 |
| `D = 3` memory there | 2090 KiB | 1001 KiB | **×0.48** |
| `D = 6` best query | 12.7 ms (b3, s2) | 27.2 ms (b8, s2) | ×2.14 |
| `D = 6` best build | 52.3 ms (b8, s1) | 27.8 ms (b8, s1) | **×0.53** |

**Query parity is not reachable, and the reason is structural.** A leaf count `N`
costs `log2 N` levels of descent against `log_{2^D} N`, so `Kd` pays `D` times the
dependent loads for the same resolution; buying levels back by fattening buckets
costs primitives per query at exactly the rate it saves levels, and the sweep shows
the trade bottoming out at ×1.55 and ×2.14. What `Kd` gives in exchange is half the
memory at its best-query setting in `D = 3`, twice the build speed at `D = 6`, and
dimensions `Orthant` cannot be instantiated in at all. It is a Pareto choice, not a
deficiency to be fixed.

## Two more negative results

**Refusing a cut that prunes nothing.** The obvious way to spare `Kd` its wasted
levels, and a local-greedy trap: the first cut of the domain prunes nothing
whatever, since a ball straddling it survives on both sides, so the tree never
reaches the depth at which cuts *do* bite. At `D = 6` it collapsed to a single leaf
holding all 60 primitives — queries ×3.2, the arena down to 1.6 KiB. Scoped to a
whole round rather than one cut it recovers (39 → 30 ms) and still does not beat
having no heuristic (27 ms). Reverted; the comment at the call site records it.

A first attempt required *every* child to shrink, which additionally broke
`insertion_domain` by refusing the ordinary orthant split where one child
legitimately keeps the whole cluster. Two distinct errors in one idea.

**Depth budget.** Shallower is not cheaper: at `D = 6`, one full subdivision
instead of two takes the build from 52 ms to 28 ms but the query from 12.7 ms to
35 ms, because 64 leaves hold 52 primitives each. The optimum sits where descent
and occupancy balance, and for both layouts here that is the deepest setting
measured.

## Twenty dimensions, thirty seconds each

`tests/stress_kd.rs` (ignored; ~13 min) runs the full pipeline — ascend, place a
ball of *diameter* half the clearance, repeat — in every dimension from 1 to 20,
with nothing capped but wall clock and a 256 MiB arena ceiling. `Orthant` cannot be
instantiated past 6, so this is k-d only. Every attempt in every dimension placed a
ball; the field stayed sound throughout.

|  D | circles | ins/s | depth | cap |     nodes |  leaves | occ | memory | d first | d last |   stop |
|---:|--------:|------:|------:|----:|----------:|--------:|----:|-------:|--------:|-------:|-------:|
|  1 |  64 604 | 2 153 |     6 |   6 |        91 |      46 |1815 |  1.9 M |  0.5000 | 0.0000 |   time |
|  2 | 198 852 | 6 628 |    12 |  12 |     7 705 |   3 853 | 173 | 15.7 M |  0.5000 | 0.0010 |   time |
|  3 |  96 588 | 3 220 |    18 |  18 |   513 501 | 256 751 |15.6 |  131 M |  0.5000 | 0.0097 |   time |
|  4 |     832 |   200 |    24 |  24 | 2 018 037 |1 009 019| 3.3 |  261 M |  0.5000 | 0.0748 | memory |
|  5 |     192 |  14.1 |    30 |  30 | 3 233 675 |1 616 838| 4.1 |  499 M |  0.5000 | 0.1470 | memory |
|  6 |      64 |   5.4 |    31 |  36 | 2 278 715 |1 139 358| 5.5 |  423 M |  0.5000 | 0.2367 | memory |
|  7 |      36 |   1.2 |    29 |  42 | 1 135 979 | 567 990 | 7.4 |  252 M |  0.5000 | 0.2885 |   time |
|  8 |      31 |   0.9 |    29 |  48 | 1 412 469 | 706 235 | 8.6 |  355 M |  0.5000 | 0.3160 |   time |
|  9 |      20 |   0.5 |    18 |  54 |   251 941 | 125 971 | 8.8 | 67.8 M |  0.5000 | 0.2950 |   time |
| 10 |      19 |   0.6 |    17 |  60 |   223 997 | 111 999 |15.0 | 79.4 M |  0.5000 | 0.3311 |   time |
| 12 |      19 |   0.3 |    17 |  72 |   256 181 | 128 091 |19.6 |  112 M |  0.5000 | 0.2776 |   time |
| 16 |      19 |   0.6 |    17 |  96 |   262 143 | 131 072 |20.0 |  132 M |  0.5000 | 0.3333 |   time |
| 20 |      20 |   0.5 |    18 | 120 |   524 287 | 262 144 |21.0 |  302 M |  0.4999 | 0.3503 |   time |

(11, 13–15, 17–19 interpolate; `occ` is mean primitives per *leaf*, internal nodes
holding a scalar.)

**The proof budget is dimensional.** `prune_subdiv` had to drop from its default 8
to 1 for the test to run at all. `sdf_geq_everywhere` clears a box when the centre
margin covers `(L_f + L_g)·h(R)`, and `h` is the *half diagonal* — `√D/2` on the
unit cube, so `√D` times the half-side. Every cell is `√D` harder to certify than
in 1-D, the early exit fires that much less often, and the branch-and-bound
descends toward its budget instead of returning. Measured at the default: a
**single insertion** took 6.1 s at `D = 8` and 4.4 s at `D = 10`, against 3 ms at
`D = 20` with a budget of 1. Since k-d spends `subdiv × D` levels per proof, 8 is a
two-dimensional default and nothing more.

**The tree doubles on every insertion.** From `D = 13` up the node counts are
131071, 262143, 524287 — exactly `2^k − 1`, *complete* binary trees, with mean leaf
occupancy 20 out of 21 primitives present. No cut ever pruned anything: a ball
straddling a one-axis cut survives in both children, both children inherit the full
bucket, both overflow again next time. Twenty insertions therefore buy 2^18 leaves
holding 5.5 M copies of the same twenty-item list, and 300 MiB. The binding
constraint is no longer depth — the cap of 120 levels is never approached, the tree
stops at 18 — but the doubling.

This re-frames the negative result above. **Refusing a cut that prunes nothing is
harmful at `D ≤ 6` and would be the only thing standing between the arena and
`2^120` leaves at `D = 20`**; the correct policy is dimension-dependent, and the
crossover sits near where `Orthant` stops being constructible.

**Clearance decays as `k^{−1/D}`, within 12%.** The last column against
`d_1·k^{−1/D}` with `d_1 = 0.5`: `D = 2`, 198 852 balls, 0.0010 measured against
0.00112 predicted; `D = 3`, 0.0097 against 0.0109; `D = 6`, 64 balls, 0.2367 against
0.2500; `D = 8`, 0.3160 against 0.3255. The mid dimensions run 19–24% fast
(`D = 4`, 0.0748 against 0.0931), the first balls near the centre biting harder than
a saturated packing's asymptote. The exponent is the one the publication's
diagnostic predicts, so the wall at high `D` is not the tree's — 20 balls in 30 s
still leave 70% of the original clearance because that is all `k^{−1/20}` permits,
and no amount of tree engineering changes the exponent.

**Two harness artefacts, not library behaviour.** `D = 1` is slower than `D = 2`
(2153 against 6628 ins/s) because 6 subdivisions give a 1-D tree only 64 cells, so
each of 46 leaves carries 1815 primitives and every query pays for them; low
dimensions want far more depth than the same constant provides. And the three
`memory` stops overshoot the 256 MiB ceiling to 499 MiB, because the check runs
every 64 insertions and one insertion can double the arena.

## The split policy is dimensional

### Added

| API | Meaning |
|---|---|
| `ADF::with_cut_must_prune(bool)` | An overflowing leaf divides only if the division prunes something. |
| `adf::CUT_MUST_PRUNE_MIN_DIMS` | `4` — the dimension from which that is the default. |

`divide_round` now also returns the smallest bucket any cell of the round reached,
which costs nothing: every `kept` was already in hand. The decision compares it
against the bucket the round started from. A refused leaf sits over capacity, so it
retries only once its bucket has doubled — `O(log n)` trials rather than one `O(n^2)`
trial per insertion, with the bucket's own length as the clock and no per-node state.

Not a correctness knob: pruning only ever drops primitives it has *proved* redundant
over the cell, so the field is bit-identical either way. `cut_must_prune_leaves_the_field_alone`
pins that over 2000 probes at `D = 12` while the arena shrinks by more than 8×.

### Measurements

`tests/stress_kd.rs` now runs both arms per dimension, same seed, ten seconds each.

|  D | divide | bite | circles | divide mem | bite mem |
|---:|-------:|-----:|--------:|-----------:|---------:|
|  1 | 37 674 | 37 556 | ×1.00 | 1.1 MiB | 1.1 MiB |
|  2 | 101 409 | 101 537 | ×1.00 | 8.7 MiB | 8.6 MiB |
|  3 | **37 246** | 3 821 | **×0.10** | 92.7 MiB | 89.7 KiB |
|  4 | 832 | 3 273 | ×3.93 | 261.5 MiB | 76.9 KiB |
|  5 | 169 | 1 419 | ×8.40 | 384.6 MiB | 33.5 KiB |
|  6 | 64 | 2 230 | ×34.8 | 422.9 MiB | 52.5 KiB |
|  8 | 25 | 1 717 | ×68.7 | 99.8 MiB | 40.5 KiB |
| 10 | 18 | 1 351 | ×75.1 | 45.3 MiB | 32.0 KiB |
| 12 | 18 | 1 094 | ×60.8 | 56.4 MiB | 26.0 KiB |
| 16 | 19 | 799 | ×42.1 | 132.0 MiB | 19.1 KiB |
| 20 | 19 | 549 | ×28.9 | 148.0 MiB | 13.3 KiB |

Every memory-ceiling stop disappears. The `k^{−1/D}` wall does not: 549 balls at
`D = 20` leave the clearance at 0.347 where 19 left it at 0.332, which is the
exponent doing exactly what it did before — the policy buys throughput and memory,
never the packing.

**The threshold is 4, and the first guess of 10 was wrong.** It was picked from
per-leaf retention — 8.5% of the bucket at `D = 6` rising to ~100% by 13 — on the
reasoning that a tree still pruning 91% of its buckets must be earning its keep. It
is not: at `D = 6` that tree costs 422.9 MiB and 64 circles against 52.5 KiB and
2230. Retention measures whether cuts prune *at all*, not whether they prune enough
to pay for the nodes they create. The covering overhead `(√D/2)^D = (D/4)^(D/2)`
does measure that, and it passes 1 at exactly `D = 4` — 0.65 at 3, 1 at 4, 3.4 at 6,
9.8e6 at 20. Predicted crossover and measured crossover are the same integer.

**`D = 3` is the one loss, at ×0.10, and it is the local-greedy trap.** The first cut
of the domain prunes nothing whatever, so refusing it forecloses the depth at which
cuts would have bitten, and the field collapses to a single leaf holding 3822
primitives. Above the crossover there is no such depth to foreclose — every cell
keeps the whole bucket however deep it goes — which is why the same trap is harmless
at 4 and above. This is the same heuristic listed as a negative result earlier; what
was missing was not a better rule but the dimension it applies in.

**The policy is self-selecting by layout.** At `D = 6` it never fires under
`Orthant`: the benchmark's tree is 4161 nodes either way, because an all-axes cut
divides the cell volume by `2^D` and does prune. It is k-d's one-axis cuts that prune
nothing, so the constant is really a guard on a k-d trap that happens to be indexed
by dimension.

**Query-dominated workloads in `4..10` may still want it off.** At `D = 6` over 150
balls with 300 000 queries the policy is build ×5.0 faster (1.004 s → 199 ms), query
×2.96 slower (81.5 ms → 241 ms) and memory ×620 smaller (2.3 MiB → 3.7 KiB). Net
positive here, and negative past roughly 1.5 M queries, since a fat leaf trades an
`O(1)` descent for an `O(n)` bucket scan. Hence a setter and not only a constant.

## Weight-ordered axes

Roadmap step 2 of `doc/publications/infinite_dimensions`: subdivide in descending
weight, so cost follows the *effective* dimension rather than the ambient one.

### Added

| API | Meaning |
|---|---|
| `Tree::new_in`, `ADF::new_in` | a field over an arbitrary domain box, not only the unit cube |
| `sdf::boundary_box(domain)` | the walls of that box; `boundary_rect` is this at `Aabb::unit()`, bit for bit |
| `tree::CutPolicy<DIMS>` | which axis a binary cut halves, given the cell and its depth |
| `tree::Cyclic`, `tree::Widest` | round-robin `depth % DIMS`, and the widest axis |
| `tree::KdBy<P>` | the binary layout, parameterised by the policy |
| `tree::WeightedKd` | `KdBy<Widest>` |

`Kd` is now `KdBy<Cyclic>` — a type alias, so every existing `ADF<f64, 6, Kd>`
still compiles and behaves identically.

Weights are the domain box's extents. A root of extent `γ` makes axis `i` matter
in proportion to `γᵢ`, which is the ellipsoid `Ωₐ` of §2.3 in axis-aligned
clothing, and needs no state threaded through `Layout`: `child_rect` and
`child_index` already receive the cell.

`CutPolicy::axis` takes the *cell*, not its extent. Passing `rect.size()` cost
`Cyclic` a vector subtraction per descent step that it then discarded — worth
×1.6 on the `D = 2` k-d build before it was removed.

### The mechanism

The certificate clears a cell when the margin covers `(L_f + L_g)·h(R)`, and
`h = ½√(Σ sᵢ²)`, so halving side `sᵢ` buys a reduction proportional to `sᵢ²`.
Cutting the widest axis is the greedy move against the only quantity the proof
reads. On a cube it *is* round-robin — sides start equal, so the argmax walks
`0, 1, .., D-1` and back — and `widest_reduces_to_cyclic_on_a_cube` pins the
schedule level for level, plus the trees node for node while the proof is barred
from refining.

Barred, because with `prune_subdiv > 0` the trees legitimately diverge even on a
cube: `sdf_geq_everywhere_in` subdivides through the same `Layout`, and the box it
starts from is the node's cell, which is *not* a cube at any depth that is not a
multiple of `D`. `Widest` cuts those along their longest axis too. That is an
unlooked-for second helping of the same idea — the roadmap only claimed the
domain — and it is why the `s = 0` control below is not exactly ×1.00.

### Measurements

`tests/weights.rs`, `D = 6`, 500 insertions, `γᵢ = (i+1)^(−s)`. Ratios are
weight-ordered ÷ round-robin, so below 1.0 the policy wins.

| s | build | query | nodes | memory |
|---:|---:|---:|---:|---:|
| 0.0 (cube) | ×0.99 | ×1.33 | ×0.97 | ×0.98 |
| 0.5 | ×0.90 | ×0.93 | ×1.00 | ×0.98 |
| 1.0 | ×0.61 | ×0.97 | ×0.93 | ×0.81 |
| **2.0** | **×0.04** | ×0.63 | ×0.18 | **×0.09** |

At `s = 2` — the paper's `log N(10⁻²) ≈ 10` row — that is 30.4 s and 122 MiB
against 1.2 s and 11.3 MiB, for the same 500 balls and a bit-identical field. The
gain is monotone in the decay rate, which is the claim: cost tracks the number of
axes that matter. Round-robin gets *worse* from `s = 1` to `s = 2` (9.1 s → 30.4 s)
because it keeps spending `D-1` of every `D` cuts on axes already too short to
matter, while weight-ordered cuts get monotonically better (5.5 s → 1.2 s).

**It also makes cuts bite.** The prediction was that weight-ordered cuts would
survive `cut_must_prune`, which at `D = 6` refuses a division that prunes nothing.
Nodes surviving with the policy on: at `s = 1`, 14 671 round-robin against 59 197
weight-ordered; at `s = 2`, **27 against 7 487**. Round-robin collapses to
essentially nothing; weight-ordered keeps a real tree.

### Fixed

**`pt_to_node` hardcoded the unit cube**, in the containment test and as the
carry-cell descent's starting cell, so every domain that was not `[0,1]^D`
silently resolved to the wrong leaf: 44 627 of 200 000 probes wrong, by up to 0.1,
under *both* layouts. Comparing two trees against each other would never have
caught it — they were both wrong — so `an_anisotropic_domain_reads_exactly` checks
against brute-force `min` over the primitives instead, across three aspect ratios
and all three layouts.

`BoundingBox for ADF` likewise returned `Aabb::unit()` regardless of the tree, and
now returns the root's cell.

## Anisotropic bodies, compact manifolds, and a hundred dimensions

Roadmap step 3. The engineering all works and the ceiling is genuinely raised;
the headline claim — that per-axis radii beat a ball at high `N` — **does not
survive measurement**, and the reason is worth more than the claim was.

### Added

| API | Meaning |
|---|---|
| `adf::Reach` | what an insertion can reach: `Ball { centre, radius }` or `Box(Aabb)` |
| `ADF::insert_within_reach` | the insertion walk against a `Reach`; `insert_within` is now the `Ball` case |
| `adf::Manifold<Float, D>` | a compact manifold as per-axis weights `γ` |
| `Manifold::{sobolev, exponential, finite_rank, from_extents}` | the standard families |
| `Manifold::{domain, walls, aspect, effective_dimension, log_volume, field}` | and what they derive |
| `ADF::box_is_free` | certify a box free — the redundancy proof with zero on the right |
| `ADF::grow_box` | the largest aspect-locked box the field will certify at a point |
| `ADF::with_levels` | depth budget in tree levels rather than full subdivisions |
| `ADF::with_prune_levels` | proof budget likewise |
| `adf::sdf_geq_everywhere_levels` | the proof, budgeted in levels |
| `Layout::halve_into` | halve a cell in place; defaulted, overridden by `KdBy` |

`Tree::max_depth` and `Node::depth` widen from `u8` to `u16`, and
`Layout::child_rect` takes `&Aabb` rather than an `Aabb` by value.

### Levels, not subdivisions

Every budget in the crate was denominated in *full subdivisions*, which a binary
layout spends `D` levels on. That is unusable at `D = 100`: the mildest setting of
`with_prune_subdiv` is a hundred levels, and an undecided proof then walks a binary
tree a hundred deep. Budgets are now stored in levels, with the subdivision
setters kept as exact-behaviour wrappers (`subdiv × LEVELS_PER_SPLIT`), so nothing
below `D ≈ 12` moves at all.

The same applies to the arena: `max_depth × D ≤ 255` allowed **two** subdivisions
at `D = 100`. `u16` and `with_levels` remove it. Per-node cost is the other half —
`child_rect` copied a 1.6 KiB `Aabb` per call at `D = 100`, and `halve_into` writes
one scalar instead.

Result: `a_hundred_dimensions` builds a weighted field at `D = 100`, grows and
places bodies, and reads back exact against brute force. 150–880 µs per grow,
30–300 µs per insertion.

### Effective dimension is the number that matters

`Manifold::effective_dimension` is the participation ratio `(Σγ)²/Σγ²`: `D` on a
cube, `k` on a rank-`k` manifold, and for Sobolev decay a constant that does not
move with `D` — **2.38 at `D = 24` and 2.47 at `D = 100`**, both `s = 2`. That is
the whole compact-manifold claim in one number, and it is now computable rather
than inferred from timings.

### The negative result: a box cannot be certified

`tests/high_d.rs` scores claimed volume in logs, because nothing here is
representable otherwise — a Sobolev domain's volume at `D = 100` is `10^(−316)`.
Ratios of claimed to domain volume, and the gap between the two bodies:

|  D |  s | eff dim | ln ball÷dom | ln box÷dom | **ln gap** |
|---:|---:|---:|---:|---:|---:|
| 24 | 1.0 | 8.89 | −69.8 | −64.2 | **+5.6** |
| 24 | 2.0 | 2.38 | −88.9 | −99.1 | −10.1 |
| 48 | 1.0 | 12.24 | −166.3 | −172.5 | −6.2 |
| 48 | 2.0 | 2.44 | −200.3 | −266.3 | −66.0 |
| 100 | 1.0 | 16.46 | −392.7 | −442.5 | −49.8 |
| 100 | 2.0 | 2.47 | −483.2 | −725.3 | **−242.1** |

The box wins in exactly one row — the mildest anisotropy at the lowest dimension —
and loses by a widening margin everywhere else. **The geometry is not what binds;
the certificate is.** A box of aspect `γ` is free out to the walls, but proving it
free needs the Lipschitz test, and that test compares the cell's *half-diagonal*
against the field's clearance. The half-diagonal is set by the **longest** axis and
the clearance by the **shortest**, so an aspect ratio of `10^4` at `D = 100, s = 2`
is a factor of `10^4` the proof has to close by refinement alone.

It closes it, slowly, and pays exponentially. At `D = 24, s = 2`, the gap against
the levels `box_is_free` may refine:

| levels | 0 | 2 | 4 | 8 | 12 |
|---|---:|---:|---:|---:|---:|
| ln gap | −72.4 | −48.3 | −35.8 | −21.4 | −10.1 |

Roughly halving per four levels while the cost doubles per level. Twenty levels at
`D = 100` did not finish inside five minutes.

So step 3 as stated is not pure engineering after all. Per-axis radii need a
certificate that is anisotropic too — the natural one being the Lipschitz bound in
the *weighted* metric, where the manifold is a cube again and the half-diagonal
stops being dominated by an axis the clearance knows nothing about. That is a
change to the proof, which §2.3 explicitly promised would not be needed, and it is
the honest prerequisite for this step rather than an optimisation of it.

Everything else stands: `Reach::Box` is sound and strictly tighter than the ball
that contains it (`a_box_reaches_less_far_than_its_ball` pins both), `grow_box`
returns only certified-free boxes, and on the one workload where the certificate
can keep up it does win.

## Inclusion functions, and the negative result reversed

The previous entry concluded that per-axis radii lose to balls, and that the
Lipschitz certificate was why. The second half was right and the first was a
consequence of it: give the primitives a way to answer for themselves and the
result flips.

### Added

| API | Meaning |
|---|---|
| `Primitive::lower` | an inclusion function: a lower bound of the field over a whole box |
| `Primitive::with_lower` | supply one directly |
| `Primitive::centred` | derive an exact one for a body whose field grows with distance from its centre |
| `Primitive::enclosing` | the same for a container whose field falls away from it — walls |
| `ADF::lower_bound_over` | the best bound available over a box, per primitive |

`Manifold::field` seeds with `Primitive::enclosing`, so a weighted domain's walls
carry theirs from the start.

### One observation does all the work

For a body whose field is **monotone in the componentwise distance from a
centre** — balls, boxes, ellipsoids, crosses, anything written from `|p − c|` per
axis — the exact minimum over a box is attained at the box's point nearest that
centre. Every component of `|x − c|` is minimised there simultaneously, and a
monotone function of those components is minimised with them. So the inclusion
function is one field evaluation at a clamped point: `O(D)`, exact, no
refinement. `enclosing` is the mirror image, taking the *farthest* corner, and
using the wrong one of the two is unsound rather than merely loose.

### Measurements

The same benchmark, the same workload, the only change being that bodies and
walls now carry inclusion functions:

|  D |  s | ln gap before | ln gap after |
|---:|---:|---:|---:|
| 24 | 1.0 | +5.6 | **+50.4** |
| 24 | 2.0 | −10.1 | **+69.6** |
| 48 | 1.0 | −6.2 | **+118.7** |
| 48 | 2.0 | −66.0 | **+109.0** |
| 100 | 1.0 | −49.8 | **+282.4** |
| 100 | 2.0 | −242.1 | **+122.5** |

Every row flips, and the whole run goes from **94.4 s to 0.6 s**. The refinement
sweep that used to show the shortfall closing four levels at a time is now flat —
`ln gap` is 69.6 at 0 levels and 69.6 at 12 — because the branch-and-bound is
never reached at all. That flat column is the clearest statement of the finding:
the geometry was never the obstacle.

So roadmap step 3 does hold, with inclusion functions as its prerequisite rather
than a weighted metric. §8 of the publication carries the amendment.

### Fixed

`lower_bound_over` first shipped with its traversal predicate inverted.
`Tree::visit_leaves` takes a **prune** predicate — `true` skips the subtree — and
reading it as "keep" scanned exactly the leaves that say nothing about the box,
leaving the bound at `Float::MAX` and certifying everything free. It was visible
in the benchmark rather than in a test: single bodies claiming `e^46` times the
volume of the domain containing them. Bodies that cannot fit in their own domain
are the kind of impossible number worth reading a table for.

## What the generalization cost the plane

Three questions worth answering with numbers rather than reassurance, after a
crate that began two-dimensional grew a hundred-dimensional path through it.

### Working in low dimension is unchanged

No example moved. `examples/gd_adf/02` still reads

```rust
representation.write().unwrap().insert_at_maximum(local_max, Primitive::from_shape(circle))
```

and `Primitive::new`, `from_shape`, `insert_at_maximum`, `insert_within` are as
they were. Everything added for high dimension — `Reach`, `Manifold`, `grow_box`,
`with_levels`, `with_prune_levels`, inclusion functions — is additive and optional,
and none of it appears in any example. The only friction anywhere in the arc is
naming the layout in the turbofish, which is deliberate.

### `u8 → u16` on the depth costs nothing

`size_of::<Node>()` is **64 bytes at `D = 2` and 128 at `D = 6`, on both the
pre-branch commit and this one**. The extra byte fell into padding that was
already there. Measured, not assumed — it was the obvious thing to suspect.

### `Primitive` grew, and that is the real cost

24 bytes to **40**, because `Option<Arc<dyn Fn>>` is a fat pointer. Benchmarked
back to back against the pre-branch commit, serially, same machine:

| | before | after | |
|---|---:|---:|---:|
| `D = 2` orthant | 841.5 KiB | 1.2 MiB | ×1.46 |
| `D = 3` orthant | 6.2 MiB | 8.4 MiB | ×1.35 |
| `D = 6` orthant | 1.8 MiB | 2.7 MiB | ×1.50 |

**Every user pays it, whether or not they ever supply an inclusion function.**

Timings from the same runs are not worth quoting: the ratios scatter ×0.83 to
×1.52 in both directions — k-d queries got *faster*, orthant `D = 3` build swung
400–896 ms across runs on the same commit — so the machine's noise band is wider
than any effect. Memory is the only clean signal, and it has an exact mechanical
explanation, which is why it is stated and the timings are not.

**Future work.** Fold the inclusion function into the same trait object as the
field and `Primitive` returns to 24 bytes with the bound free:

```rust
trait Field<F, const D: usize>: Send + Sync {
  fn eval(&self, p: Point<F, D>) -> F;
  fn lower(&self, _rect: &Aabb<F, D>) -> Option<F> { None }
}
pub struct Primitive<F, const D: usize> { f: Arc<dyn Field<F, D>>, lipschitz: F }
```

A blanket impl over `Fn(Point) -> F` keeps `Primitive::new(closure)` compiling
verbatim. Not done here.

### The Lipschitz-only contract still holds

GD-ADF asked nothing of a primitive but its Lipschitz bound, and still asks
nothing more. `lower` is an `Option`, `Primitive::new` leaves it `None`, and every
`None` falls back to `f(c) − L·h(R)` exactly as before. Nothing in the library
*requires* more: `grow_box` and `box_is_free` work without inclusion functions,
merely poorly above `D ≈ 24`. Every example and every test but three exercises the
Lipschitz-only path.

What the escape hatch costs, besides the 16 bytes, is a sharper failure mode.
An over-large Lipschitz constant is safe — it only makes pruning lazier. A `lower`
that is not really a lower bound is **unsound**: it certifies occupied space as
free. `centred` and `enclosing` derive exact ones and should be preferred to
writing one by hand.
