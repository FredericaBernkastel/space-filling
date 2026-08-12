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
Ratios reproduce across runs: build ×1.03, ×0.90, ×0.45.

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

Worth trying next, in the order they are likely to pay: prune against the
*eventual* child cell rather than the immediate half-cell, so a `Kd` split prunes
as sharply as an orthant one; and drop the bucket from internal nodes in favour of
a scalar bound, which would take most of the memory difference with it.
