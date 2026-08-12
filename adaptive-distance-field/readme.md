Adaptively sampled distance fields in ℝᴺ, with Lipschitz-certified pruning.

An `ADF` stores a signed distance field as a `2^N`-tree whose leaves each hold a handful of `Primitive`s — a field
closure paired with an honest upper bound on its gradient. The tree represents their pointwise `min`, so a field
composed of millions of primitives is sampled in logarithmic time by descending to the leaf covering the query
point, instead of evaluating every primitive.

What makes that sound rather than merely fast is the Lipschitz bound. Every decision the structure makes — whether
a new primitive can be skipped, whether a stored one has become redundant, which subtrees an insertion can possibly
change — is settled by branch-and-bound over

```
f - g  ≥  (f - g)(c) - (L_f + L_g)·h
```

which either *proves* a region clean or descends to a witness. A primitive is never dropped on a heuristic, only on
a proof, so the stored field never deviates from the `min` over everything inserted.

```rust
use adaptive_distance_field::{
  adf::{self, Primitive},
  geometry::{Point, VectorExt},
  line_search::LineSearch,
  sdf::{self, SDF},
};

// seed a 3D field with the walls of the unit cube, positive inside. The scalar,
// the dimension count and the subdivision layout are all named — the builder takes
// them one at a time, `ADF::<f64, 3, Orthant>::new(6, ..)` all at once
let mut field = adf::builder().f64().dims::<3>().orthant().bounded(6);

// climb to a local maximum of the free space — the deepest point around
let start = Point::from([0.3, 0.6, 0.5]);
let peak = LineSearch::default().optimize(|p| field.sdf(p), start);
let clearance = field.sdf(peak);

// drop a ball there, and let the tree work out which cells that can affect
let ball = move |p: Point<f64, 3>| (p - peak).length() - clearance / 2.0;
field.insert_at_maximum(
  adaptive_distance_field::geometry::DistPoint { point: peak, distance: clearance },
  Primitive::new(ball),
);
```

## Layout

- [`geometry`](src/geometry.rs) — the vocabulary: `Point` and `Vector` (both [`nalgebra`](https://docs.rs/nalgebra/)'s,
  re-exported so downstream code can name the same versions), the axis-aligned `Aabb`, `BoundingBox`, and the
  `Combinator` trait — every way to transform or combine one field into another. Each combinator preserves or
  `max`-combines its operands' constants, so a composed chain still reports an honest bound.
- [`sdf`](src/sdf.rs) — the `SDF` and `Lipschitz` traits everything is written against, the field types the
  combinators return, and `sdf_geq_everywhere`: the branch-and-bound proof that underpins all of it.
  `boundary_rect` seeds a field with the walls of the unit cube.
- [`adf`](src/adf/mod.rs) — the structure itself, over the [`tree`](src/adf/tree.rs) arena, in either the
  `2^N`-way `Orthant` layout or the binary `Kd` one; [`builder`](src/adf/builder.rs) names the three type
  parameters one at a time.
- [`line_search`](src/line_search.rs) — adaptive gradient ascent, for locating the maxima that `insert_at_maximum`
  consumes.

Dimension count and layout are both compile-time constants throughout, so `ADF<f64, 2, Orthant>` and
`ADF<f64, 3, Kd>` monomorphize separately and the 2D orthant case costs exactly what a quadtree-only
implementation would. See [CHANGELOG.md](CHANGELOG.md) for what each layout costs, measured.

## Beyond this crate

The [`space-filling`](https://docs.rs/space-filling/) crate builds on this one: a catalogue of shape primitives in
2, 3, 4 and N dimensions, a discrete bitmap solver for exact global maxima, batched parallel maxima search, and 2D
rasterization. Its [readme](../readme.md#implementation) states the ADF algebra in full.

## Licence

GPL-3.0 — see [LICENCE](../LICENCE).
