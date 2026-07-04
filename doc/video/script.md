<!--
Conventions:
  body text        narration, read aloud
  {visual: …}      manim scene direction
  {formula: …}     equation rendered on screen (LaTeX)
  {read: "…"}      verbatim narration of the preceding formula (spoken)
Target audience: graduate level. Terse; no general-audience detours.
-->

### Scene 1
Liserotte here. The story begins with a following equation:   
{formula: \vec{x}^{*} \;=\; \arg\max_{\vec v \in \Omega}\ \min_{n}\ \mathrm{sdf}_{n}(\vec v)}
{read: "x-star equals the arg-max, over the domain omega, of the pointwise minimum over all primitives n of sdf-n of v."}. Solving it numerically is not a problem. Solving it in logarithmic time and linear memory is another task entirely. This brought me into some fascinating mathematics.

{visual: ../../readme.md, "Implementation" section, scrolling}
This video introduces my paper alongside its reference implementation, which I've worked on for the last 6 years — mostly due to lack of theoretical knowledge on my part.

But I want to begin from even earlier.
{visual: https://paulbourke.net/fractals/randomtile, scrolling}
In 2011, Paul Bourke published his ["Random space filling of the plane"](https://paulbourke.net/fractals/randomtile) in relation to statistical geometry. The work described an iterative algorithm for tiling E^n space with non-overlapping shapes. The visuals were stunning, but the algorithm itself was far from the best.

{visual: assets/bourke_zoom.png, zooming out}
This is "[A Million-Circle Fractal](https://www.d.umn.edu/~ddunham/circlepat.html)", generation run time was stated as "14.7 hours". Neither did the algorithm allow for much control over the resulting distribution. Can we do better?

### Scene 2
{visual: densely filled space with shapes. random locations are tried one by one at least 10 times, all rejected due to being inside some shape, marked in red. until one is found to be outside, marked in green}
The major issue is purely in random sampling of the space. As more and more shapes are added, there is a vanishingly small chance of finding an empty spot. In the limit, it takes an unbounded amount of iterations to insert one more shape.

Let's consider a different approach: represent each shape with a signed distance function.
{visual: all distance functions of shapes from `shapes.rs` visualized in a grid, alongside their field notations}
For a set S in the plane, its signed distance function returns the distance to the boundary, negative inside and positive outside:
{formula: \mathrm{sdf}_{S}(\vec v)\;=\;\bigl(\mathbb{1}_{\vec v \notin S} - \mathbb{1}_{\vec v \in S}\bigr)\ \inf_{\vec u \in \partial S}\lVert \vec v - \vec u\rVert}
{read: "sdf-S of v equals plus-or-minus — plus outside S, minus inside — the infimum over boundary points u of the distance from v to u."}
Two properties matter. Its gradient has unit magnitude almost everywhere and points directly away from the surface; and it is one-Lipschitz — the value cannot change faster than you move.

{visual: combining all the shapes from the grid into one compound distance field}
Next, combine all primitive SDFs into one compound distance field, using the aggregate pointwise minimum of them all.
{formula: g(\vec v)\;=\;\min_{n}\ \mathrm{sdf}_{n}(\vec v)}
{read: "g of v is the minimum, over all primitives n, of sdf-n of v."}
The minimum of one-Lipschitz functions is again one-Lipschitz, so `g` inherits the same regularity.

{visual: a single gradient ascent trajectory across a real SDF field}
Since by definition the gradient of any SDF points towards the outside, we can build an iterative algorithm to travel along that gradient, naturally finding some local maximum outside of any shape. The name of that algorithm is "gradient ascent".
Problem solved, right? Well, not quite. Computing the aggregate pointwise minimum of N SDFs takes O(N) time. Filling the space with N shapes takes O(N^2) time. We've traded one limitation for another with similar time semantics.

### Scene 3
{visual: an N×N grid whose cells are threaded by a Z-order (Morton) curve; the compound field is rasterized onto it as a heatmap; a single linear scan sweeps the cells, tracking the running maximum, which lands on the brightest cell}
Consider a bitmap of size N×N. The SDF field is rasterized onto it, and each (x, y) pixel holds a single-precision value at this point. Its memory layout follows a Z-order curve, so that spatially close samples stay close in memory. Scan every pixel one by one to find the global maximum of the field. Let's see whether it works.
{visual: ../../examples/argmax2d/01_fractal_distribution.rs, iteratively add one more shape every frame}
Sampling now takes O(1) time theoretically, but increasing precision costs both quadratic time *and* quadratic memory. A 4096×4096 bitmap in this visual corresponds to an average discretization error of 2.44e-4 and consumes 64MB of memory. This is a dead end.

### Scene 4
{visual: assets/Adaptively sampled distance fields.pdf, scrolling}
A paper "*Adaptively Sampled Distance Fields* (2000)" (doi:10.1145/344779.344899) proposes reducing memory by approximating the field per node of a k-d tree, yet gives little guidance for a practical implementation beyond fitting a polynomial in each node. Precision would then be limited by chosen polynomial degree as well as the quality of the regression algorithm.
Current work takes a different route: each node (bucket) stores the primitives themselves.

{visual: a quadtree subdividing over the plane; each leaf highlights only the few primitives whose fields reach into it; a query point descends root-to-leaf, then min-reduces over that leaf's short bucket}
The purpose of the buckets is spatial locality of relevance. Far from a shape, that shape is never the minimum, so it need not be stored there. A leaf keeps only the handful of primitives that can define the field inside it. To sample `g` at a point, descend to the leaf containing it and minimise over that leaf's bucket alone:
{formula: g(\vec v)\;=\;\min_{f \in B(\ell(\vec v))} f(\vec v),\qquad \text{cost }\;O(\mathrm{depth} + \beta)}
{read: "g of v is the minimum over the primitives f in the bucket of the leaf containing v; the cost is order depth plus beta, the bucket size."}
For a balanced tree the depth is logarithmic, and the bucket size `β` is bounded by a constant — so a query is O(log N), never touching the millions of distant primitives.

The soundness of the whole scheme rests on Lipschitz continuity. A function `f` is `L`-Lipschitz when
{formula: \lvert f(\vec a) - f(\vec b)\rvert \;\le\; L\,\lVert \vec a - \vec b\rVert \qquad \forall\, \vec a, \vec b}
{read: "the absolute value of f of a minus f of b is at most L times the distance from a to b, for all a and b."}
Geometrically the graph is trapped in a cone of slope `L`: one sample plus the constant bounds the function over an entire neighbourhood. True SDFs are one-Lipschitz; estimators declare a larger `L`.

{visual: a single contiguous array of nodes; each internal node points to the index of the first of its four children, drawn as an arrow into the packed block; sibling cells adjacent in memory}
Implementation-wise, the quadtree is a flat arena: all nodes live in one contiguous array. A node refers to its four children by the array index of the first — they are stored contiguously — rather than by a boxed pointer. This keeps siblings cache-adjacent, needs a single allocation, and, being purely index-based, permits safe parallel traversal. The child link is an `Option<NonZeroU32>`, which the niche packs so that a node stays 64 bytes — exactly one cache line; a `None` marks a leaf.

Now the field algebra, in full. A primitive is a pair — a field `f` and its declared Lipschitz constant `L`. A bucket represents their pointwise minimum, whose constant is the maximum of theirs; the `Lipschitz` trait derives it automatically along any combinator chain.

The core predicate decides whether `f ≥ g` everywhere on a rectangle. Since `f − g` is `(L_f + L_g)`-Lipschitz,
{formula: (f-g)(\vec v)\;\ge\;(f-g)(\vec c)\;-\;(L_f + L_g)\,h(R)\qquad \forall\, \vec v \in R}
{read: "for every v in the rectangle R, f minus g at v is at least its value at the centre c, minus the sum of the Lipschitz constants L-f and L-g, times h of R, the half-diagonal of R."}
This drives a branch-and-bound over rectangles. Let delta be `f − g` at the centre:
{formula: \begin{aligned}
\delta < 0 &\;\Rightarrow\; \textbf{false} && (\exists\,\vec v:\ f(\vec v) < g(\vec v))\\
\delta \ge (L_f+L_g)\,h(R) &\;\Rightarrow\; \text{discard } R && (f \ge g \text{ on all of } R)\\
d \ge k &\;\Rightarrow\; \textbf{false} && (\text{undecided within budget})\\
\text{else} &\;\Rightarrow\; \text{split } R \text{ into 4 quadrants}
\end{aligned}}
{read: "If delta is negative, return false — a witness where f is below g exists. If delta is at least the Lipschitz sum times the half-diagonal, discard R — f dominates g on all of it. If we've hit the depth budget k, return false, conservatively. Otherwise split R into four quadrants and recurse. If the stack empties, return true."}
The test is *sound* — it returns true only when `f ≥ g` genuinely holds, so pruning can never corrupt the field — and merely *conservative* otherwise. Its cost is adaptive: well-separated fields settle at the root; a real witness is reached by descent, not by a fixed grid.

Insertion of a primitive `f` over a domain visits each overlapping leaf independently, in parallel, deciding via that predicate:
{formula: \begin{aligned}
f \ge g_{B_n} \text{ on } R_n &\;\Rightarrow\; \text{no-op} && (f \text{ never lowers the field})\\
g_{B_n} \ge f \text{ on } R_n &\;\Rightarrow\; B_n \leftarrow \{f\} && (f \text{ dominates the node})\\
\mathrm{depth}(n) = D \ \lor\ \lvert B_n\rvert < \beta &\;\Rightarrow\; B_n \leftarrow B_n \cup \{f\} && (\text{append})\\
\text{else} &\;\Rightarrow\; \text{subdivide},\ B_c \leftarrow \mathrm{prune}(B_n \cup \{f\},\, R_c)
\end{aligned}}
{read: "If f is at least the bucket field on the leaf, it lowers nothing — no-op. If the bucket field is everywhere at least f, then f dominates — replace the bucket with f alone. If we're at maximum depth D, or the bucket is under capacity beta, append f. Otherwise subdivide, and give each child the combined set, pruned."}
{formula: \mathrm{prune}(B, R)\;=\;\bigl\{\, (f_i, L_i) \in B \ :\ \lnot\bigl(f_i \ge \min_{j \ne i} f_j \ \text{on } R\bigr)\,\bigr\}}
{read: "prune of B over R keeps exactly those primitives f-i that are not everywhere dominated by the minimum of the others — that is, that still define the field somewhere in R."}
Because every comparison uses the sound predicate, a primitive is dropped only when *provably* redundant: the stored field never deviates from the true minimum. Pruning errs toward keeping, never toward corrupting.

Finally, the insertion domain — where can placing a shape actually change the field? A shape placed at a maximum `x_0` fits inside the free ball of radius `d = g(x_0)`, so `f(v) ≥ ‖v − x_0‖ − d`, and the field can change only inside
{formula: D^{*}\;=\;\bigl\{\, \vec v \ :\ g(\vec v) > \lVert \vec v - \vec x_0\rVert - d \,\bigr\}}
{read: "D-star is the set of points v where the current field g of v exceeds the distance from v to x-zero, minus d."}
For a **global** maximum, `g ≤ d` everywhere, so `D*` sits inside the ball of radius `2d`; its minimal axis-aligned cover is the square of side `4d` — and that constant `4` is optimal, attained by two tangent balls. Historically, I've come up with `4·√2·d` bound by trial and error, lacking any theoretical basis. In the end, it was sound but twice oversized.
For a **local** maximum, no square of side proportional to `d` is sound at all: three contact points with gaps under 180 degrees leave an escape ray along which the field stays above `f`, so the update region is unbounded in units of `d`. Instead of a fixed rectangle, `insert_at_maximum` prunes the tree walk itself, discarding a subtree `R` once
{formula: \hat g(\vec c_R) + L_B\, h(R)\;\le\;\mathrm{dist}(R, \vec x_0) - d}
{read: "g-hat at the centre of R, plus the bucket constant L-B times the half-diagonal, is at most the distance from R to x-zero, minus d."}
Here `g-hat` is the node's own bucket field — exact at leaves, and at internal nodes a pre-subdivision snapshot, which stays a valid upper bound because insertions only ever lower the field.

### Scene 5
Unlike the bitmap approach, newly obtained representation is lossless, offering a 10–100× memory reduction, and is continuously differentiable — making it possible to reintroduce an iterative optimizer. For practical purposes, I use an adaptive gradient ascent, which makes GD-ADF a __local-maximum method__.

{visual: a trajectory climbing a distance field; near the medial-axis ridge the raw gradient alternates sides, while the momentum-blended direction runs straight along the ridge; the step grows on accepted moves and halves on rejected ones}
A candidate step of length `h` is taken only when it improves the field, so the iterate is monotone. The step grows on acceptance and shrinks on rejection. Crucially, the previous accepted direction is blended into the next:
{formula: \vec d_k \;=\; \widehat{\ \frac{\nabla g(\vec p_k)}{\lVert \nabla g(\vec p_k)\rVert} + \vec d_{k-1}\ }}
{read: "d-k is the normalized sum of the unit gradient of g at p-k and the previous direction d-k-minus-one."}
{formula: \vec p_{k+1} = \begin{cases}\vec p_k + h_k\,\vec d_k & \text{if } g \text{ improves}\\ \vec p_k & \text{otherwise}\end{cases}\qquad h \leftarrow \begin{cases} h\cdot\text{growth}\\ h\cdot\text{decay}\end{cases}}
{read: "p-k-plus-one is p-k plus h-k times d-k if g improves, else p-k unchanged; and h is multiplied by the growth factor on acceptance, or the decay factor on rejection."}
The maxima of a distance field lie on the medial axis, where `g` is not differentiable and the raw gradient zig-zags across the ridge. The momentum term cancels the across-ridge components, leaving travel along it. This beats my original fixed exponential-decay schedule on two counts: that schedule burns a constant number of steps regardless of convergence and cannot refine a kink, whereas the adaptive rule stops the moment the step falls below tolerance and bisects onto the exact maximum. On my benchmark it locates the maximum up to 3 orders of magintude smaller, in fewer field evaluations. And a vanishing gradient — a flat region such as the clamped interior of an estimator — terminates the ascent immediately, rather than wasting the whole iteration budget.

{visual: `find_max_parallel`: a batch of random seeds ascend independently and in parallel; then a sequential pass keeps a maximum only if its free ball is disjoint from every already-kept ball; survivors are inserted}
Full parallelism is delicate, because a whole batch of maxima is found against one snapshot of the field, yet each insertion changes it. The ascents themselves are read-only, hence embarrassingly parallel. The danger is in applying their results: two maxima from the same batch could place overlapping shapes. So a batch is deduplicated — a maximum is accepted only if its free ball is disjoint from every already-accepted one:
{formula: \text{accept } \vec m_j \iff \lVert \vec m_i - \vec m_j\rVert > d_i + d_j \quad \forall\ \text{accepted } i}
{read: "accept m-j if and only if the distance from m-i to m-j exceeds d-i plus d-j, for every already-accepted maximum i."}
Disjoint free balls guarantee the shapes cannot intersect, regardless of insertion order, so the batch commits without a lock; the next batch re-reads the field fresh. The tree update mirrors this: a read-only parallel pass decides each leaf, then a short sequential pass applies the structural changes.

### Scene 6
So, how long does it take to make "A Million-Circle Fractal"?
{visual: ../../examples/gd_adf/02_random_distribution.rs, but with 1M circles, adf max depth = 10, gradually filling the space each frame}
{visual: assets/1M.png (final render), zooming in, panning around}
49 seconds on a 4-core machine. The resulting ADF tree contains 113k nodes, totalling 76MiB.

### Scene 7
So far we've only seen simple circle shapes. Let's try something much more funky: implement a Mandelbrot distance estimator, and fit 20k instances of it.
{visual: single mandelbrot DE SDF, with a grid of gradient arrows}
Can the current GD-ADF implementation handle it in reasonable time, if at all? The Mandelbrot estimator is not a true SDF — its gradient is unbounded near the boundary filaments, so the one-Lipschitz bound no longer applies — and moreover, the interior is clamped to 0 altogether. I therefore declare a larger Lipschitz constant, which keeps every soundness guarantee intact while merely relaxing the pruning; the constant is stated once on the type and propagates through every rotated, scaled, translated copy automatically.
{visual: ../../examples/gd_adf/06_custom_primitive.rs, gradually filling the space each frame}
7 seconds, 4.74 MiB, no obvious errors.

### Scene 8
Limitations and future work.
1. The current implementation only supports the 2D plane, but generalizing to N dimensions should be trivial. Neither the quadtree, the SDFs, nor the GD optimizer hold any assumptions about the dimensionality of the problem.
2. Only shape insertion is currently supported — no deletion or movement.
3. Basic drawing capabilities are provided, but it is advised to use a third-party library. The implementation was intended to be compatible with any drawing API out of the box.

### Scene 9
This is all, I hope you've learnt something new. Excited to see what kind of art you will make — feel free to share!
The rest of the video contains varied visuals from my time of working on the project. Enjoy!
