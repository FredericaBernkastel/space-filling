<!--
Conventions:
  body text        narration, read aloud
  {visual: …}      manim scene direction
  {formula: …}     equation rendered on screen (LaTeX)
  {read: "…"}      verbatim narration of the preceding formula (spoken)
Target audience: graduate level. Terse; no general-audience detours.
-->

### Scene 1
Liserotte here. The story opens with one equation:
{formula: \vec{x}^{*} \;=\; \arg\max_{\vec v \in \Omega}\ \min_{n}\ \mathrm{sdf}_{n}(\vec v)} (1)
{read: "x-star equals the arg-max, over the domain omega, of the pointwise minimum over all primitives n of sdf-n of v."}
Solving it numerically poses no difficulty; solving it in logarithmic time and linear memory is another matter entirely — and chasing that goal drew me into some genuinely fascinating mathematics.

> {visual: ../../readme.md, "Implementation" section, scrolling}

This video presents my original paper together with its reference implementation, which I've worked on for the past six years, mostly due to lack of theoretical knowledge on my part.

But let me start earlier still.
> {visual: https://paulbourke.net/fractals/randomtile, scrolling}

In 2011, Paul Bourke published *Random space filling of the plane*, a note in statistical geometry. This is actually a great read, I encourage you to pause here and check it out in full. 
It introduced an iterative algorithm for tiling E^n space with non-overlapping shapes. The visuals were indeed stunning; the algorithm behind it, far less so.

> {visual: assets/bourke_zoom.png, zooming out}

Here is *A Million-Circle Fractal*; the quoted generation time was 14.7 hours. Nor did the method grant much control over the resulting distribution. Can we do better?

### Scene 2
> {visual: densely filled space with shapes. random locations are tried one by one at least 10 times, all rejected due to being inside some shape, marked in red. until one is found to be outside, marked in green}

The flaw lies entirely in the random sampling. As more and more shapes are added, there is a vanishingly small chance of finding an empty spot. In the limit, it takes an unbounded amount of iterations to insert one more shape.  
In this visual we see several random samples ending up inside some shape, therefore rejected. Finally one is found to be outside; it is quite close to the edge though, but we have no choice here. A new circle will be added centered at current sample, even though a more optimal solution is clearly available. **This** is what I meant by the lack of control over distribution. 

Consider a different approach: represent every shape by a signed distance function.
> {visual: all distance functions of shapes from `shapes.rs` visualized in a grid, alongside their field notations}

For a planar set S, its signed distance function returns the distance to the boundary — negative inside the set, positive outside:
{formula: \mathrm{sdf}_{S}(\vec v)\;=\;\bigl(\mathbb{1}_{\vec v \notin S} - \mathbb{1}_{\vec v \in S}\bigr)\ \inf_{\vec u \in \partial S}\lVert \vec v - \vec u\rVert}
{read: "sdf-S of v equals plus-or-minus — plus outside S, minus inside — the infimum over boundary points u of the distance from v to u."}
Two properties will carry us through: the gradient has unit magnitude almost everywhere, and the function is one-Lipschitz — its value cannot change faster than you move.

> {visual: combining all the shapes from the grid into one compound distance field}

Now fold all the primitive SDFs into a single compound field, by taking their pointwise minimum:
{formula: g(\vec v)\;=\;\min_{n}\ \mathrm{sdf}_{n}(\vec v)}
{read: "g of v is the minimum, over all primitives n, of sdf-n of v."}
A minimum of one-Lipschitz functions is itself one-Lipschitz, so `g` inherits the very same regularity.

> {visual: a single gradient ascent trajectory across a real SDF field}

Because an SDF's gradient always points away from its surface, we can build an iterative algorithm to travel along that gradient, naturally finding a local maximum outside of any shape — precisely the arg-max of equation (1). This algorithm has a name: gradient ascent.
Problem solved? Not quite. Evaluating that pointwise minimum over N primitives costs O(N); filling the plane with N shapes therefore costs O(N^2). We have merely traded one bottleneck for another of the same order.

### Scene 3
> {visual: an N×N grid whose cells are threaded by a Z-order (Morton) curve; the compound field is rasterized onto it as a heatmap; a single linear scan sweeps the cells, tracking the running maximum, which lands on the brightest cell}

Consider a bitmap of size N×N and rasterize the field onto it, one single-precision sample per pixel. Lay the pixels out along a Z-order curve, so that spatially close samples stay close in memory, then scan them one by one for the global maximum. Does it hold up?
> {visual: ../../examples/argmax2d/01_fractal_distribution.rs, iteratively add one more shape every frame}

Good enough for a toy model. A lookup is now O(1), in principle — but increasing precision costs both quadratic time *and* quadratic memory. The 4096×4096 grid shown here already suffers an average discretization error of 2.44e-4 while consuming 64MB. One can can come up with more optimizations, split it in chunks; but this will merely change the constant before O. A dead end.

### Scene 4
> {visual: assets/Adaptively sampled distance fields.pdf, scrolling}

The paper *Adaptively Sampled Distance Fields* (2000) (doi:10.1145/344779.344899) proposes reducing memory by approximating the field within each node of a k-d tree — yet gives little guidance for a practical implementation beyond fitting a polynomial per node. Precision would then be limited by chosen polynomial degree as well as the quality of the regression algorithm.
Current work takes a different route: each node — a *bucket* — stores the primitives (function pointers) themselves.

> {visual: a quadtree subdividing over the plane; each leaf highlights only the few primitives whose fields reach into it; a query point descends root-to-leaf, then min-reduces over that leaf's short bucket}

The purpose of the buckets is spatial locality of relevance.  Far from a shape, that shape never wins the minimum, so there is no reason to keep it there. Each leaf retains only the handful of primitives sufficient to fully define the field inside it. To sample `g` at a point, descend to its leaf and minimise over that one short bucket:
{formula: g(\vec v)\;=\;\min_{f \in B(\ell(\vec v))} f(\vec v),\qquad \text{cost }\;O(\mathrm{depth} + \beta)}
{read: "g of v is the minimum over the primitives f in the bucket of the leaf containing v; the cost is order depth plus beta, the bucket size."}
On a balanced tree the depth is logarithmic and the bucket size `β` stays constant, so a query runs in O(log N) — never once touching the millions of distant primitives.

The next section develops the algebraic formalism in its entirety, on which the crucial optimizations rely. If you'd rather see the final results instead, jump to 00:00.

### Scene 5
<!-- Lipzschitz-continuity -->
> {visual: a 1-D slice of the field drawn as a curve; at a sample point c, draw the forward cone of slope ±L; slide c along the axis and show the curve never leaving the swept cone. Inset: two nearby points a, b with the bars |f(a)−f(b)| and L·|a−b|, the first always the shorter}

The soundness of everything that follows rests on Lipschitz continuity. A function `f` is `L`-Lipschitz when
{formula: \lvert f(\vec a) - f(\vec b)\rvert \;\le\; L\,\lVert \vec a - \vec b\rVert \qquad \forall\, \vec a, \vec b}
{read: "the absolute value of f of a minus f of b is at most L times the distance from a to b, for all a and b."}
Geometrically, the graph is confined to a cone of slope `L`, so a single sample together with the constant pins the function across an entire neighbourhood. True SDFs are one-Lipschitz; estimators simply declare a larger `L`.

<!-- sdf_geq_everywhere -->
> {visual: a 1-D interval; f and g hidden. Regular samples appear one at a time — at each, a dot of f above a dot of g, a green check; all pass. Then the true curves are revealed: a sharp V-spike of f between two adjacent samples dips below g, the crossing window flashes red — the checks stay green, yet the conclusion is false. Closing note: a regular 2D grid costs O(N²)}

Next, assume you do not yet know the definition of Lipschitz-continuity we just discussed. A quick task for you: given 1-dimensional interval and two functions - f and g: how would you prove that `f ≥ g` on the entire interval? To clarify, you do not yet know anything about the functions at this moment either, and the only operation you are permitted is to sample a value of the function, one place at a time.  
Perhaps you can sample it on regular intervals, stopping once you found a spot where `f < g`. But what if all your samples were positive, but there was a sharp spike in between your samples? Your strategy will yield a false-positive, leading to some nasty issues down the line. And besides, in 2D case a regular grid test will take O(N^2) in terms of the resolution, you can only increase it so far.  
Another idea? Perhaps come with some fancy interior point method optimizer? Think on your own.  

Here is my solution: since `f − g` is `(L_f + L_g)`-Lipschitz,
{formula: (f-g)(\vec v)\;\ge\;(f-g)(\vec c)\;-\;(L_f + L_g)\,h(R)\qquad \forall\, \vec v \in R}
{read: "for every v in the rectangle R, f minus g at v is at least its value at the centre c, minus the sum of the Lipschitz constants L-f and L-g, times h of R, the half-diagonal of R."}
> {visual: a live ADF-node rectangle over the field f−g. At each rectangle: sample the centre δ and draw the Lipschitz slack (L_f+L_g)·h as a symmetric interval around δ; colour the rectangle green when δ ≥ slack (f≥g proven — discard), red when δ<0 (witness — abort, return false), amber otherwise (undecided). Recurse, subdividing only the amber rectangles into quadrants, so the amber frontier races down the f=g contour until every rectangle is green or one turns red}  

That inequality powers a branch-and-bound over rectangles. Let delta be `f − g` at the centre:
{formula: \begin{aligned}
\delta < 0 &\;\Rightarrow\; \textbf{false} && (\exists\,\vec v:\ f(\vec v) < g(\vec v))\\
\delta \ge (L_f+L_g)\,h(R) &\;\Rightarrow\; \text{discard } R && (f \ge g \text{ on all of } R)\\
d \ge k &\;\Rightarrow\; \textbf{false} && (\text{undecided within budget})\\
\text{else} &\;\Rightarrow\; \text{split } R \text{ into 4 quadrants}
\end{aligned}}
{read: "If delta is negative, return false — a witness where f is below g exists. If delta is at least the Lipschitz sum times the half-diagonal, discard R — f dominates g on all of it. If we've hit the depth budget k, return false, conservatively. Otherwise split R into four quadrants and recurse. If the stack empties, return true."}
The test is *sound*: it answers true only when `f ≥ g` truly holds, and is otherwise merely *conservative*. Its cost adapts — well-separated fields resolve at the root, while a genuine witness is found by descent rather than by any fixed grid.

<!-- insert_primitive_domain -->
> {visual: a live ADF over the compound field (boundary wall w + three circles c₁..c₃), drawn as a contour plot with the leaf tree overlaid; every leaf shows its bucket size |Bₙ|. The four insertion rules are listed with colour keys. A new circle f (dashed, amber) appears at the field's argmax with radius d = g(x₀); classification ripples outward from f — grey no-op leaves dim out, blue replace (digit → 1), green append (digit + 1, one of them depth-capped), amber subdivide. Subdividing leaves split into four with per-child bucket sizes; one such leaf is magnified beside the plot: its set Bₙ ∪ {f} flows into the four children, each child's prune strikes out the provably redundant members (w, c₂ — or f itself where it never wins), and the surviving lists are visibly shorter. Finally the field commits to min(g, f): tints lift, the split lines join the tree, and the refined ADF sits over the updated field}  

Inserting a primitive `f` over a domain visits every overlapping leaf independently and in parallel, resolving each through that predicate:
{formula: \begin{aligned}
f \ge g_{B_n} \text{ on } R_n &\;\Rightarrow\; \text{no-op} && (f \text{ never lowers the field})\\
g_{B_n} \ge f \text{ on } R_n &\;\Rightarrow\; B_n \leftarrow \{f\} && (f \text{ dominates the node})\\
\mathrm{depth}(n) = D \ \lor\ \lvert B_n\rvert < \beta &\;\Rightarrow\; B_n \leftarrow B_n \cup \{f\} && (\text{append})\\
\text{else} &\;\Rightarrow\; \text{subdivide},\ B_c \leftarrow \mathrm{prune}(B_n \cup \{f\},\, R_c)
\end{aligned}}
{read: "If f is at least the bucket field on the leaf, it lowers nothing — no-op. If the bucket field is everywhere at least f, then f dominates — replace the bucket with f alone. If we're at maximum depth D, or the bucket is under capacity beta, append f. Otherwise subdivide, and give each child the combined set, pruned."}
{formula: \mathrm{prune}(B, R)\;=\;\bigl\{\, (f_i, L_i) \in B \ :\ \lnot\bigl(f_i \ge \min_{j \ne i} f_j \ \text{on } R\bigr)\,\bigr\}}
{read: "prune of B over R keeps exactly those primitives f-i that are not everywhere dominated by the minimum of the others — that is, that still define the field somewhere in R."}
Each comparison runs through that same sound predicate, so a primitive is discarded only when provably redundant. The stored field is never corrupted; pruning can err only toward keeping.

<!-- Insertion domains -->
> {visual: two panels. GLOBAL maximum: the free ball B(x₀,d), the enclosing B(x₀,2d) that bounds where the field can still exceed |v−x₀|−d, and the tight axis-aligned 4d square around it; a second ball, tangent, kisses the square's edge — witnessing that the constant 4 cannot be lowered. LOCAL maximum: a maximum pinned by three contact points with gaps under 180°, an escape ray w along which the field stays above f and the update region D* streaks outward without bound; then the insert_at_maximum walk lights up only the subtrees satisfying ĝ(c_R)+L_B·h(R) > dist(R,x₀)−d, hugging D* instead of any fixed box}  

One question remains: where can an insertion actually alter the field? A shape placed at a maximum `x_0` fits inside the free ball of radius `d = g(x_0)`, so `f(v) ≥ ‖v − x_0‖ − d`, and the field can shift only within
{formula: D^{*}\;=\;\bigl\{\, \vec v \ :\ g(\vec v) > \lVert \vec v - \vec x_0\rVert - d \,\bigr\}}
{read: "D-star, the set of points v where the current field g of v exceeds the distance from v to x-zero, minus d."}
For a **global** maximum, `g ≤ d` everywhere, so `D*` sits inside the ball of radius `2d`; its minimal axis-aligned cover is the square of side `4d` — and that constant `4` is optimal, attained by two tangent balls. Historically, I'd arrived at a `4·√2·d` bound by trial and error, lacking any theoretical basis. In the end, it was sound but twice oversized.
For a **local** maximum, no square of side proportional to `d` is sound at all: three contact points with gaps under 180 degrees leave an escape ray along which the field stays above `f`, and so we say that `D*` is inexpressible in units of `d`. Instead of a fixed rectangle, `insert_at_maximum` prunes the tree walk itself, discarding a subtree `R` once
{formula: \hat g(\vec c_R) + L_B\, h(R)\;\le\;\mathrm{dist}(R, \vec x_0) - d}
{read: "g-hat at the centre of R, plus the bucket constant L-B times the half-diagonal, is at most the distance from R to x-zero, minus d."}
Here `g-hat` is the node's own bucket field — exact at a leaf, and at an internal node a pre-subdivision snapshot, which stays a valid upper bound because insertions only ever push the field down.

<!-- quadtree arena -->
> {visual: a single contiguous array of nodes; each internal node points to the index of the first of its four children, drawn as an arrow into the packed block; sibling cells adjacent in memory}  

Lastly, I'd like to briefly mention quadtree implementation itself.
Rather than dynamically allocating each node in heap, the tree lives in a flat arena — every node in one contiguous array. A node names its four children by the index of the first, since the four sit contiguously. Siblings stay cache-adjacent, the whole structure takes a single allocation, and, being purely index-based, it admits lock-free parallel traversal. The child link is an `Option<NonZeroU32>`; the niche fits it exactly into 4 bytes, and the whole node is exactly 64 bytes — one cache line — with `None` variant marking a leaf. Optimizing memory footprint further is definitely possible, but not in mine scope.

### Scene 6
> {visual: title card — GD-ADF: lossless, 10–100× less memory than the bitmap, continuous field ⇒ gradient ascent; a local-maximum method}  

Unlike the bitmap approach, newly obtained representation is lossless, offering a 10–100× memory reduction, and is continuously differentiable — making it possible to reintroduce an iterative optimizer. For practical purposes, I use an adaptive gradient ascent, which makes GD-ADF a __local-maximum method__.

> {visual: the optimize_precision test field (two point obstacles + the walls, mapped to [-2,2]²) as a contour plot, the two update formulas at the right. From one seed the raw-gradient ascent runs step by step — accepted moves solid red, rejected trials dashed and faded — zig-zagging across the medial-axis ridge, while an h-history bar chart (log scale, baseline Δ) grows green on accepts, red on rejects. The momentum ascent then runs from the same seed in cyan, straight along the ridge, its own h-chart below; the kink maximum is marked and each method's live verdict lands: field evaluations and final distance to the apex (raw stalls ~1e-1 away; momentum ends ~5e-3 away on fewer evaluations)}  

A trial step of length `h` is accepted only if it raises the field, so the iterate climbs monotonically. The step lengthens when a move succeeds and contracts when it fails. Watch the plain rule run first, its direction taken straight from the gradient. The maxima of a distance field lie on the medial axis, where `g` is not differentiable: nearing the ridge, the raw gradient flips sides on every trial — the path zig-zags across the crease, rejected steps burn through the evaluation budget, and the iterate stalls short of the apex.
The cure costs a single vector — fold the last accepted direction into the next:
{formula: \vec d_k \;=\; \mathrm{normalize}\!\left(\frac{\nabla g(\vec p_k)}{\lVert \nabla g(\vec p_k)\rVert} + \vec d_{k-1}\right)}
{read: "d-k is the normalized sum of the unit gradient of g at p-k and the previous direction d-k-minus-one."}
The across-ridge components now cancel, and the second run travels straight along the ridge, settling onto the kink. The complete update:
{formula: \vec p_{k+1} = \begin{cases}\vec p_k + h_k\,\vec d_k & \text{if } g \text{ improves}\\ \vec p_k & \text{otherwise}\end{cases}\qquad h \leftarrow \begin{cases} h\cdot\text{growth}\\ h\cdot\text{decay}\end{cases}}
{read: "p-k-plus-one is p-k plus h-k times d-k if g improves, else p-k unchanged; and h is multiplied by the growth factor on acceptance, or the decay factor on rejection."}
This beats my original fixed exponential-decay schedule on two counts: that schedule burns a constant number of steps regardless of convergence and cannot refine a kink, whereas the adaptive rule stops the moment the step falls below tolerance and bisects onto the exact maximum. On my benchmark it locates the maximum up to 3 orders of magnitude more precisely, in fewer field evaluations. And a vanishing gradient — a flat region such as the clamped interior of an estimator — terminates the ascent immediately, rather than wasting the whole iteration budget. Compare them once more. {visual: repeat last one}

> {visual: `find_max_parallel` — a mid-fill field (wall + five circles); ten seeds ascend simultaneously along cyan trajectories to their local maxima, each capped by its free ball, the accept rule shown above. A sequential sweep in batch order flashes each ball green (kept) or red (overlaps an already-kept ball — the two pockets found twice lose their duplicate) and rejected balls fade out. The survivors commit together: the field crossfades to the union with the eight new circles, the free balls settling onto the new white zero contours}  

Full parallelism is delicate: an entire batch of maxima is located against a single snapshot of the field, yet every insertion mutates it. The ascents are read-only, and so embarrassingly parallel; the hazard lies in committing their results, since two maxima from one batch might place overlapping shapes. The batch is therefore deduplicated — a maximum survives only if its free ball is disjoint from every one already kept:
{formula: \text{accept } \vec m_j \iff \lVert \vec m_i - \vec m_j\rVert > d_i + d_j \quad \forall\ \text{accepted } i}
{read: "accept m-j if and only if the distance from m-i to m-j exceeds d-i plus d-j, for every already-accepted maximum i."}
Disjoint free balls guarantee the shapes cannot overlap, whatever the insertion order, so the batch commits without a lock, and the next batch reads the field afresh. The tree update follows the same shape: a read-only parallel pass decides every leaf, then a brief sequential pass applies the structural edits.

### Scene 7
So, how long does it take to make "A Million-Circle Fractal"?
> {visual: the real run, embedded: render/src/bin/random_distribution.rs — 02_random_distribution verbatim, driven to 1M circles at ADF max depth 10 — pre-rendered to assets/derived/random_distribution.mp4 on a geometric frame schedule (~330 frames from the first 4 circles to the full million, so the fill accelerates smoothly). Beside the footage a live "circles inserted" counter follows the same schedule up to 1 000 000} 

> {visual: assets/1M.png (the finished 8192² render), a pre-baked zoom/pan pass (assets/derived/million_zoom.mp4): full view, dive to 6.4×, diagonal pan across, pull back out; then the quoted figures land on a dim strip over the artwork — 49 s / 113 k nodes / 76 MiB}  

49 seconds on a 4-core machine. The resulting ADF tree contains 113k nodes, totalling 76MiB.

### Scene 8
So far we've only seen simple circle shapes. Let's try something much more funky: implement a Mandelbrot distance estimator, and fit 20k instances of it.
> {visual: single mandelbrot DE SDF, with a grid of gradient arrows}

Can the current GD-ADF implementation handle it in reasonable time, if at all? The Mandelbrot estimator is not a true SDF — its gradient is unbounded near the boundary filaments, so the one-Lipschitz bound no longer applies — and moreover, the interior is clamped to 0 altogether. I therefore declare a larger Lipschitz constant, which keeps every soundness guarantee intact while merely relaxing the pruning; the constant is stated once on the type and propagates through every rotated, scaled, translated copy automatically.
> {visual: ../../examples/gd_adf/06_custom_primitive.rs, gradually filling the space each frame}

7 seconds, 4.74 MiB, no obvious errors.

### Scene 9
> {visual: four numbered rows, each with a small glyph — a wireframe cube (N-D), operation chips (+ kept, − and ↔ struck out), a picture frame (drawing), an IC package (GPU)}

Limitations and future work.
1. Today the implementation covers only the 2D plane, though extending it to N dimensions ought to be straightforward: nothing in the quadtree, the SDFs, or the optimizer assumes a particular dimensionality.
2. Only insertion is supported for now — no deletion, no movement.
3. Basic drawing is included, but I recommend a third-party library; the implementation was intended to be compatible with any drawing API out of the box.
> {visual: the GPU assessment as its own composition — a CPU panel ("owns every mutation": insert/subdivide/prune, the sound f64 path, dedup + refine the survivors) facing a GPU panel ("read-only field oracle": the 64-B node arena drawn as a strip that uploads as-is, mega-batches of f32 ascents), joined by two arrows: arena deltas →, ← batch maxima (f32). Below, Arc<dyn Fn> struck through → "shape IR" and "baked distance fields" pills; then the verdict chips — 10–100× (ascent phase) vs 3–10× (end to end, Amdahl) — with the batch-size caveat as a footnote}

4. A GPU port is feasible, with following assessment: the device should serve as a *read-only field oracle*, never the mutator. The flat quadtree arena uploads as-is — and the embarrassingly parallel ascents of `find_max_parallel` are the natural kernel: coarse single-precision batches on the device, with double-precision refinement of the few survivors, and every insertion, staying on the sound CPU path. The price is generality: buckets store arbitrary function pointers, which a GPU cannot process, so primitives must be reified into a fixed shape IR — baked distance fields covering everything exotic.  
The theoretical gains are lopsided: 10–100× throughput on the ascent phase itself, yet perhaps 3–10× end to end, by Amdahl's law, while the structural updates remain on the CPU. Nor can the batch simply grow with the hardware — a batch's survivors are capped by how many disjoint free balls the current field admits, so the useful batch size scales with the fill state, not with the core count.

### Scene 10
This is all, I hope you've learnt something new. Excited to see what kind of art you will make — feel free to share!
The rest of the video contains varied visuals from my time of working on the project. Enjoy!
