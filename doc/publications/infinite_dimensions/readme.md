# Beyond $\mathbb{R}^N$: space filling in infinite-dimensional function spaces

*Publication draft.*

---

## 0. Abstract

**Prior art.** Two lines of work meet here. Bourke's plane filling and Shier's fractal packings grow a
distribution one shape at a time, but hunt for each vacancy by direct search, which dominates their cost — Shier
reports 14.7 hours for a million circles. Separately, Frisken et al. compressed distance fields into a hierarchy
of per-node approximations: a representation built for rendering and modelling, not for finding room to put
something.

**Purpose.** The [`space-filling`](../../../readme.md) crate joins the two. Finding room becomes maximizing the
compound field,

$$
x^{\ast} = \arg\max_{v \in \Omega} \min_n \mathrm{sdf}_n(v) ,
$$

and the hierarchy carries the primitives exactly rather than approximating them, so dropping one demands a proof
rather than a tolerance — branch-and-bound over Lipschitz bounds, which leaves the represented field equal to the
true minimum everywhere. [`Argmax2D`](../../../src/solver/argmax2d/mod.rs) maximizes exactly over a bitmap and
pays quadratically for resolution; GD-ADF climbs to a local maximum, sampling the tree in logarithmic time. Both
are $N$-dimensional as of version 0.6.0.

**The question here.** Generalizing from $\mathbb{R}^2$ to $\mathbb{R}^N$ was trivial — nothing in the
pruning proof, the insertion bound or the optimizer depended on problem dimensionality. Going from $\mathbb{R}^N$ to a
function space is **not** the same kind of move. One of the two solvers dies, and it dies for an
information-theoretic reason that no amount of engineering repairs:

| | `Argmax2D` (exact global) | GD-ADF (local ascent) |
|---|---|---|
| what it needs | a $\delta$-cover of $\Omega$ | a descent direction |
| cost model | $\varepsilon^{-N}$ oracle calls | $L^2R^2/\varepsilon^2$ iterations |
| dimension in the exponent | **yes** | **no** |
| survives $N \to \infty$ | never | conditionally |

The Lipschitz certificate is a **covering argument**, and covering
numbers are exactly what blow up. Gradient ascent, by contrast, has dimension-free rates in any Hilbert space,
which is precisely why function-space optimization is a mature field while function-space *global* optimization
is not.

But the negative result has a sharp condition attached, and the condition is the interesting part:

> **The obstruction is not infinite dimension. It is infinite dimension without an ordering of importance.**
> On a compact ellipsoid $\lbrace x : \sum_i a_i x_i^2 \le 1 \rbrace$ with $a_i \to \infty$, covering numbers are
> finite, the maximum is attained, and the certificate returns — with cost polynomial in $1/\varepsilon$ rather
> than exponential in $N$. Infinitely many coordinates are affordable as long as they matter progressively less.

Four results follow. Coordinate weights are the precondition for any of this machinery to survive at all, and they
convert a cost exponential in $N$ into one polynomial in $1/\varepsilon$ (§2). The greedy strategy keeps a
dimension-free 2-approximation in any metric space, so what degrades is the proof of local redundancy and not the
quality of the search (§4). The clearance sequence measures the dimension a distribution actually occupies —
recovering $2.10$ for a known planar point set and $1.89$ for the fractal distribution of example 01, a
measurement run through the library itself (§2.1). And the present greedy step leaves a factor of $\kappa^N$ of
placed volume unclaimed, because it never optimizes orientation (§6.1).

---

## 1. Four different things "infinite-dimensional" can mean

The phrase hides at least four distinct programmes with very different prospects (Figure 1). They should not be
conflated.

![Figure 1. Four readings of "infinite-dimensional": A the domain becomes a Hilbert ball or weighted ellipsoid; B each primitive's boundary becomes a function; C the field itself becomes the variable, ranging over a lattice of 1-Lipschitz functions; D the field is drawn from a parametric family with a certified Lipschitz bound. A comparison table gives, for each, what is varied, where it lives, the form the gradient takes, whether the Lipschitz certificate survives, and what drives the cost.](figures/fig1-readings.svg)

---

## 2. Reading A — the domain becomes infinite-dimensional

Let $H$ be a separable Hilbert space, $\Omega \subset H$ the region to fill, and as before

$$
g_k(x) = \min_{n \le k} \mathrm{sdf}_n(x), \qquad x_{k+1}^{\ast} = \arg\max_{x \in \Omega} g_k(x).
$$

### 2.1 What geometry does to you first

Before any algorithmic concern, the *problem statement* degrades. In $[0,1]^N$:

| $N$ | children per split, $2^N$ | volume of inscribed ball, $r = 1/2$ | fraction within $0.05$ of $\partial\Omega$ | $\mathcal{N}(\Omega, 1/8)$ |
|---:|---:|---:|---:|---:|
| 2 | 4 | $7.9 \times 10^{-1}$ | 0.19 | $10^{2}$ |
| 3 | 8 | $5.2 \times 10^{-1}$ | 0.27 | $10^{3}$ |
| 4 | 16 | $3.1 \times 10^{-1}$ | 0.34 | $10^{4}$ |
| 8 | 256 | $1.6 \times 10^{-2}$ | 0.57 | $10^{7}$ |
| 10 | 1 024 | $2.5 \times 10^{-3}$ | 0.65 | $10^{9}$ |
| 20 | 1 048 576 | $2.5 \times 10^{-8}$ | 0.88 | $10^{18}$ |
| 32 | $4.3 \times 10^{9}$ | $1.0 \times 10^{-15}$ | 0.97 | $10^{29}$ |
| 64 | $1.8 \times 10^{19}$ | $1.7 \times 10^{-39}$ | 0.999 | $10^{58}$ |

Three separate deaths are visible in that table.

**The cube stops being round.** Its inradius is $1/2$ for every $N$, but its circumradius is $\sqrt{N}/2$. The
inscribed ball occupies a vanishing fraction of the volume, so *filling with balls becomes meaningless*: the
distribution can achieve $10^{-15}$ coverage and still be optimal at every step. The right primitives in high
dimension are anisotropic — which the crate already has, in `ProductBall` and the `duocylinder`.

**Everything is boundary.** The fraction of $[0,1]^N$ within $\varepsilon$ of the boundary is
$1 - (1-2\varepsilon)^N \to 1$. `boundary_rect` stops being a mild constraint and becomes the entire field.

**The maxima stop being informative.** Greedy farthest-point insertion has covering radius
$d_k \asymp k^{-1/N}$, so over a million insertions the clearance falls by $10^{-3}$ at $N = 2$, by $0.25$ at
$N = 10$, and by $0.76$ at $N = 50$. The sequence flattens: every insertion is nearly as good as the first, which
is the same as saying no insertion accomplishes anything.

![Figure 2. Clearance decay measured through the crate's own solvers. Panel (a) validates the estimator on farthest-first traversal in 2, 3 and 4 dimensions, where the covering radius must scale as k to the minus one over D; the fit recovers 2.10 against a true 2 but reads high in 3 and 4 dimensions, and a control series with four times the restarts shows this is a pre-asymptotic effect rather than search error. Panel (b) applies the same fit to the fractal distribution of example 01, obtaining non-integer effective dimensions of 1.89 and 1.21 with excellent fit quality. A results table gives slopes, predictions, fit quality and measured dimensions for all six series.](figures/fig2-decay.svg)

### 2.2 Where the certificate breaks, exactly

The proof at the heart of `sdf_geq_everywhere` is

$$
\forall v \in R : \quad (f-g)(v) \ge (f-g)(c_R) - (L_f + L_g) h(R),
$$

refined by subdivision until the margin covers the box. To resolve a true margin of $\delta$ it must reach boxes
of half-diagonal $h < \delta / (L_f + L_g)$, and the number of such boxes needed to cover $\Omega$ is its covering
number, which for a norm ball in $\mathbb{R}^N$ satisfies

$$
\left( \frac{1}{\delta} \right)^{N} \le \mathcal{N}(\Omega, \delta) \le \left( \frac{3}{\delta} \right)^{N}.
$$

In infinite dimensions $\mathcal{N}(\Omega, \delta) = \infty$ for every $\delta$ below the radius, because the
closed unit ball of an infinite-dimensional normed space is **not compact** (Riesz). Consequently:

- **Soundness survives.** The test still answers `true` only when the bound genuinely holds — the inequality
  above is metric and never referred to dimension.
- **Completeness collapses.** The test becomes vacuous: outside the lucky case where the root box already
  settles the question, no finite budget $k$ proves anything.
- **Attainment is no longer guaranteed.** $\mathrm{sdf}_n(x) = \lVert x - p_n \rVert - r_n$ is convex, so $g_k$ is
  a min of convex functions — neither convex nor concave, and only weakly *lower* semicontinuous. Maximizing it
  over a merely weakly-compact set has no existence theorem. The $\arg\max$ may simply not exist.

And well before any of that, the implementation hits a wall the mathematics doesn't care about: a $2^N$-tree has
$2^N$ children per node. At $N = 10$ that is a thousand-way branch per subdivision; at $N = 20$, a million.
**The tree, not the theory, is what fails first.**

### 2.3 The repair: weights, and the return of compactness

Replace the ball by an ellipsoid with decaying coordinate importance,

$$
\Omega_a = \left\lbrace x \in \ell^2 : \sum_{i=1}^{\infty} a_i x_i^2 \le 1 \right\rbrace, \qquad a_i \to \infty,
$$

which is the image of the unit ball under a compact diagonal operator and therefore **norm-compact**. Compactness
buys back everything at once: finite covering numbers, attained maxima, a meaningful branch-and-bound. And the
cost is no longer exponential in a dimension — it is polynomial in $1/\varepsilon$, with the exponent set by the
decay rate. For the Sobolev-type scaling $a_i \asymp i^{2s}$,

$$
\log \mathcal{N}(\Omega_a, \varepsilon) \asymp \varepsilon^{-1/s},
$$

so smoothness $s$ trades directly against certification cost:

| smoothness $s$ | $\log \mathcal{N}(10^{-2})$ |
|---:|---:|
| $0.5$ | $10^{4}$ |
| $1$ | $10^{2}$ |
| $2$ | $10$ |
| $4$ | $3.2$ |

This is the tractability theory of weighted spaces (Sloan–Woźniakowski) arriving in geometric clothing. Its
design consequence is concrete and immediate:

> Subdivide **one axis at a time, in weight order**, not all $N$ at once. A binary $k$-d split has branching
> factor 2 in every dimension; a full round of $N$ splits recovers one level of the $2^N$-tree at identical
> resolution, and creates only the nodes actually visited. Cycling axes by descending weight $\gamma_j$ makes the
> refinement *dimension-adaptive*: cost tracks the effective dimension rather than the ambient one.

That single change is the difference between a usable ceiling of $N \approx 10$ and one of $N \approx 10^2$, and
it is a change to the tree, not to any proof — `sdf_geq_everywhere` is indifferent to how the boxes were made.

![Figure 3. Two ways to refine a cell. (a) The current 2^N-tree splits every axis at once, allocating all 2^N orthants — 1024 children at N=10, over a million at N=20. (b) A weight-ordered k-d tree cuts one axis at a time: branching factor 2 in every dimension, the same resolution after N cuts, and only the cells the search descends into are allocated. (c) Children allocated per subdivision on a log scale: the fan-out is a straight line, the binary split a constant.](figures/fig3-refinement.svg)

Where do the weights come from? Not from a hyperparameter — from the parameterization itself. A Karhunen–Loève
expansion of a random field supplies eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$ with
$\sum_i \lambda_i < \infty$ by construction. Which is the bridge to the second reading.

---

## 3. Reading B — the primitives become functions

Here the infinite dimension sits in the *shape*, not the ambient space, and this is where classical machinery
applies almost untouched.

Let a primitive be star-shaped with radial profile $\rho \in L^2(S^{N-1})$, or a tube around a path
$\gamma \in H^1([0,1], \mathbb{R}^N)$. The design variable is now a function; the objective is a functional. Two
classical results carry the weight.

**Hadamard's structure theorem.** For $J(\Omega) = \int_{\Omega} j$, the shape derivative in the direction of a
deformation field $V$ depends only on the *normal* trace of $V$ on the boundary:

$$
dJ(\Omega)[V] = \int_{\partial \Omega} j (V \cdot n) \mathrm{d}s.
$$

Infinitely many degrees of freedom in $V$ collapse to one scalar field on $\partial\Omega$. This is the reason
shape optimization is possible at all, and it is a genuine dimension reduction rather than a discretization.

**The level-set method.** Advect the field itself by a Hamilton–Jacobi equation,

$$
\partial_t \phi + v \lVert \nabla \phi \rVert = 0,
$$

with $v$ read off the shape derivative. Note what this is: *gradient flow in function space, whose state variable
is an SDF*. This library already stores exactly that state. The classical difficulty of the method — $\phi$ drifts
away from being a distance function and must be periodically reinitialized — is, in this library's language, the
problem of maintaining the Lipschitz constant. The ADF's declared bound $L$ is a reinitialization-free surrogate:
a field that has drifted to $\lVert \nabla \phi \rVert \le L$ is still admissible, just with weaker pruning.
**The crate could host a level-set shape optimizer without changing its representation**, which is the most
interesting unexplored direction here.

---

## 4. Reading C — the field is the variable, and greedy is a known algorithm

Take the state space to be

$$
\mathcal{L} = \lbrace g : \Omega \to \mathbb{R} \mid g \text{ is } 1\text{-Lipschitz and } g = 0 \text{ on } \partial\Omega \rbrace,
$$

a convex, closed, bounded subset of $C(\Omega)$ — infinite-dimensional, and closed under pointwise $\min$, hence
a **lattice**. Insertion is the operator $T_f : g \mapsto \min(g, f)$, and the algorithm is the orbit
$g_{k+1} = T_{f_k} g_k$.

Three observations follow for free, and they are more than reformulation.

**1. The algorithm is a monotone flow.** $g_{k+1} \le g_k$ pointwise, always. A decreasing sequence bounded below
converges pointwise to some $g_{\infty}$, and by equi-Lipschitzness plus Arzelà–Ascoli the convergence is uniform
on compacts. The space-filling problem then becomes a question about limits: which $g_{\infty}$ are reachable? The
existing readme's remark that "insertions only ever lower the field" — used to justify the pre-subdivision
snapshot as a valid upper bound — is precisely monotonicity of $T_f$ on this lattice, and it is why the local
insertion-domain pruning is sound.

**2. Greedy insertion is Frank–Wolfe.** At each step the algorithm evaluates the linear functional
$g \mapsto \langle \delta_x, g \rangle = g(x)$ over all $x$, takes its maximizer, and moves the state toward the
extreme point that maximizer selects. That is the Frank–Wolfe / conditional-gradient template, whose $O(1/k)$
rate in Hilbert spaces is dimension-free. In an RKHS the same scheme is **kernel herding**. The greedy heuristic
is not a heuristic; it is conditional gradient on a lattice, and it inherits that literature's rates.

**3. The approximation guarantee is dimension-free.** Farthest-first traversal is a 2-approximation to
$k$-center in *any* metric space (Gonzalez, 1985). Greedy maximin insertion is exactly farthest-first traversal,
so:

> The clearance achieved after $k$ greedy insertions is within a factor 2 of the best possible $k$-point
> configuration — in $\mathbb{R}^2$, in $\mathbb{R}^{100}$, in $L^2$, in path space, in any metric space
> whatsoever. The *combinatorial* guarantee survives infinite dimensions intact. Only the certificate does not.

That asymmetry is the cleanest summary of the whole report: what breaks is the *proof of local redundancy*, not
the *quality of the global strategy*.

![Figure 4. The asymmetry at the centre of the report. (a) Farthest-first traversal, illustrated by five greedy insertions and their clearance balls, is a 2-approximation to k-center in any metric space, so the quality of the strategy is indifferent to dimension. (b) The pruning test must reach boxes of half-diagonal below delta over the summed Lipschitz constants, and the number of such boxes is exponential in N and infinite when the unit ball is non-compact; the test then remains sound but proves nothing below the root box. (c) On a compact weighted ellipsoid the entropy is polynomial in one over epsilon with the exponent set by the smoothness of the weight decay, so the ambient dimension leaves the exponent entirely.](figures/fig4-survival.svg)

---

## 5. Can a classical gradient optimizer be applied in function space?

Yes — with three caveats, one of which this library has already solved by accident.

### 5.1 In Hilbert space: essentially unchanged

For $J : H \to \mathbb{R}$ Fréchet differentiable, the derivative is a functional $DJ(u) \in H^{\ast}$, and the
Riesz representation theorem supplies a unique $\nabla J(u) \in H$ with
$DJ(u)v = \langle \nabla J(u), v \rangle$. The iteration $u_{k+1} = u_k + h \nabla J(u_k)$ is then literally the
same algorithm. Crucially, the standard rates never mention dimension. For an $L$-smooth objective,

$$
J^{\ast} - J(u_k) \le \frac{2L \lVert u_0 - u^{\ast} \rVert^2}{k+4},
$$

and for the nonsmooth subgradient method the rate is $O(GR / \sqrt{k})$. Dimension-freeness is not a convenience
here; it is the *entire reason* the infinite-dimensional case is viable while global search is not.

### 5.2 In Banach space: the gradient stops being canonical

$C(\Omega)$ and $\mathrm{Lip}(\Omega)$ — the natural homes for a distance field — are not Hilbert. There is no
Riesz map, so a differential does not determine a descent *direction*; you must choose a metric first. The same
functional under an $L^2$ versus an $H^1$ inner product yields different flows, related by

$$
\nabla_{H^1} J = (I - \Delta)^{-1} \nabla_{L^2} J,
$$

that is, the Sobolev gradient is a smoothed version of the $L^2$ one (Neuberger). In shape optimization this is
the difference between a flow that develops oscillations and one that stays regular. **The choice of inner
product is not a detail of the analysis; it is the design of the algorithm.**

### 5.3 Non-smoothness: already handled here

The objective is a pointwise $\min$, hence non-differentiable, and the maxima being sought sit *exactly* on the
non-differentiable set — the medial axis. The correct object is the Clarke subdifferential, and the correct
algorithm is a monotone nonsmooth ascent.

Which the crate already implements. `LineSearch` uses only the gradient's **direction**, blends the previous
accepted direction to cancel the across-ridge zigzag, accepts only improving steps, and terminates on a
vanishing sampled gradient. Not one line of that argument uses finite dimension. Written abstractly, with
$u_k$ the unnormalized blend,

let $a_k \in \lbrace 0, 1 \rbrace$ record whether the trial step improved the field,

$$
a_k = 1 \iff g(p_k + h_k d_k) > g(p_k),
$$

and the whole method is three lines with no piecewise definition anywhere:

$$
u_k = \frac{\nabla g(p_k)}{\lVert \nabla g(p_k) \rVert} + a_{k-1} d_{k-1},
\qquad
d_k = \frac{u_k}{\lVert u_k \rVert},
$$

$$
p_{k+1} = p_k + a_k h_k d_k,
\qquad
h_{k+1} = \min \left( h_0, h_k \gamma_{+}^{a_k} \gamma_{-}^{1 - a_k} \right),
\qquad
\gamma_{+} > 1 > \gamma_{-} > 0 .
$$

The indicator carries the entire control structure. In $p_{k+1}$ it makes the iterate monotone — a rejected step
multiplies the displacement by zero rather than branching. In $h_{k+1}$ it selects growth or decay as an exponent,
capped at the initial length $h_0$ so the step can never exceed its starting scale. And in $u_k$ it appears as
$a_{k-1}$, because momentum is *reset* on rejection rather than retained: the previous direction is blended in only
if it earned its place. The iteration ends when $h_k < \Delta$, when the sampled gradient vanishes, or at a step
limit. This is a Hilbert-space algorithm as it stands. **One thing must change**: the gradient is currently sampled by
finite differences along $N+1$ axis directions, which is meaningless when $N = \infty$. The replacement is a
randomized directional estimate over $m \ll N$ Gaussian directions,

$$
\widehat{\nabla} g(p) = \frac{1}{m} \sum_{i=1}^{m} \frac{g(p + \sigma \xi_i) - g(p)}{\sigma} \xi_i, \qquad
\xi_i \sim \mathcal{N}(0, C),
$$

with the covariance $C$ carrying the coordinate weights of §2.3.

Here honesty requires a caveat that optimism usually omits: **derivative-free rates are not dimension-free.**
Gaussian-smoothed methods (Nesterov–Spokoiny) pay an explicit factor of the ambient dimension, so this estimator
is only viable when the field has a low-dimensional *active subspace* — when $g$ genuinely varies in few
directions. Weighted domains are exactly the setting where that holds. Without weights, the ascent degrades to
random search; with them, it tracks the effective dimension.

---

## 6. Applications worth the trouble

Ordered by how far they are from working code.

### 6.1 Pose, not only position — the lift that fixes today's greedy step

Every reading so far has taken the greedy step for granted: find a maximum of $g$, place a shape there. But the
solvers hand the generator a *position and a clearance*, and nothing else. Orientation never enters the
optimization — in [`examples/gd_adf/04`](../../../examples/gd_adf/04_polymorphic.rs) it is drawn from
`rng.random_range(0.0..45)`, sampled rather than chosen. That is exact for balls and lossy for everything else,
and the loss has a closed form.

A unit shape is inscribed in the unit sphere, so scaling by the clearance matches the shape's **circumradius**
$R_S$ to $d$. But a shape clears the walls of a region through its **inradius** $r_S$. Writing
$\mathrm{vol}_{\mathrm{best}}$ for the largest admissible copy under any translation and rotation,

$$
\left( \frac{d}{R_S} \right)^{N} \mathrm{vol}(S)
\quad \le \quad \mathrm{vol}_{\mathrm{best}} \quad \le \quad
\left( \frac{d}{r_S} \right)^{N} \mathrm{vol}(S) .
$$

The left-hand side is what the crate places today. The right-hand side cannot be beaten: any admissible copy
contains its own inradius ball, that ball lies in free space, so $g$ at its centre is at least $s r_S$, whence
$s r_S \le \max g = d$. The two bounds differ by the shape's **anisotropy** raised to the dimension,

$$
\kappa^{N}, \qquad \kappa = R_S / r_S ,
$$

and the bound is attained. For a ball $\kappa = 1$ and today's rule is already optimal — which is precisely why
the deficiency has stayed invisible.

![Figure 5. The anisotropy bound. Panel (a) places a shape of aspect one-to-two in a free region of two by one, three ways: the present rule inscribes it in the free ball for an area of 0.40; the best axis-aligned copy reaches 0.50; the copy rotated by a right angle fills the region exactly at area 2.00, a factor of five. Panel (b) derives the bound from the inradius ball and shows the inradius and circumradius of the shape. Panel (c) tabulates the anisotropy and the volume left unclaimed for common shapes, from unity for a ball to twenty-seven for a regular tetrahedron.](figures/fig5-orientation.svg)

Figure 5 makes it concrete with a free region of $2 \times 1$ and a shape of aspect $1 : 2$. Today's rule places
area $0.40$; rotating by a right angle fills the region exactly, at area $2.00$. The ratio is $5.00$, which is
exactly $\kappa^2$ for $\kappa = \sqrt{5}$. Worse, $0.40$ is also the area at the *worst* orientation for that
region, so the orientation-blind rule realises the least favourable pose available to it. The shortfall grows
sharply with dimension: a regular tetrahedron has $\kappa = 3$ and so leaves $27 \times$ unclaimed, and even the
$N$-cube — the least anisotropic box there is — has $\kappa = \sqrt{N}$, hence a gap of $N^{N/2}$: sixteenfold in
$\mathbb{R}^4$ and $10^5$-fold in $\mathbb{R}^{10}$.

**The lift is the right instrument, and it is smaller than it looks.** Scale is not a free coordinate but the
objective itself, since a pose determines the largest admissible copy. The domain is therefore
$\mathbb{R}^N \rtimes SO(N)$, of dimension $N + N(N-1)/2$ — three for the plane, six for space, ten for
$\mathbb{R}^4$ — and the objective is

$$
s^{\ast}(c, R) = \sup \lbrace s : s R S + c \subseteq F \rbrace , \qquad F = \lbrace g > 0 \rbrace .
$$

For a star-shaped $S$ with radial function $\rho_S$ this is not merely well-defined but computable in closed
form. Every point of a star-shaped body lies on a segment from its centre, so containment is containment of
segments:

$$
s^{\ast}(c, R) = \inf_{u} \frac{\lambda(c, Ru)}{\rho_S(u)} ,
$$

where $\lambda(c, v)$ is the ray clearance — the first exit of the ray from free space — obtained by sphere
tracing, which the 1-Lipschitz field already makes safe with no new machinery: step by $g(p)$ and never overshoot.

![Figure 6. The configuration-space lift. Panel (a) draws one slice of the lifted domain per orientation, with the largest admissible scale marked on each, and tabulates the dimension of the lift and the resulting branching factor for two, three and four ambient dimensions. Panel (b) plots the exact largest-scale objective against orientation for the region of Figure 5, showing a nearly flat landscape that rises sharply at a right angle, with its minimum coinciding with the orientation-blind placement. Panel (c) contrasts covering the shape by one ball, which reproduces the present rule, with covering it by eight, and states the resulting Lipschitz surrogate.](figures/fig6-configuration.svg)

**What breaks is the certificate, not the search.** The exact $s^{\ast}$ is discontinuous: a shape that just
slots through a gap becomes inadmissible under an infinitesimal move, so $s^{\ast}$ jumps and admits no honest
Lipschitz constant. It is therefore not a legal `Primitive`, and `sdf_geq_everywhere` does not apply to it. The
repair is to cover the shape by balls $B(y_i, r_i)$ and ask the field only at their centres:

$$
s_{\mathrm{lb}}(c, R) = \max \lbrace s : g(c + s R y_i) \ge s r_i \text{ for every } i \rbrace .
$$

Each constraint is 1-Lipschitz in $c$ and $s R_S$-Lipschitz in $R$, so $s_{\mathrm{lb}}$ carries an honest
constant; it is monotone in $s$, so bisection evaluates it; and it is sound, never claiming an infeasible
placement. The pleasing part is the degenerate case: **one ball at the centre with $r = R_S$ gives exactly
$g(c)/R_S$, which is the rule the crate uses today.** Refining the cover interpolates continuously from present
behaviour to the exact optimum, trading a tighter fit against a larger Lipschitz constant.

Two levels of ambition, then. The cheap one needs no library changes at all: keep the existing maxima search in
$\mathbb{R}^N$, then polish $(c, R)$ locally with `LineSearch` on the three- or six-dimensional
$s_{\mathrm{lb}}$, warm-started from the ball-inscribed pose. `LineSearch` is already dimension-generic and asks
only for a scalar function, so this is a drop-in that captures most of $\kappa^N$. The expensive one — storing
the lifted objective in an ADF — is harder than it looks: an insertion changes $g$ locally in $\mathbb{R}^N$ but
invalidates $s_{\mathrm{lb}}$ at *every* pose whose shape meets the change, a region
$\lVert c - x_0 \rVert \le s R_S + d$ crossed with all of $SO(N)$. The insertion-domain argument of §2 survives
the lift, but the domain is a cylinder rather than a box of side $4d$.

Two threads from earlier sections converge here, for entirely independent reasons. At $N = 3$ the lift is
six-dimensional, where a $2^D$ tree branches 64 ways — so the $k$-d refinement of Figure 3 stops being an
optimization and becomes a precondition. And the rotation coordinates must be weighted by $R_S$ to be
commensurate with the translations, since a rotation through $\delta$ moves a boundary point by about
$R_S \delta$ — the anisotropic axis weighting of §2.3, arriving from the geometry of poses rather than from
tractability theory.

Finally, two honest limits. The global objective — least empty space with fewest shapes — contains
two-dimensional bin packing and is NP-hard, so greedy-plus-polish is the right target and not optimality; and
the farthest-first 2-approximation of §4 covers the *point* problem, with no volume analogue to inherit.
Note also that $\kappa^N$ is a constant factor, not a better decay exponent, which makes it a prediction the
diagnostic of §2.1 can test directly: optimizing pose should leave the fitted slope of Figure 2 unchanged while
raising its intercept.

### 6.2 Packing trajectories (near-term, striking)

Let the ambient space be $C([0,T], \mathbb{R}^d)$ under the sup-norm. A "ball" is then a **tube** around a path,
and clearance is the guaranteed separation between world-lines. Greedy insertion becomes: *add the trajectory
that stays as far as possible from every trajectory already committed.* The 2-approximation of §4 applies
verbatim, so the result carries a quality guarantee, and non-intersection is certified rather than checked.

The practical route is unusually short. Represent a path by its B-spline coefficients $c \in \mathbb{R}^M$; the
sup-norm distance to a tube is 1-Lipschitz in $c$; therefore `Primitive::new` accepts it with the default
constant and every existing proof applies. **Multi-agent motion planning as a space-filling problem, with no new
mathematics** — only a change in what the coordinates mean.

### 6.3 Certified neural primitives (buildable today)

A neural implicit SDF trained with an Eikonal penalty (IGR) approximates $\lVert \nabla f \rVert = 1$ only
softly, so it is not admissible. But a network with spectral-normalized or Lipschitz-regularized layers carries a
*certified* upper bound $L$ on its gradient — and `Primitive::with_lipschitz(L)` accepts exactly that. The crate
needs no changes: a learned shape becomes a first-class citizen alongside a polytope, with pruning that remains
sound because the bound is honest. This is the same mechanism the Mandelbrot estimator in `06_custom_primitive`
already exploits, pointed at a different function class.

### 6.4 Diversity as space filling (research)

Maximin-distance design, sensor placement, and diverse sampling from a generative model are the same problem in a
latent space: choose points to maximize mutual separation. In an RKHS this is kernel herding with $O(1/k)$ rates
(§4). The `05_image_dataset` example already packs $10^5$ images *in the plane*; packing them in **feature
space** instead makes the field a novelty measure and the distribution a coverage-maximizing dataset.

### 6.5 Shape optimization on the existing representation (research)

Per §3: host a level-set optimizer whose state is an ADF, using the declared Lipschitz bound in place of periodic
reinitialization. This is the deepest of the four and the one most likely to produce something publishable.

### 6.6 Metric-space ADF (the structural generalization)

Note that the pruning inequality is *metric*: it needs $\lVert x - y \rVert$ and nothing else. The only Euclidean
ingredient in the whole structure is the box decomposition. Replace boxes with a hierarchical cover — a cover
tree or navigating net, with a node's radius playing the role of $h(R)$ — and the ADF generalizes to any metric
space of bounded doubling dimension: paths, images, molecules, strings. **The doubling dimension then replaces
$N$ everywhere in the cost analysis**, which is the correct notion of dimension for this problem and is finite in
many infinite-dimensional settings of interest.

---

## 7. What cannot be recovered

Stated plainly, so the proposals above are not mistaken for optimism.

1. **Exact global optimization.** Guaranteeing an $\varepsilon$-optimal global maximum of a Lipschitz function
   requires $\Theta((L/\varepsilon)^N)$ evaluations (Nemirovski–Yudin); in infinite dimension no finite bound
   exists. `Argmax2D` has no infinite-dimensional descendant, and no clever data structure changes that — it is
   an information-theoretic floor, not an implementation deficiency.
2. **The certificate on unweighted domains.** Without compactness there is no finite cover, and
   `sdf_geq_everywhere` remains sound but proves nothing below the root. Weights are not a convenience; they are
   the precondition.
3. **Volume intuition.** In high dimension "filling" ceases to mean what it means in the plane: the inscribed
   ball is a $10^{-15}$ fraction of the cube at $N = 32$. Any success criterion must be restated in terms of
   covering radius or entropy, never occupied volume.

---

## 8. Recommended order of work

![Figure 7. Recommended order of work, in three bands. Engineering: k-d splits, weight-ordered axis selection, anisotropic primitives — together raising the practical ceiling from N about 10 to N about 100 while touching none of the proofs. Demonstration: a randomized gradient in LineSearch, then trajectory packing in path space, the most persuasive result per unit of effort. Research: a cover-tree ADF for general metric spaces, and level-set shape optimization whose state is the ADF itself; these two require new proofs rather than new code alone.](figures/fig7-roadmap.svg)

Steps 1–2 are pure engineering with a large payoff: they move the practical ceiling from roughly $N = 10$ to
$N = 10^2$ and touch no proof. Step 5 is the most persuasive demonstration per unit of effort. Step 7 is the
research contribution.

One further item belongs between steps 3 and 4, and it is the only one that improves output quality rather than
reach: the pose polish of §6.1. It needs no new theory — `LineSearch` on $s_{\mathrm{lb}}$ over
$\mathbb{R}^N \rtimes SO(N)$, warm-started from the ball-inscribed pose — and it recovers up to $\kappa^N$ of the
volume every existing example currently discards. It pairs naturally with step 3, since anisotropic primitives
are exactly the shapes with $\kappa \gg 1$ and therefore the ones with most to gain.

The cheapest experiment of all is already done (§2.1, Figure 2): the slope of $\log d_k$ against $\log k$ recovers
$2.10$ for a known 2-dimensional point set, and $1.89$ for the distribution of example 01 — a non-integer
dimension below the ambient plane, which is the number that governs everything above. Worth running on any new
distribution before reasoning about it, and worth re-running after step 3, since anisotropic primitives should
lower the measured dimension further.

---

## References

**Prior art**
- P. Bourke, "Random space filling of the plane", 2011 —
  [paulbourke.net/fractals/randomtile](http://paulbourke.net/fractals/randomtile/).
- J. Shier, "A Million-Circle Fractal" —
  [d.umn.edu/~ddunham/circlepat.html](https://www.d.umn.edu/~ddunham/circlepat.html). *"…Run time was 14.7 hours."*
- S. Frisken, R. Perry, A. Rockwood, T. Jones, "Adaptively sampled distance fields: a general representation of
  shape for computer graphics", SIGGRAPH 2000, doi:[10.1145/344779.344899](https://dl.acm.org/doi/10.1145/344779.344899).

**Complexity and tractability**
- A. Nemirovski, D. Yudin, *Problem Complexity and Method Efficiency in Optimization*, 1983 — the
  $\varepsilon^{-N}$ floor for Lipschitz global optimization.
- E. Novak, H. Woźniakowski, *Tractability of Multivariate Problems*, 2008–2012.
- I. Sloan, H. Woźniakowski, "When are quasi-Monte Carlo algorithms efficient for high dimensional integrals?",
  *J. Complexity* 14 (1998) — weighted spaces.
- F. Kuo, C. Schwab, I. Sloan, "Quasi-Monte Carlo methods for high-dimensional integration", 2012 —
  infinite-dimensional QMC in practice.

**Greedy and conditional gradient**
- T. Gonzalez, "Clustering to minimize the maximum intercluster distance", *TCS* 38 (1985) — the
  2-approximation.
- Y. Chen, M. Welling, A. Smola, "Super-samples from kernel herding", UAI 2010.
- F. Bach, S. Lacoste-Julien, G. Obozinski, "On the equivalence between herding and conditional gradient",
  ICML 2012.

**Function-space optimization**
- J. Neuberger, *Sobolev Gradients and Differential Equations*, 2010.
- M. Delfour, J.-P. Zolésio, *Shapes and Geometries*, 2nd ed. 2011 — shape derivatives, Hadamard structure.
- G. Allaire, F. Jouve, A.-M. Toader, "Structural optimization using sensitivity analysis and a level-set
  method", *JCP* 194 (2004).
- S. Osher, J. Sethian, "Fronts propagating with curvature-dependent speed", *JCP* 79 (1988).
- Y. Nesterov, V. Spokoiny, "Random gradient-free minimization of convex functions", *FoCM* 17 (2017) — the
  dimension factor in derivative-free rates.
- F. Clarke, *Optimization and Nonsmooth Analysis*, 1983.

**Learned implicit fields**
- A. Gropp et al., "Implicit geometric regularization for learning shapes", ICML 2020 — the Eikonal penalty.
- H.-T. D. Liu et al., "Learning smooth neural functions via Lipschitz regularization", SIGGRAPH 2022 —
  certified bounds, the form this crate can accept.

**Measure-space flows**
- R. Jordan, D. Kinderlehrer, F. Otto, "The variational formulation of the Fokker–Planck equation", 1998.
- F. de Goes et al., "Blue noise through optimal transport", SIGGRAPH Asia 2012.
