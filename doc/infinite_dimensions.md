# Beyond $\mathbb{R}^N$: space filling in infinite-dimensional function spaces

*A design report. Speculative in its proposals, but every claim about what breaks is derived rather than
guessed.*

---

## 0. The short answer

Generalizing from $\mathbb{R}^2$ to $\mathbb{R}^N$ was a matter of removing an accident — nothing in the
pruning proof, the insertion bound or the ascent ever mentioned the number two. Going from $\mathbb{R}^N$ to a
function space is **not** the same kind of move. One of the two solvers dies, and it dies for an information-theoretic
reason that no amount of engineering repairs:

| | `Argmax2D` (exact global) | GD-ADF (local ascent) |
|---|---|---|
| what it needs | a $\delta$-cover of $\Omega$ | a descent direction |
| cost model | $\varepsilon^{-N}$ oracle calls | $L^2R^2/\varepsilon^2$ iterations |
| dimension in the exponent | **yes** | **no** |
| survives $N \to \infty$ | never | conditionally |

The Lipschitz certificate — this library's whole claim to soundness — is a **covering argument**, and covering
numbers are exactly what blow up. Gradient ascent, by contrast, has dimension-free rates in any Hilbert space,
which is precisely why function-space optimization is a mature field while function-space *global* optimization
is not.

But the negative result has a sharp condition attached, and the condition is the interesting part:

> **The obstruction is not infinite dimension. It is infinite dimension without an ordering of importance.**
> On a compact ellipsoid $\{x : \sum_i a_i x_i^2 \le 1\}$ with $a_i \to \infty$, covering numbers are finite,
> the maximum is attained, and the certificate returns — with cost polynomial in $1/\varepsilon$ rather than
> exponential in $N$. Infinitely many coordinates are affordable as long as they matter progressively less.

Everything below elaborates those two paragraphs, then proposes what to build.

---

## 1. Four different things "infinite-dimensional" can mean

The phrase hides at least four distinct programmes with very different prospects. They should not be conflated.

```mermaid
flowchart TD
    Q["Where does the infinite dimension live?"]
    Q --> A["A. The domain<br/>Omega is a ball in a Hilbert space"]
    Q --> B["B. The primitives<br/>each shape's boundary is a function"]
    Q --> C["C. The state<br/>the field itself is the variable"]
    Q --> D["D. The representation<br/>fields drawn from a parametric family"]

    A --> A1["packing in the Hilbert cube<br/>concentration of measure<br/>weighted tractability"]
    B --> B1["shape optimization<br/>shape derivatives, level-set flow"]
    C --> C1["greedy insertion as Frank-Wolfe<br/>monotone flow in a lattice"]
    D --> D1["neural implicit SDFs<br/>certified Lipschitz bounds"]

    A1 --> V["Verdict: hardest.<br/>Needs coordinate weights to be tractable at all"]
    B1 --> V2["Verdict: most immediately useful.<br/>Classical gradient methods apply"]
    C1 --> V3["Verdict: best explanatory value.<br/>Explains why greedy works"]
    D1 --> V4["Verdict: buildable today.<br/>The crate already accepts any L"]
```

---

## 2. Reading A — the domain becomes infinite-dimensional

Let $H$ be a separable Hilbert space, $\Omega \subset H$ the region to fill, and as before

$$
g_k(x) \;=\; \min_{n \le k} \mathrm{sdf}_n(x), \qquad
x_{k+1}^{*} \;=\; \arg\max_{x \in \Omega} g_k(x).
$$

### 2.1 What geometry does to you first

Before any algorithmic concern, the *problem statement* degrades. In $[0,1]^N$:

| $N$ | children per tree split, $2^N$ | volume of the inscribed ball, $r=\tfrac12$ | fraction of the cube within $0.05$ of $\partial\Omega$ | $\mathcal{N}(\Omega, \tfrac18)$ |
|---:|---:|---:|---:|---:|
| 2 | 4 | $7.9\times10^{-1}$ | 0.19 | $10^{2}$ |
| 3 | 8 | $5.2\times10^{-1}$ | 0.27 | $10^{3}$ |
| 4 | 16 | $3.1\times10^{-1}$ | 0.34 | $10^{4}$ |
| 8 | 256 | $1.6\times10^{-2}$ | 0.57 | $10^{7}$ |
| 10 | 1 024 | $2.5\times10^{-3}$ | 0.65 | $10^{9}$ |
| 20 | 1 048 576 | $2.5\times10^{-8}$ | 0.88 | $10^{18}$ |
| 32 | $4.3\times10^{9}$ | $1.0\times10^{-15}$ | 0.97 | $10^{29}$ |
| 64 | $1.8\times10^{19}$ | $1.7\times10^{-39}$ | 0.999 | $10^{58}$ |

Three separate deaths are visible in that table.

**The cube stops being round.** Its inradius is $\tfrac12$ for every $N$, but its circumradius is $\sqrt N/2$.
The inscribed ball occupies a vanishing fraction of the volume, so *filling with balls becomes meaningless*: the
distribution can achieve $10^{-15}$ coverage and still be optimal at every step. The right primitives in high
dimension are anisotropic — which the crate already has, in `ProductBall` and the `duocylinder`.

**Everything is boundary.** The fraction of $[0,1]^N$ within $\varepsilon$ of the boundary is
$1-(1-2\varepsilon)^N \to 1$. `boundary_rect` stops being a mild constraint and becomes the entire field.

**The maxima stop being informative.** Greedy farthest-point insertion has covering radius $d_k \asymp k^{-1/N}$,
so over a million insertions the clearance falls by $10^{-3}$ at $N=2$, by $0.25$ at $N=10$, and by $0.76$ at
$N=50$. The sequence flattens: every insertion is nearly as good as the first, which is the same as saying no
insertion accomplishes anything.

> **A free diagnostic.** Since $\log d_k \approx \text{const} - \tfrac{1}{N}\log k$, the slope of a log–log plot
> of observed clearance against insertion count *measures the effective dimension* of a distribution. This is
> testable today against the existing solvers, and would be a genuinely informative plot for the readme: a
> fractal distribution should show a non-integer slope.

### 2.2 Where the certificate breaks, exactly

The proof at the heart of `sdf_geq_everywhere` is

$$
\forall v \in R: \quad (f-g)(v) \;\ge\; (f-g)(c_R) - (L_f + L_g)\,h(R),
$$

refined by subdivision until the margin covers the box. To resolve a true margin of $\delta$ it must reach boxes
of half-diagonal $h < \delta/(L_f+L_g)$, and the number of such boxes needed to cover $\Omega$ is its covering
number, which for a norm ball in $\mathbb{R}^N$ satisfies

$$
\left(\frac{1}{\delta}\right)^{N} \;\le\; \mathcal{N}(\Omega,\delta) \;\le\; \left(\frac{3}{\delta}\right)^{N}.
$$

In infinite dimensions $\mathcal{N}(\Omega,\delta) = \infty$ for every $\delta$ less than the radius, because the
closed unit ball of an infinite-dimensional normed space is **not compact** (Riesz). Consequently:

- **Soundness survives.** The test still answers `true` only when the bound genuinely holds — the inequality
  above is metric and never referred to dimension.
- **Completeness collapses.** The test becomes vacuous: outside the lucky case where the root box already
  settles the question, no finite budget $k$ proves anything.
- **Attainment is no longer guaranteed.** $\mathrm{sdf}_n(x) = \lVert x-p_n\rVert - r_n$ is convex, so $g_k$ is a
  min of convex functions — neither convex nor concave, and only weakly *lower* semicontinuous. Maximizing it
  over a merely weakly-compact set has no existence theorem. The $\arg\max$ may simply not exist.

And well before any of that, the implementation hits a wall the mathematics doesn't care about: a $2^N$-tree has
$2^N$ children per node. At $N=10$ that is a thousand-way branch per subdivision; at $N=20$, a million. **The
tree, not the theory, is what fails first.**

### 2.3 The repair: weights, and the return of compactness

Replace the ball by an ellipsoid with summable-ish weights,

$$
\Omega_a \;=\; \Big\{ x \in \ell^2 \;:\; \sum_{i=1}^{\infty} a_i x_i^2 \le 1 \Big\},
\qquad a_i \to \infty ,
$$

which is the image of the unit ball under a compact diagonal operator and therefore **norm-compact**. Compactness
buys back everything at once: finite covering numbers, attained maxima, a meaningful branch-and-bound. And the
cost is no longer exponential in a dimension — it is polynomial in $1/\varepsilon$ with the exponent set by the
decay rate. For the Sobolev-type scaling $a_i \asymp i^{2s}$,

$$
\log \mathcal{N}(\Omega_a, \varepsilon) \;\asymp\; \varepsilon^{-1/s},
$$

so smoothness $s$ trades directly against certification cost:

| smoothness $s$ | $\log \mathcal N(10^{-2})$ |
|---:|---:|
| $0.5$ | $10^{4}$ |
| $1$ | $10^{2}$ |
| $2$ | $10$ |
| $4$ | $3.2$ |

This is the tractability theory of weighted spaces (Sloan–Woźniakowski) arriving in geometric clothing. Its
design consequence is concrete and immediate:

> Subdivide **one axis at a time, in weight order**, not all $N$ at once. A binary $k$-d split has branching
> factor 2 in every dimension; a full round of $N$ splits recovers one level of the $2^N$-tree at identical
> resolution and $N$ times fewer nodes than the fan-out costs. Cycling axes by descending weight $\gamma_j$ makes
> the refinement *dimension-adaptive*: cost tracks the effective dimension rather than the ambient one.

That single change is the difference between a usable ceiling of $N \approx 10$ and one of $N \approx 10^2$, and
it is a change to the tree, not to any proof — `sdf_geq_everywhere` is indifferent to how the boxes were made.

```mermaid
flowchart LR
    subgraph now["current: 2^N-tree"]
        P1["node"] --> C1["child 1"]
        P1 --> C2["child 2"]
        P1 --> C3["..."]
        P1 --> C4["child 2^N"]
    end
    subgraph next["proposed: weight-ordered k-d tree"]
        Q1["node, split axis j"] --> D1["x_j below mid"]
        Q1 --> D2["x_j above mid"]
        D1 --> E1["split axis j+1"]
        D2 --> E2["split axis j+1"]
    end
```

Where do the weights come from? Not from a hyperparameter — from the parameterization itself. A Karhunen–Loève
expansion of a random field supplies eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$ with
$\sum_i \lambda_i < \infty$ by construction. Which is the bridge to the second reading.

---

## 3. Reading B — the primitives become functions

Here the infinite dimension sits in the *shape*, not the ambient space, and this is where classical machinery
applies almost untouched.

Let a primitive be star-shaped with radial profile $\rho \in L^2(S^{N-1})$, or a tube around a path
$\gamma \in H^1([0,1],\mathbb{R}^N)$. The design variable is now a function; the objective is a functional. Two
classical results carry the weight:

**Hadamard's structure theorem.** For $J(\Omega) = \int_{\Omega} j$, the shape derivative in the direction of a
deformation field $V$ depends only on the *normal* trace of $V$ on the boundary:

$$
dJ(\Omega)[V] \;=\; \int_{\partial \Omega} j \,(V \cdot n)\, ds .
$$

Infinitely many degrees of freedom in $V$ collapse to one scalar field on $\partial\Omega$. This is the reason
shape optimization is possible at all, and it is a genuine dimension reduction rather than a discretization.

**The level-set method.** Advect the field itself by a Hamilton–Jacobi equation,

$$
\partial_t \phi + v\,\lVert \nabla \phi \rVert \;=\; 0 ,
$$

with $v$ read off the shape derivative. Note what this is: *gradient flow in function space, whose state variable
is an SDF*. This library already stores exactly that state. The classical difficulty of the method — $\phi$ drifts
away from being a distance function and must be periodically reinitialized — is, in this library's language, the
problem of maintaining the Lipschitz constant. The ADF's declared bound $L$ is a reinitialization-free surrogate:
a field that has drifted to $\lVert\nabla\phi\rVert \le L$ is still admissible, just with weaker pruning. **The
crate could host a level-set shape optimizer without changing its representation**, which strikes me as the single
most interesting unexplored direction here.

---

## 4. Reading C — the field is the variable, and greedy is a known algorithm

Take the state space to be

$$
\mathcal{L} \;=\; \{\, g : \Omega \to \mathbb{R} \;\mid\; g \text{ is } 1\text{-Lipschitz},\; g|_{\partial\Omega} = 0 \,\},
$$

a convex, closed, bounded subset of $C(\Omega)$ — infinite-dimensional, and closed under pointwise $\min$, hence
a **lattice**. Insertion is the operator $T_f : g \mapsto \min(g, f)$, and the algorithm is the orbit
$g_{k+1} = T_{f_k} g_k$.

Three observations follow for free, and they are more than reformulation:

**1. The algorithm is a monotone flow.** $g_{k+1} \le g_k$ pointwise, always. A decreasing sequence bounded below
converges pointwise to some $g_\infty$, and by equi-Lipschitzness plus Arzelà–Ascoli the convergence is uniform on
compacts. *The space-filling problem is the problem of characterizing which $g_\infty$ are reachable.* The
existing readme's remark that "insertions only ever lower the field" — used to justify the pre-subdivision
snapshot as a valid upper bound — is precisely monotonicity of $T_f$ on this lattice, and it is why the local
insertion-domain pruning is sound.

**2. Greedy insertion is Frank–Wolfe.** At each step the algorithm evaluates the linear functional
$g \mapsto \langle \delta_x, g\rangle = g(x)$ over all $x$, takes its maximizer, and moves the state toward the
extreme point that maximizer selects. That is the Frank–Wolfe / conditional-gradient template, whose $O(1/k)$
rate in Hilbert spaces is dimension-free. In an RKHS the same scheme is **kernel herding**. The greedy heuristic
is not a heuristic; it is conditional gradient on a lattice, and it inherits that literature's rates.

**3. The approximation guarantee is dimension-free.** Farthest-first traversal is a 2-approximation to $k$-center
in *any* metric space (Gonzalez, 1985). Greedy maximin insertion is exactly farthest-first traversal, so:

> The clearance achieved after $k$ greedy insertions is within a factor 2 of the best possible $k$-point
> configuration — in $\mathbb{R}^2$, in $\mathbb{R}^{100}$, in $L^2$, in path space, in any metric space
> whatsoever. The *combinatorial* guarantee survives infinite dimensions intact. Only the certificate does not.

That asymmetry is the cleanest summary of the whole report: what breaks is the *proof of local redundancy*, not
the *quality of the global strategy*.

```mermaid
flowchart TD
    S["greedy insertion in a metric space"]
    S --> G1["quality: 2-approx to k-center<br/>holds in ANY metric space"]
    S --> G2["certificate: Lipschitz branch and bound<br/>needs finite covering numbers"]
    G1 --> OK["survives dim = infinity"]
    G2 --> COND{"is Omega compact<br/>with summable weights?"}
    COND -->|yes| OK2["polynomial cost in 1/eps"]
    COND -->|no| BAD["vacuous: proves nothing<br/>below the root box"]
```

---

## 5. Can a classical gradient optimizer be applied in function space?

Yes — with three caveats, one of which this library has already solved by accident.

### 5.1 In Hilbert space: essentially unchanged

For $J : H \to \mathbb{R}$ Fréchet differentiable, the derivative is a functional $DJ(u) \in H^{*}$, and the Riesz
representation theorem supplies a unique $\nabla J(u) \in H$ with $DJ(u)v = \langle \nabla J(u), v\rangle$. The
iteration $u_{k+1} = u_k + h\,\nabla J(u_k)$ is then literally the same algorithm. Crucially, the standard rates
never mention dimension:

$$
J^{*} - J(u_k) \;\le\; \frac{2L\lVert u_0-u^{*}\rVert^2}{k+4} \quad (\text{smooth}),
\qquad
O\!\left(\frac{GR}{\sqrt{k}}\right) \quad (\text{nonsmooth subgradient}).
$$

Dimension-freeness is not a convenience here; it is the *entire reason* the infinite-dimensional case is viable
while global search is not.

### 5.2 In Banach space: the gradient stops being canonical

$C(\Omega)$ and $\mathrm{Lip}(\Omega)$ — the natural homes for a distance field — are not Hilbert. There is no
Riesz map, so a differential does not determine a descent *direction*; you must choose a metric first. The same
functional under an $L^2$ versus an $H^1$ inner product yields different flows, related by

$$
\nabla_{H^1} J \;=\; (I - \Delta)^{-1} \nabla_{L^2} J ,
$$

i.e. the Sobolev gradient is a smoothed version of the $L^2$ one (Neuberger). In shape optimization this is the
difference between a flow that develops oscillations and one that stays regular. **The choice of inner product is
not a detail of the analysis; it is the design of the algorithm.**

### 5.3 Non-smoothness: already handled here

The objective is a pointwise $\min$, hence non-differentiable, and the maxima being sought sit *exactly* on the
non-differentiable set — the medial axis. The correct object is the Clarke subdifferential, and the correct
algorithm is a monotone nonsmooth ascent.

Which the crate already implements. `LineSearch` uses only the gradient's **direction**, blends the previous
accepted direction to cancel the across-ridge zigzag, accepts only improving steps, and terminates on a vanishing
sampled gradient. Not one line of that argument uses finite dimension. Written abstractly:

$$
d_k = \frac{\nabla g(p_k)/\lVert \nabla g(p_k)\rVert + d_{k-1}}{\lVert \cdot \rVert},
\qquad
p_{k+1} = \begin{cases} p_k + h_k d_k, & g \text{ improves} \\ p_k, & \text{otherwise} \end{cases}
$$

is a Hilbert-space algorithm as it stands. **One thing must change**: the gradient is currently sampled by finite
differences along $N+1$ axis directions, which is meaningless when $N=\infty$. The replacement is a randomized
directional estimate over $m \ll N$ Gaussian directions,

$$
\widehat{\nabla} g(p) \;=\; \frac{1}{m}\sum_{i=1}^{m} \frac{g(p+\sigma \xi_i) - g(p)}{\sigma}\, \xi_i,
\qquad \xi_i \sim \mathcal{N}(0, C),
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

### 6.1 Packing trajectories (near-term, striking)

Let the ambient space be $C([0,T], \mathbb{R}^d)$ under the sup-norm. A "ball" is then a **tube** around a path,
and clearance is the guaranteed separation between world-lines. Greedy insertion becomes: *add the trajectory
that stays as far as possible from every trajectory already committed.* The 2-approximation of §4 applies
verbatim, so the result carries a quality guarantee, and non-intersection is certified rather than checked.

The practical route is unusually short. Represent a path by its B-spline coefficients $c \in \mathbb{R}^M$; the
sup-norm distance to a tube is 1-Lipschitz in $c$; therefore `Primitive::new` accepts it with the default
constant and every existing proof applies. **Multi-agent motion planning as a space-filling problem, with no new
mathematics** — only a change of what the coordinates mean.

### 6.2 Certified neural primitives (buildable today)

A neural implicit SDF trained with an Eikonal penalty (IGR) approximates $\lVert\nabla f\rVert = 1$ only softly,
so it is not admissible. But a network with spectral-normalized or Lipschitz-regularized layers carries a
*certified* upper bound $L$ on its gradient — and `Primitive::with_lipschitz(L)` accepts exactly that. The crate
needs no changes: a learned shape becomes a first-class citizen alongside a polytope, with pruning that remains
sound because the bound is honest. This is the same mechanism the Mandelbrot estimator in
`06_custom_primitive` already exploits, pointed at a different function class.

### 6.3 Diversity as space filling (research)

Maximin-distance design, sensor placement, and diverse sampling from a generative model are the same problem in a
latent space: choose points to maximize mutual separation. In an RKHS this is kernel herding with $O(1/k)$ rates
(§4). The `05_image_dataset` example already packs $10^5$ images *in the plane*; packing them in **feature space**
instead makes the field a novelty measure and the distribution a coverage-maximizing dataset.

### 6.4 Shape optimization on the existing representation (research)

Per §3: host a level-set optimizer whose state is an ADF, using the declared Lipschitz bound in place of periodic
reinitialization. This is the deepest of the four and the one most likely to produce something publishable.

### 6.5 Metric-space ADF (the structural generalization)

Note that the pruning inequality is *metric*: it needs $\lVert x - y\rVert$ and nothing else. The only Euclidean
ingredient in the whole structure is the box decomposition. Replace boxes with a hierarchical cover — a cover
tree or navigating net, with a node's radius playing the role of $h(R)$ — and the ADF generalizes to any metric
space with bounded doubling dimension: paths, images, molecules, strings. **The doubling dimension then replaces
$N$ everywhere in the cost analysis**, which is the correct notion of dimension for this problem and is finite in
many infinite-dimensional settings of interest.

---

## 7. What cannot be recovered

Stated plainly, so the proposals above are not mistaken for optimism:

1. **Exact global optimization.** Guaranteeing an $\varepsilon$-optimal global maximum of a Lipschitz function
   requires $\Theta((L/\varepsilon)^N)$ evaluations (Nemirovski–Yudin); in infinite dimension no finite bound
   exists. `Argmax2D` has no infinite-dimensional descendant, and no clever data structure changes that — it is
   an information-theoretic floor, not an implementation deficiency.
2. **The certificate on unweighted domains.** Without compactness there is no finite cover, and
   `sdf_geq_everywhere` remains sound but proves nothing below the root. Weights are not a convenience; they are
   the precondition.
3. **Volume intuition.** In high dimension "filling" ceases to mean what it means in the plane: the inscribed
   ball is a $10^{-15}$ fraction of the cube at $N=32$. Any success criterion must be restated in terms of
   covering radius or entropy, never occupied volume.

---

## 8. Recommended order of work

```mermaid
flowchart LR
    A["1. k-d splits<br/>replace 2^N fan-out"] --> B["2. weight-ordered<br/>axis selection"]
    B --> C["3. anisotropic primitives<br/>per-axis ProductBall radii"]
    C --> D["4. randomized gradient<br/>in LineSearch"]
    D --> E["5. trajectory packing demo<br/>path space, sup-norm"]
    E --> F["6. cover-tree ADF<br/>general metric spaces"]
    F --> G["7. level-set shape<br/>optimization on the ADF"]
```

Steps 1–2 are pure engineering with a large payoff: they move the practical ceiling from roughly $N=10$ to
$N=10^2$ and touch no proof. Step 5 is the most persuasive demonstration per unit of effort. Step 7 is the
research contribution.

A cheap first experiment, requiring no new code: plot $\log d_k$ against $\log k$ for the existing distributions
and read off the slope. If it is $-1/N$, the effective and ambient dimensions agree; if it is shallower, the
distribution is already living on a lower-dimensional structure — and that number is the one that governs
everything above.

---

## References

**Complexity and tractability**
- A. Nemirovski, D. Yudin, *Problem Complexity and Method Efficiency in Optimization*, 1983 — the
  $\varepsilon^{-N}$ floor for Lipschitz global optimization.
- E. Novak, H. Woźniakowski, *Tractability of Multivariate Problems*, 2008–2012.
- I. Sloan, H. Woźniakowski, "When are quasi-Monte Carlo algorithms efficient for high dimensional integrals?",
  *J. Complexity* 14 (1998) — weighted spaces.
- F. Kuo, C. Schwab, I. Sloan, "Quasi-Monte Carlo methods for high-dimensional integration", 2012 — infinite-dimensional QMC in practice.

**Greedy and conditional gradient**
- T. Gonzalez, "Clustering to minimize the maximum intercluster distance", *TCS* 38 (1985) — the 2-approximation.
- Y. Chen, M. Welling, A. Smola, "Super-samples from kernel herding", UAI 2010.
- F. Bach, S. Lacoste-Julien, G. Obozinski, "On the equivalence between herding and conditional gradient", ICML 2012.

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
