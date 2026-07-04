### Scene 1
Liserotte here. The story begins with a following equation {visual: ../eq1.svg}. Solving it numerically is not a problem. Solving it in logarithmic time and linear memory is another task entirely. This brought me into some fascinating mathematics.

{visual: ../../readme.md, "Implementation" section, scrolling}  
This video introduces my paper alongside its reference implementation, which I work on for last 6 years - mostly due to lack of theoretical knowledge on my part.

But I want to begin from even earlier.   
{visual: https://paulbourke.net/fractals/randomtile, scrolling}  
In 2011, Paul Bourke published his ["Random space filling of the plane"](https://paulbourke.net/fractals/randomtile) in relation to statistical geometry. The work described an iterative algorithm for tiling E^n space with non-overlapping shapes. The visuals were stunning, but algorithm itself was far from the best.   

{visual: assets/bourke_zoom.png, zooming out}  
This is "[A Million-Circle Fractal](https://www.d.umn.edu/~ddunham/circlepat.html)", generation run time was stated as "14.7 hours". Neither did the algorithm allow for much control over resulting distribution. Can we do better?

### Scene 2
{visual: densely filled space with shapes. random locations are tried one by one at least 10 times, all rejected due to being inside some shape, marked in red. until one is found to be outside, marked in green}
The major issue is purely in random sampling of the space. As more and more shapes are added, there is a vanishingly small chance of finding an empty spot. In the limit, it takes an inbounded amount of iterations to insert one more shape.  

Let's consider a different approach: represent each shape with a signed distance function.  
{TODO: introduce the definition of SDF}
{visual: all distance functions of shapes from `shapes.rs` visualized in a grid, alongside their field notations}

{visual: combining all the shapes from the grid into one compound distance field}  
Next, combine all primitive SDFs into one compound distance field, using aggregate pointwise minimum of them all.

{visual: a single gradient ascent trajectory across a real SDF field}
Since by definition, the gradient of any SDF always points towards outside, we can build an iterative algorithm to travel across that gradient, naturally finding some local maxima outside of any shape. Name of that algorithm is "gradient ascent".  
Problem solved, right? Well, not quite. Computing the aggregate pointwise minimum of N SDFs take O(N) time. Filling the space with N shapes takes O(N^2) time. We've traded one limitation for another with similar time semantics.

### Scene 3
{visual: TODO, come up with a fitting visual}  
Consider a bitmap of size N*N. The SDF field is rasterized onto it, and each (x, y) pixel holds a single-precision value at this point. Scan every pixel one by one to find the global maximum of the field. Let's see whether it works.  
{visual: ../../examples/argmax2d/01_fractal_distribution.rs, iteratively add one more shape every frame }
Sampling now takes O(1) time theoretically, but increasing precision costs both quadrating time *and* quadratic memory. A 4096x4096 bitmap in this visual corresponds to average discretezation error of 2.44e-4 and consumes 64MB memory. This is a dead end.

### Scene 4
{visual: assets/Adaptively sampled distance fields.pdf, scrolling}  
A paper "*Adaptively Sampled Distance Fields* (2000)" (doi:10.1145/344779.344899) proposes reducing memory by approximating the field per node of a k-d tree, yet gives little guidance for a practical implementation beyond fitting a polynomial in each node. Precision would then be limited by chosen polynomial degree as well as the quality of regression algorithm.  
Current work takes a different route: each node (bucket) stores the primitives themselves (function pointers). {TODO: explain what is the purpose of splitting primitives into buckets in the first place, and how it achieves a logarithmic sampling time}  
Redundant primitives are eliminated from a bucket by a bounded search over the node.  
{TODO: introduce the definition of Lipschitz-continuity}  
{TODO: explain briefly implementation detail of quadtree, flat arena node storage}
{TODO: explain "Implementation" section from readme.md in full, as well as how they affect performance, memory, and soundness guarantees. Use same terse algebraic notation}  

### Scene 5
Unlike the bitmap approach, this representation is lossless, offers a 10–100× memory reduction, and continuously differentiable - making it possible reintroduce an iterative optimizer. For practical purposes, I use an adaptive gradient ascent, which makes GD-ADF a __local-maximum method__.  
{TODO: explain implementation of adaptive gradient ascent used, exactly like in readme.md, as well as why it is more performant than simple GD with exponential decay}
{TODO: explain, how full parallelism was achieved (`util::find_max_parallel`) without corrupting the field in the process}

### Scene 6
So, how long does it take to make "A Million-Circle Fractal"?  
{visual: ../../examples/gd_adf/02_random_distribution.rs, but with 1M circles, adf max depth = 10, gradually filling the space each frame}  
{visual: assets/1M.png (final render), zooming in, panning around}
49 seconds on a 4-core machine. Resulting ADF tree contains 113k nodes, totalling 76MiB 

### Scene 7
So far we only tried simple circle shapes. Let's try somethinng much more funky: implement a mandelbrot distance estimator, and fit 20k instances of it.   
{visual: single mandlebrod DE SDF, with a grid of gradient arrows}  
Can the current GD-ADF implementation handle it in reasonable time, if at all? Since mandlebrot estimator is not a true SDF (1-Lipschitz bound no longer applies), and moreover, the interior is clamped to 0 alltogether.  
{visual: ../../examples/gd_adf/06_custom_primitive.rs, gradually filling the space each frame}  
7 seconds, 4.74 MiB, no obvious errors.

### Scene 7
Limitations and future work.
1. Current implementation only supports 2D plane, but generalizing to N dimensions should be trivial. Neither quadree, SDFs, and GD optimizer hold any assumptions about dimensionality of the problem.
2. Only shape insertion is currently supported, no deletion or any movement. 
3. Basic drawing capabilities are provided, but it is adviced to use a third-party library. Implementation was intended to be compatible with any drawing API out of the box.

### Scene 8
This is all, I hope you've learnt something new. Excited to see what kind of art you will make, feel free to share!  
Rest of the video contains varied visuals from time of working on the project. Enjoy!