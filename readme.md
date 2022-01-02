[<img alt="crates.io" src="https://img.shields.io/crates/v/space-filling.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/space-filling)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-space--filling-66c2a5?style=for-the-badge&labelColor=555555&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">](https://docs.rs/space-filling)

You can read this paper for introduction: 
[Paul Bourke - Random space filling of the plane (2011)](http://paulbourke.net/fractals/randomtile/).  
However, provided search algorithm for the next location is inefficient, 
and offers very limited control over the distribution.
In this work, i present a new solver over discrete signed distance field:   
![](doc/eq1.svg)  
Where **sdf<sub>n</sub>** are custom signed distance functions. Aggregate minima of which is stored in a bitmap. 
**c<sub>n+1</sub>** marks a point with highest value of the field, which then supplied to the next iteration
of the algorithm.

Currently, the solver is parallel, and highly generic.
Supported:
- Regular (fractal) distributions
- Random distributions
- Any shapes which can be represented with SDF: curves, regular polygons, 
  non-convex and disjoint areas, fractals or any sets with non-integer
  hausdorff dimension (as long as the distance can be approximated)

## Examples
You can run examples with following command:  
`cargo run --release --features "drawing" --example <example name> -- -C target-cpu=native`

[`examples/fractal_distribution`](examples/fractal_distribution.rs)  
Each subsequent circle is inserted at the maxima of distance field.  
![](doc/fractal_distribution.png)

[`examples/random_distribution`](examples/random_distribution.rs)  
Given `(xy, value)` of the maxima, a new random circle is inserted within a domain of radius `value` and center `xy`.     
![](doc/random_distribution.png)

[`examples/embedded`](examples/embedded.rs)   
A regular distribution embedded in a random one.
1. Insert a random distribution of circles;
1. Invert the distance field;
1. Insert a fractal distribution.

[`examples/polymorphic`](examples/polymorphic.rs)  
Showcasing:
- Dynamic dispatch interface;
- Random distribution of mixed shapes;
- Random color and texture fill style;
- Parallel generation and drawing.

![](doc/polymorphic.png)

[`examples/image_dataset`](examples/image_dataset.rs)  
Display over 100'000 images.  
Run with `cargo run --release --features "drawing" --example image_dataset -- "<image folder>" -C target-cpu=native`  
![](doc/image_dataset.gif)

## Past work
In `src/legacy` you can find numeruos algorithms which are worth re-exploring, including quadtree and GPU implementations. 

## Future work
[x] Add more sample SDFs, and generic draw trait  
[x] Extend to support precision below 2<sup>-16</sup> (gigapixel resolution)

A new algorithm is being developed in the separate branch, offering 10-100x memory reduction, as well as 
continuous field representation (as opposed to discrete).  
Based on the paper "Adaptively Sampled Distance Fields" (doi:[10.1145/344779.344899](http://dx.doi.org/10.1145/344779.344899)),
and my implementation of gradient descent with a custom convergence factor, as follows:
![](doc/eq2.svg)  
Where `D` is a control parameter, and `y` specifies the base exponential convergence rate.  
Further will be referenced as `GradientDescent<ADF>`, or to be more specific, 
`GradientDescent<Quadtree<Vec<Rc<dyn Fn(Point2D<f64>) -> f64>>>>`.  
Each node contains multiple signed distance functions. The actual value at a point is computed as 
minimum of all the functions. However, more theoretical research is required 
in order to speed up the algorithm. How to efficiently approximate following logical statements:
- ![](doc/eq3.svg)
- ![](doc/eq4.svg)

where `f` and `g` are arbitrary non-analytical functions.  
Additionally, in order to limit growth rate of the quadtree, 
it's possible to use "bin splitting" strategy.  
Alternatively, would storing polynomial approximations instead offer more advantages?  

Once above are done, I will use this library for my next project "Gallery of Babel".