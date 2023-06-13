use {
  crate::{
    solver::{
      line_search::LineSearch
    },
    geometry::{Shape, shapes, P2, WorldSpace, BoundingBox},
    sdf::SDF,
  },
  quadtree::{
    Quadtree, TraverseCommand
  },
  std::{
    sync::{
      Arc, atomic::{AtomicBool, Ordering}
    },
    fmt::{Debug, Formatter}
  },
  euclid::{Point2D, Box2D, Rect},
  num_traits::{Float, Signed}
};

#[cfg(test)] mod tests;
pub(crate) mod quadtree;

#[derive(Clone)]
pub struct ADF<Float> {
  pub tree: Quadtree<Vec<Arc<dyn Fn(P2<Float>) -> Float>>, Float>,
  /// Gradient Descent lattice density, N^2
  /// higher values improve precision
  ipm_gd_lattice_density: u32,
  ipm_line_config: LineSearch<Float>
}

unsafe impl<Float> Send for ADF<Float> {}
unsafe impl<Float> Sync for ADF<Float> {}

impl <_Float: Float> SDF<_Float> for &[Arc<dyn Fn(P2<_Float>) -> _Float>] {
  fn sdf(&self, pixel: P2<_Float>) -> _Float {
    self.iter()
      .map(|f| f(pixel))
      .reduce(|a, b| if a <= b { a } else { b })
      .unwrap_or(_Float::max_value() / (_Float::one() + _Float::one()))
  }
}

fn sdf_partialord<_Float: Float + Signed>(
  f: impl Fn(P2<_Float>) -> _Float,
  g: impl Fn(P2<_Float>) -> _Float,
  domain: Rect<_Float, WorldSpace>,
  lattice_density: u32,
  line_search: LineSearch<_Float>
) -> bool {
  let boundary_constraint = |v| shapes::Rect { size: domain.size.to_vector().to_point() }
    .translate(domain.center().to_vector())
    .sdf(v); // IPM boundary

  let control_points = |rect: Rect<_, _>| {
    let p = (0..lattice_density).map(move |x| _Float::from(x).unwrap() / _Float::from(lattice_density - 1).unwrap());
    itertools::iproduct!(p.clone(), p.clone())
      .map(move |p| rect.origin + rect.size.to_vector().component_mul(p.into()))
  };

  let test = |v| line_search.optimize_normal(
    |v| if domain.contains(v) { g(v) - f(v) } else { -boundary_constraint(v) },
    v
  );

  !match lattice_density {
    1 => test(domain.center()),
    2..=u32::MAX => control_points(domain).any(test),
    _ => panic!("Invalid perturbation grid density: {lattice_density}")
  }
}

impl <_Float: Float + Signed + Sync> ADF<_Float> {
  //move |p| self.tree.data.as_slice().sdf(p)
  pub fn new(max_depth: u8, init: Vec<Arc<dyn Fn(P2<_Float>) -> _Float>>) -> Self {
    Self {
      tree: Quadtree::new(max_depth, init),
      ipm_gd_lattice_density: 1,
      ipm_line_config: LineSearch::default()
    }
  }
  /// Controls precision of bucket primitive pruning
  pub fn with_gd_lattice_density(mut self, density: u32) -> Self {
    self.ipm_gd_lattice_density = density;
    self
  }
  /// Underlying GD settings for the interior point method
  pub fn with_ipm_line_config(mut self, line_config: LineSearch<_Float>) -> Self {
    self.ipm_line_config = line_config;
    self
  }
  /*
    Upon insertion of a new SDF primitive (`f`), this function tests whether it does
    change the distance field within a certain domain (remember that it is considered changed
    if and only if at least one point of `f` is lower than combined distance field (`g`) within
    a certain domain `D` (tree node).
    If no change is present, therefore updating `g` within the domain may be safely
    skipped. However, it is imperfect: the test is only performed within a static square grid of
    25 control points, thus sometimes yielding incorrect result, and generally being very slow.

    Proposition. Use gradient descent (see `sdf_partialord`) in order to pick the control points
    more carefully, and answer following questions:
    \begin{align*}
&f(\overrightarrow{v}) < g(\overrightarrow{v}), \forall\, \overrightarrow{v}\epsilon \,\mathfrak{D} &(1)\\
&f(\overrightarrow{v}) > g(\overrightarrow{v}), \forall\, \overrightarrow{v}\epsilon \,\mathfrak{D} &(2)\\
&\exists \overrightarrow{v}\epsilon \,\mathfrak{D}: f(\overrightarrow{v}) < g(\overrightarrow{v}) &(3)
\end{align}

    Proposition 2. Use interior point method in order to specify the boundary constraint of `D`

    Update: implemented in [adf::sdf_partialord]
   */

  /// f(v) > g(v) forall v e D
  #[deprecated] #[allow(unused)]
  fn higher_all(
    f: &(dyn Fn(Point2D<f64, WorldSpace>) -> f64),
    g: &(dyn Fn(Point2D<f64, WorldSpace>) -> f64),
    d: Rect<f64, WorldSpace>
  ) -> bool {
    let control_points = |rect: Rect<_, _>| {
      let n = 5;
      let p = (0..n).map(move |x| x as f64 / (n - 1) as f64);
      itertools::iproduct!(p.clone(), p.clone())
        .map(move |p| rect.origin + rect.size.to_vector().component_mul(p.into()))
    };

    !control_points(d)
      .any(|v| g(v) > f(v))
  }

  pub fn insert_sdf_domain(&mut self, domain: Rect<_Float, WorldSpace>, f: Arc<dyn Fn(P2<_Float>) -> _Float + Send + Sync>) -> bool {
    let change_exists = AtomicBool::new(false);

    self.tree.traverse_managed_parallel(|node| {
      // no intersection with domain
      if !node.rect.intersects(&domain) {
        return TraverseCommand::Skip;
      }

      // not a leaf node
      if node.children.is_some() {
        return TraverseCommand::Ok;
      }

      // f(v) > g(v) forall v e D, no refinement is required
      if sdf_partialord(
        f.as_ref(),
        |p| node.data.as_slice().sdf(p),
        node.rect,
        self.ipm_gd_lattice_density,
        self.ipm_line_config
      ) {
        return TraverseCommand::Skip;
      }

      // f(v) <= g(v) forall v e D, a minor optimization
      if sdf_partialord(
        |p| node.data.as_slice().sdf(p),
        f.as_ref(),
        node.rect,
        self.ipm_gd_lattice_density,
        self.ipm_line_config
      ) {
        node.data = vec![f.clone()];
        change_exists.store(true, Ordering::Relaxed);
        return TraverseCommand::Skip;
      };

      change_exists.store(true, Ordering::Relaxed);
      const BUCKET_SIZE: usize = 3;

      // remove SDF primitives, that do not affect the field within `D`
      let prune = |data: &[Arc<dyn Fn(P2<_Float>) -> _Float>], rect| {
        let mut g = vec![];
        for (i, f) in data.iter().enumerate() {
          let sdf_old = |p|
            data.iter().enumerate()
              .filter_map(|(j, f)| if i != j {
                Some(f(p))
              } else { None })
              .fold(_Float::max_value() / (_Float::one() + _Float::one()), |a, b| a.min(b));
          // there exists v e D, such that f(v) < g(v)
          if !sdf_partialord(
            f.as_ref(),
            sdf_old,
            rect,
            self.ipm_gd_lattice_density,
            self.ipm_line_config
          ) {
            g.push(f.clone())
          }
        };
        g
      };

      // max tree depth is reached, just append the primitive
      if node.depth == node.max_depth {

        node.data.push(f.clone());
        //node.data = prune(node.data.as_slice(), node.rect);

      } else if node.data.len() < BUCKET_SIZE {

        //node.data = prune(node.data.as_slice(), node.rect);
        //if !Self::higher_all(f.as_ref(), &node.sdf_vec(), node.rect) {
          node.data.push(f.clone());
        //}

      } else { // max bucket size is reached, subdivide

        let mut g = node.data.clone();
        g.push(f.clone());

        node.subdivide(|rect_ch| prune(g.as_slice(), rect_ch));
        /*node.subdivide(|rect_ch| prune(&g, rect_ch))
          .as_deref_mut()
          .unwrap()
          .into_iter()
          .for_each(|child| {
            child.insert_sdf_domain(domain, f.clone());
          });*/
      }
      //TODO: despite issuing Skip, newly divided node continues to be explored. Rework Traverse API
      TraverseCommand::Skip
    });

    change_exists.load(Ordering::SeqCst)
  }

  pub unsafe fn as_mut(&self) -> &mut Self {
    let ptr = self as *const _ as usize;
    &mut *(ptr as *const Self as *mut _)
  }
}

impl <_Float: Float> SDF<_Float> for ADF<_Float> {
  fn sdf(&self, pixel: P2<_Float>) -> _Float {
    match self.tree.pt_to_node(pixel) {
      Some(node) => node.data.as_slice().sdf(pixel),
      None => self.tree.data.as_slice().sdf(pixel),
    }}}

impl <_Float: Float> BoundingBox<_Float> for ADF<_Float> {
  fn bounding_box(&self) -> Box2D<_Float, WorldSpace> {
    Box2D::new(
      P2::splat(_Float::zero()),
      P2::splat(_Float::one())
    )}}

impl <_Float: Float> Debug for ADF<_Float> {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    use humansize::{FileSize, file_size_opts as options};

    let mut total_nodes = 0u64;
    let mut total_size = 0usize;
    let mut max_depth = 0u8;
    self.tree.traverse(&mut |node| {
      total_nodes += 1;
      total_size += std::mem::size_of::<Self>()
        + node.data.capacity() * std::mem::size_of::<Arc<dyn Fn(P2<f64>) -> f64>>();
      max_depth = (max_depth).max(node.depth);
      Ok(())
    }).ok();
    f.debug_struct("ADF")
      .field("total_nodes", &total_nodes)
      .field("max_depth", &max_depth)
      .field("size", &total_size.file_size(options::BINARY).unwrap())
      .finish()
  }
}
