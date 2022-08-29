use {
  crate::{
    solver::quadtree::{
      Quadtree, TraverseCommand
    },
    geometry::WorldSpace,
    sdf::SDF
  },
  std::rc::Rc,
  euclid::{Point2D, Rect}
};

#[cfg(test)] mod tests;

pub type ADF = Quadtree<Vec<Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>>>;

impl SDF<f64> for &[Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>] {
  fn sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    self.iter()
      .map(|f| f(pixel))
      .reduce(|a, b| if a <= b { a } else { b })
      .unwrap_or(f64::MAX / 2.0)
  }
}

impl ADF {

  // TODO: optimization is required
  /*
    Upon insertion of a new SDF primitive (`f`), this function tests whether it does
    change the distance field within a certain domain (remember that it is considered changed
    if and only if at least one point of `f` is lower than combined distance field (`g`) within
    a certain domain `D` (tree node).
    If no change is present, therefore updating `g` within the domain may be safely
    skipped. However, it is imperfect: the test is only performed within a static squared grid of
    25 control points, thus sometimes yielding incorrect result, and generally being very slow.

    Proposition. use gradient descent to find the maximal error value, and answer following questions:
    \begin{align*}
&|f(\overrightarrow{v})-g(\overrightarrow{v})| < error, \forall\, \overrightarrow{v}\epsilon \,\mathfrak{D} &(1)\\
&f(\overrightarrow{v}) < g(\overrightarrow{v}), \forall\, \overrightarrow{v}\epsilon \,\mathfrak{D} &(2)\\
&f(\overrightarrow{v}) > g(\overrightarrow{v}), \forall\, \overrightarrow{v}\epsilon \,\mathfrak{D} &(3)\\
&\exists \overrightarrow{v}\epsilon \,\mathfrak{D}: f(\overrightarrow{v}) < g(\overrightarrow{v}) &(4)
\end{align}

    But, how to properly specify the boundary condition of `D`?
   */
  fn error(
    sdf_1: &dyn Fn(Point2D<f64, WorldSpace>) -> f64,
    sdf_2: &dyn Fn(Point2D<f64, WorldSpace>) -> f64,
    rect: Rect<f64, WorldSpace>
  ) -> bool {
    let epsilon = 1e-9;

    let control_points = |rect: Rect<_, _>| {
      let p = [0.0, 1.0/4.0, 2.0/4.0, 3.0/4.0, 1.0];
      itertools::iproduct!(p, p).map(move |p| {
        rect.origin + rect.size.to_vector().component_mul(p.into())
      })
    };

    control_points(rect)
      .map(|point| {
        let ground_truth = sdf_1(point).min(sdf_2(point));
        (ground_truth - sdf_1(point)).abs()
      })
      .sum::<f64>() < epsilon
  }

  fn sdf_vec(&self) -> impl Fn(Point2D<f64, WorldSpace>) -> f64 + '_ {
    move |p| self.data.as_slice().sdf(p)
  }

  pub fn insert_sdf_domain(&mut self, domain: Rect<f64, WorldSpace>, f: Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>) -> bool {
    let mut change_exists = false;

    self.traverse_managed(&mut |node| {
      // no intersection with domain
      if !node.rect.intersects(&domain) {
        return TraverseCommand::Skip;
      }

      // not a leaf node
      if node.children.is_some() {
        return TraverseCommand::Ok;
      }

      // f(v) > g(v) forall v e D, no refinement is required
      if Self::error(&node.sdf_vec(), f.as_ref(), node.rect) {
        return TraverseCommand::Skip;
      }

      // f(v) <= g(v) forall v e D, a minor optimization
      if Self::error(f.as_ref(), &node.sdf_vec(), node.rect) {
        node.data = vec![f.clone()];
        change_exists = true;
        return TraverseCommand::Skip;
      };

      change_exists = true;
      const BUCKET_SIZE: usize = 2;

      // remove SDF primitives, that do not affect the field within `D`
      let prune = |data: &[Rc<dyn Fn(_) -> _>], rect| {
        let mut g = vec![];
        for (i, f) in data.iter().enumerate() {
          let sdf_old = |p|
            data.iter().enumerate()
              .filter_map(|(j, f)| if i != j {
                Some(f(p))
              } else { None })
              .fold(f64::MAX / 2.0, |a, b| a.min(b));
          // there exists v e D, such that f(v) < g(v)
          if !Self::error(&sdf_old, f.as_ref(), rect) {
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
        //if !Self::error(f.as_ref(), &node.sdf_vec(), node.rect) {
          node.data.push(f.clone());
        //}

      } else { // max bucket size is reached, subdivide

        let mut g = node.data.clone();
        g.push(f.clone());

        node.subdivide(|rect_ch| prune(&g, rect_ch))
          .as_deref_mut()
          .unwrap()
          .iter_mut()
          .for_each(|child| {
            child.insert_sdf_domain(domain, f.clone());
          });
      }
      TraverseCommand::Skip
    });

    change_exists
  }
}

impl SDF<f64> for ADF {
  fn sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    match self.pt_to_node(pixel) {
      Some(node) => node.sdf_vec()(pixel),
      None => self.sdf_vec()(pixel),
    }}}
