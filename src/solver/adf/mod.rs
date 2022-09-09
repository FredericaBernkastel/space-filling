use {
  crate::{
    solver::{
      gradient_descent::{GradientDescent, LineSearch, LineSearchConfig},
      quadtree::{
        Quadtree, TraverseCommand
      }
    },
    geometry::{Shape, shapes, WorldSpace},
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

fn sdf_partialord(
  f: &(dyn Fn(Point2D<f64, WorldSpace>) -> f64),
  g: &(dyn Fn(Point2D<f64, WorldSpace>) -> f64),
  domain: Rect<f64, WorldSpace>
) -> Option<std::cmp::Ordering> {
  let boundary_constraint = |v| shapes::Rect { size: domain.size.to_vector().to_point() }
    .translate(domain.center().to_vector())
    .sdf(v);

  let config = LineSearchConfig {
    Δ: (-16f64).exp2(),
    initial_step_size: 0.5,
    decay_factor: 0.85,
    step_limit: Some(256),
    ..Default::default()
  };

  /*
  fn remove_dir_contents<P: AsRef<Path>>(path: P) -> std::io::Result<()> {
    for entry in std::fs::read_dir(path)? {
      std::fs::remove_file(entry?.path())?;
    }
    Ok(())
  }
  remove_dir_contents("test/test_grad").ok();
  GradientDescent::<&dyn Fn(_) -> _, _>::new(
    LineSearchConfig { control_factor: -1.0, ..config },
    &|v|
      if domain.contains(v) { f(v) } else { boundary_constraint(v) }
  ).trajectory_animation(
    &mut RgbaImage::new(512, 512),
    1.0,
    |i, mut img| {
      use {image::Pixel, crate::drawing::Draw};
      shapes::Rect { size: domain.size.to_vector().to_point() }
        .translate(domain.center().to_vector())
        .texture(Rgba::from([255,0,0,16]))
        .draw(&mut img);
      img.save(format!("test/test_grad/{}.png", i)).ok();
    }
  );*/

  let fmin = GradientDescent::<&dyn Fn(_) -> _, _>::new(
    LineSearchConfig { control_factor: -1.0, ..config },
    &|v| if domain.contains(v) { f(v) } else { boundary_constraint(v) }
  ).ascend_boundary(domain.center(), domain);

  let gmax = GradientDescent::<&dyn Fn(_) -> _, _>::new(
    LineSearchConfig { control_factor: 1.0, ..config},
    &|v| if domain.contains(v) { g(v) } else { -boundary_constraint(v) }
  ).ascend_boundary(domain.center(), domain);

  /*println!("domain: {domain:?}\nfmin:{fmin:?}\ngmax:{gmax:?}");
  let mut _t = "".to_string();
  std::io::stdin().read_line(&mut _t).unwrap();*/

  match () {
    _ if f(fmin) >= g(gmax) => Some(std::cmp::Ordering::Greater),
    _ => None
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
   */

  /// f(v) > g(v) forall v e D
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
      .any(|v| g(v) - f(v) > 0.0)
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
      if Self::higher_all(f.as_ref(), &node.sdf_vec(), node.rect) {
        return TraverseCommand::Skip;
      }
      /*if sdf_partialord(f.as_ref(), &node.sdf_vec(), node.rect) == Some(std::cmp::Ordering::Greater) {
        return TraverseCommand::Skip;
      }*/

      // f(v) <= g(v) forall v e D, a minor optimization
      if Self::higher_all(&node.sdf_vec(), f.as_ref(), node.rect) {
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
          if !Self::higher_all(f.as_ref(), &sdf_old, rect) {
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
