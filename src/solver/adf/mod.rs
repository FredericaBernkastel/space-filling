use {
  crate::{
    solver::quadtree::{
      Quadtree, TraverseCommand
    },
    geometry::WorldSpace,
    sdf::SDF
  },
  std::rc::Rc,
  euclid::{Point2D, Size2D, Rect}
};

#[cfg(test)] mod tests;

pub type ADF = Quadtree<Vec<Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>>>;

impl SDF<f64> for &[Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>] {
  fn sdf(&self, pixel: Point2D<f64, WorldSpace>) -> f64 {
    self.into_iter()
      .map(|f| f(pixel))
      .reduce(|a, b| if a <= b { a } else { b })
      .unwrap_or(f64::MAX / 2.0)
  }
}

impl ADF {
  pub fn prune(&mut self, domain: Rect<f64, WorldSpace>) {
    if self.rect.intersects(&domain) && self.children.is_some() {
      for child in self.children.as_deref_mut().unwrap() {
        child.prune(domain);
      }

      let children = self.children.as_deref_mut().unwrap();
      let data = self.data.as_slice();
      let rect = self.rect;

      if children.iter()
        // only leafs
        .all(|child| child.children.is_none())
      && children.iter()
        // equality of SDF functions
          .all(|child|
            Self::error(&child.sdf_vec(), &|p| data.sdf(p), rect)) {
        self.data = children[0].data.clone();
        self.children = None
      }
    }
  }

  fn error(
    sdf_1: &dyn Fn(Point2D<f64, WorldSpace>) -> f64,
    sdf_2: &dyn Fn(Point2D<f64, WorldSpace>) -> f64,
    rect: Rect<f64, WorldSpace>
  ) -> bool {
    let epsilon = 1e-6;

    let control_points = |rect: Rect<_, _>| {
      let p = [0.0/4.0, 1.0/4.0, 2.0/4.0, 3.0/4.0, 4.0/4.0];
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

  pub fn insert_sdf_domain(&mut self, domain: Rect<f64, WorldSpace>, sdf: Rc<dyn Fn(Point2D<f64, WorldSpace>) -> f64>) -> bool {
    let mut change_exists = false;

    self.traverse_managed(&mut |node| {
      // no intersection with domain
      if !node.rect.intersects(&domain) {
        return TraverseCommand::Skip;
      }

      if node.children.is_some() {
        return TraverseCommand::Ok;
      }

      // no refinement is required
      if Self::error(&node.sdf_vec(), sdf.as_ref(), node.rect) {
        return TraverseCommand::Skip;
      }

      // new minimum
      if Self::error(sdf.as_ref(), &node.sdf_vec(), node.rect) {
        node.data = vec![sdf.clone()];
        change_exists = true;
        return TraverseCommand::Ok;
      };

      change_exists = true;
      const BUCKET_SIZE: usize = 2;

      let prune = |data: &[Rc<dyn Fn(_) -> _>], rect| {
        let mut new_data = vec![];
        for (i, f) in data.iter().enumerate() {
          let sdf_old = |p|
            data.iter().enumerate()
              .filter_map(|(j, f)| if i != j {
                Some(f(p))
              } else { None })
              .fold(f64::MAX / 2.0, |a, b| a.min(b));
          if !Self::error(&sdf_old, f.as_ref(), rect) {
            new_data.push(f.clone())
          }
        };
        new_data
      };

      if node.depth == node.max_depth {

        node.data.push(sdf.clone());
        //node.data = prune(node.data.as_slice(), node.rect);

      } else {

        if node.data.len() < BUCKET_SIZE {

          let sdf_old = |p| prune(node.data.as_slice(), node.rect)
            .as_slice().sdf(p);
          if !Self::error(sdf.as_ref(), &sdf_old, node.rect) {
            node.data.push(sdf.clone());
          }

        } else {

          let mut data = node.data.clone();
          data.push(sdf.clone());

          node.subdivide(|rect_ch| prune(&data, rect_ch))
            .as_deref_mut()
            .unwrap()
            .iter_mut()
            .for_each(|child| {
              child.insert_sdf_domain(domain, sdf.clone());
            });
        }
      }
      return TraverseCommand::Skip;
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
