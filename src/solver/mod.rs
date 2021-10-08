pub mod argmax2d;
pub use argmax2d::Argmax2D;

pub mod gradient_descent;
pub use gradient_descent::GradientDescent;

pub mod quadtree;
pub mod adf;

pub mod z_order_storage;

use {
  euclid::Point2D,
  num_traits::Float
};

#[derive(Copy, Clone, Debug)]
pub struct DistPoint<D, P, Space> {
  pub distance: D,
  pub point: Point2D<P, Space>
}

impl<F: Float, P: Default, S> Default for DistPoint<F, P, S> {
  fn default() -> Self {
    Self {
      distance: F::max_value() / (F::one() + F::one()),
      point: Point2D::default()
    }
  }
}

impl<D: PartialEq, P, S> PartialEq for DistPoint<D, P, S> {
  fn eq(&self, other: &Self) -> bool {
    self.distance.eq(&other.distance)
  }
}

impl<D: PartialOrd, P, S> PartialOrd for DistPoint<D, P, S> {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    self.distance.partial_cmp(&other.distance)
  }
}

impl<D: PartialEq, P, S> Eq for DistPoint<D, P, S> {}

impl<P, S> std::cmp::Ord for DistPoint<f32, P, S> {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    // waiting for #![feature(total_cmp)]
    fn total_cmp(left: f32, right: f32) -> std::cmp::Ordering {
      let mut left = left.to_bits() as i32;
      let mut right = right.to_bits() as i32;
      left ^= (((left >> 31) as u32) >> 1) as i32;
      right ^= (((right >> 31) as u32) >> 1) as i32;

      left.cmp(&right)
    }
    total_cmp(self.distance, other.distance)
  }
}