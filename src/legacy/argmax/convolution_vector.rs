use super::*;

impl Argmax {
  pub fn convolute_domain(&mut self, domain: [f32; 2]) {
    use rayon::prelude::*;

    let domain = [
      (domain[0].max(0.0) * self.convolution_vector.len() as f32) as u32,
      (domain[1].min(1.0) * self.convolution_vector.len() as f32) as u32
    ];

    let partial_ret = (domain[0]..domain[1])
      .into_par_iter()
      .map(|y|
        (0..self.dist_map.width())
          .into_iter()
          .map(|x| {
            ArgmaxResult {
              distance: self.dist_map.get_pixel(x, y)[0],
              point: (x, y).into()
            }
          })
          .max_by(|a, b| a.distance.total_cmp(&b.distance))
          .unwrap()
      )
      .collect::<Vec<_>>();

    self.convolution_vector.splice(
      domain[0] as usize..domain[1] as usize,
      partial_ret
    );
  }

  pub fn find_max_convolution(&self) -> ArgmaxResult<f32> {
    let result = self.convolution_vector
      .iter()
      .cloned()
      .max_by(|a, b| a.distance.total_cmp(&b.distance))
      .unwrap();
    ArgmaxResult {
      distance: result.distance,
      point:
        (result.point.into(): Point<f32>) /
          ((self.dist_map.dimensions().into(): Point<u32>).into(): Point<f32>)
    }
  }
}