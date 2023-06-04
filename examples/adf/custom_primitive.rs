use euclid::{Rect, Size2D};
use {
  space_filling::{
    error::Result,
    sdf::{self, SDF},
    solver::{adf::ADF, Argmax2D, gradient_descent::{GradientDescent, LineSearch, LineSearchConfig}},
    drawing::Draw,
    geometry::{WorldSpace, BoundingBox, Shape, Scale, Translation}
  },
  euclid::{Point2D, Vector2D as V2, Box2D},
  image::{RgbaImage, Luma, Pixel},
  num_complex::Complex,
  num_traits::Float,
  std::sync::Arc
};

#[derive(Debug, Copy, Clone)]
struct MandlelDE;

impl <T: Float> SDF<T> for MandlelDE {
  fn sdf(&self, pixel: Point2D<T, WorldSpace>) -> T {
    let c = Complex::new(pixel.x, pixel.y);
    let mut z = Complex::new(T::zero(), T::zero());
    let mut dz = Complex::new(T::one(), T::zero());
    let mut inside = true;
    for _ in 0..256 {
      let z_new = z.powi(2) + c;
      let dz_new = z.scale(T::from(2.0).unwrap()) * dz + T::one();
      z = z_new;
      dz = dz_new;
      if z.norm_sqr() > T::from(1e9).unwrap() {
        inside = false;
        break;
      }
    }
    let mut result = z.norm() * z.norm().ln() / dz.norm();
    if inside {
      result = -result - T::from(128.0).unwrap();
    }
    result
  }
}

impl<T: Float> BoundingBox<T> for MandlelDE {
  fn bounding_box(&self) -> Box2D<T, WorldSpace> {
    Box2D::new(
      Point2D::new(T::from(-2.5).unwrap(), T::from(-1.25).unwrap()),
      Point2D::new(T::from(0.5).unwrap(), T::from(1.25).unwrap())
    )}}

// scaled to a box [-1, 1]
fn mandel_de_norm<T: Float>() -> Scale<Translation<MandlelDE, T>, T> {
  MandlelDE
    .translate(V2::new(T::from(1.0).unwrap(), T::zero()))
    .scale(T::one() / T::from(1.5).unwrap())
}

fn main() -> Result<()> {
  let path = "out.png";
  let mut image = RgbaImage::new(1024, 1024);
  let mut representation = ADF::new(7, vec![Arc::new(sdf::boundary_rect)]);
  let main_de = mandel_de_norm()
    .translate(V2::new(0.4, 0.5))
    .scale(0.5);
  representation.insert_sdf_domain(
    Rect::from_size(Size2D::splat(1.0)),
    Arc::new(move |p| main_de.sdf(p)
    )
  );
  let mut grad = GradientDescent::<&mut ADF, _>::new(LineSearchConfig::default(), &mut representation);
  grad.iter().build()
    .filter_map(|(local_max, grad)| {
      let angle = {
        let mut adf = ADF::new(1, vec![Arc::new(move |p| main_de
          .sdf(p))]);
        let grad = GradientDescent::<&mut ADF, _>::new(LineSearchConfig::default(), &mut adf);
        grad.grad_f(local_max.point).angle_from_x_axis()
      };
      let primitive = mandel_de_norm()
        .rotate(angle)
        .translate(local_max.point.to_vector())
        .scale(local_max.distance / 4.0);
      grad.insert_sdf_domain(
        Argmax2D::domain_empirical(local_max.point, local_max.distance),
        move |p| primitive.sdf(p)
      ).then(|| primitive)
    })
    .enumerate()
    .take(20000)
    .for_each(|(i, _)| {
      if i % 1000 == 0 { println!("#{}", i); };
    });
  representation
    .texture(Luma([255]).to_rgba())
    .draw(&mut image);
  //grad.display_sdf(&mut image, 4.0, Some(32));
  //representation.draw_layout(&mut image);
  image.save(path)?;
  Ok(())
}