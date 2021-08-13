use space_filling::{
  geometry::Circle,
  error::Result,
  sdf::{self, SDF},
  argmax2d::Argmax2D,
  drawing
};

// 158ms, 1000 circles, Δ = 2^-10, chunk = 2^4
pub fn fractal_distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = Circle> + '_ {
  argmax.insert_sdf(sdf::boundary_rect);

  argmax.iter().build()
    .take(1000)
    .map(|(argmax_ret, argmax)| {
      let circle = Circle {
        xy: argmax_ret.point,
        r: argmax_ret.distance / 4.0
      };
      argmax.insert_sdf_domain(
        Argmax2D::domain_empirical(circle.xy, argmax_ret.distance).into(),
        |pixel| circle.sdf(pixel)
      );

      circle
    })
}

fn main() -> Result<()> {
  let path = "out.png";
  let mut argmax = Argmax2D::new(1024, 16)?;
  let circles = fractal_distribution(&mut argmax);
  drawing::draw_circles(
    circles,
    (2048, 2048).into()
  ).save(path)?;
  open::that(path)?;
  Ok(())
}