use {
  super::*,
  crate::{
    geometry::{Circle, Square, VectorExt}
  },
  nalgebra::Rotation2,
  anyhow::Result,
  image::{Rgba, RgbaImage},
};

#[test] fn texture() -> Result<()> {
  let mut image = RgbaImage::new(128, 128);
  Circle
    .translate(V2::repeat(0.5))
    .scale(0.5)
    .texture(&image::open("doc/embedded.jpg")?)
    .draw(&mut image);
  image.save("test/test_texture.png")?;
  Ok(())
}

#[test] fn polymorphic_a() -> Result<()> {
  let mut image = RgbaImage::new(128, 128);
  let shapes: Vec<Box<dyn Draw<_, _>>> = vec![
    Box::new(Circle.translate(V2::repeat(0.25)).scale(0.25)),
    Box::new(Square.translate(V2::repeat(0.75)).scale(0.25))
  ];
  shapes.into_iter()
    .for_each(|shape| shape
      .rotate(Rotation2::new(45f64.to_radians()))
      .texture(|_| Luma([255u8]).to_rgba())
      .draw(&mut image)
    );
  image.save("test/test_polymorphic_a.png")?;
  Ok(())
}

#[test] fn polymorphic_b() -> Result<()> {
  let mut image = RgbaImage::new(128, 128);
  let shapes: Vec<Box<dyn Draw<_, _>>> = vec![
    Box::new(Circle
      .translate(V2::repeat(0.25))
      .scale(0.25)
      .texture(Luma([255u8]).to_rgba())),
    Box::new(Square
      .translate(V2::repeat(0.75))
      .scale(0.25)
      .texture(Luma([127u8]).to_rgba()))
  ];
  shapes.into_iter()
    .for_each(|shape| shape.draw(&mut image));
  image.save("test/test_polymorphic_b.png")?;
  Ok(())
}

#[test] fn texture_fn() -> Result<()> {
  let mut image = RgbaImage::new(128, 128);
  Circle
    .translate(V2::repeat(0.5))
    .scale(0.5)
    .texture(|pixel: P2<f64>| {
      let c = 1.0 - (pixel - P2::new(0.5, 0.5)).length() * 2.0;
      Rgba([(c * 255.0) as u8, 32, 128, 255])
    })
    .draw(&mut image);
  image.save("test/test_texture_fn.png")?;
  Ok(())
}
