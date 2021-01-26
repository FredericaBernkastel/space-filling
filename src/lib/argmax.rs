use {
  image::{ ImageBuffer, Luma },
  rayon::prelude::*,
  crate::{
    sdf::SDF,
    geometry::{Point, Circle, TLBR},
    error::Result,
    quadtree::Quadtree
  }
};

pub struct Argmax {
  pub dist_map: ImageBuffer::<Luma<f32>, Vec<f32>>,
  pub size: Point<u32>
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ArgmaxResult<T> {
  pub distance: f32,
  pub point: Point<T>
}

impl<T: Default> Default for ArgmaxResult<T> {
  fn default() -> Self {
    ArgmaxResult {
      distance: -f32::MAX / 2.0,
      point: Point { x: T::default(), y: T::default() }
    }
  }
}

impl Argmax {
  pub fn new(size: Point<u32>) -> Self {
    Argmax {
      dist_map: ImageBuffer::<Luma<f32>, _>::from_fn(
        size.x,
        size.y,
        |_, _| { Luma([f32::MAX / 2.0]) }
      ),
      size
    }
  }

  pub fn insert_sdf(&mut self, sdf: impl Fn(Point<f32>) -> f32 + Sync + Send) -> Result<()> {
    self.insert_sdf_domain(
      TLBR {
        tl: Point { x: 0.0, y: 0.0 },
        br: Point { x: 1.0, y: 1.0 }
      },
      sdf
    )
  }

  pub fn insert_sdf_domain(&mut self, domain: TLBR<f32>, sdf: impl Fn(Point<f32>) -> f32 + Sync + Send) -> Result<()> {
    if domain.br.x > 1.0 || domain.br.y > 1.0 {
      error_chain::bail!("domain is out of bounds");
    }

    let (width, height) = self.dist_map.dimensions();
    let dist_map_ptr = self.dist_map.as_mut_ptr() as usize;

    // cartesian product
    ((domain.tl.y * self.size.y as f32) as u32 .. (domain.br.y * self.size.y as f32) as u32)
      .into_par_iter()
      .flat_map(move |y|
        ((domain.tl.x * self.size.x as f32) as u32 .. (domain.br.x * self.size.x as f32) as u32)
          .into_par_iter().map(move |x| (x, y))
      )
      .for_each(move |(x, y)| {
        // don't look
        unsafe {
          let pixel = (dist_map_ptr as *mut f32).offset((y * width + x) as isize);
          *pixel = (*pixel).min(
            sdf(
              Point { x: x as f32 / width as f32, y: y as f32 / height as f32 }
            )
          )
        }
      });
    Ok(())
  }

  pub fn find_max(&self) -> ArgmaxResult<f32> {
    self.find_max_domain(
      TLBR {
        tl: Point { x: 0.0, y: 0.0 },
        br: Point { x: 1.0, y: 1.0 }
      }
    )
  }

  pub fn find_max_domain(&self, domain: TLBR<f32>) -> ArgmaxResult<f32> {
    // cartesian product
    let result = ((domain.tl.y * self.size.y as f32) as u32 .. (domain.br.y * self.size.y as f32) as u32)
      .into_par_iter()
      .flat_map(|y|
        ((domain.tl.x * self.size.x as f32) as u32 .. (domain.br.x * self.size.x as f32) as u32)
          .into_par_iter()
          .map(move |x| Point { x, y })
      )
      .map(move |xy| {
        ArgmaxResult { distance: self.dist_map.get_pixel(xy.x, xy.y)[0], point: xy }
      })
      .reduce(
        || ArgmaxResult::default(), // identity element
        |a, b| if a.distance < b.distance { b } else { a }
      );
    ArgmaxResult {
      point: Point{
        x: result.point.x as f32 / self.size.x as f32,
        y: result.point.y as f32 / self.size.y as f32
      },
      distance: result.distance
    }
  }

  pub fn display_debug(&self, output_file: &str, point: Option<Point<f32>>) -> Result<()> {
    let dimms = self.dist_map.dimensions();
    exr::image::write::write_rgb_f32_file(
      output_file,
      (dimms.0 as usize, dimms.1 as usize),
      |x, y| {
        let color = self.dist_map.get_pixel(x as u32, y as u32).0[0] * 4.0;
        let mut color = if color > 0.0 {
          (color.abs(), color.abs(), color.abs())
        } else {
          (color.abs(), 1.0 / 32.0, 1.0 / 32.0)
        };
        if let Some(point) = point {
          if { Circle {
            xy: Point {
              x: point.x * self.dist_map.width() as f32,
              y: point.y * self.dist_map.height() as f32
            },
            r: 8.0
          }}.sdf(Point { x: x as f32, y: y as f32 }) < 0.0 {
            color = (0.0, 0.0, 1.0)
          }
        }
        color
      })?;
    Ok(())
  }
}

impl Quadtree<ArgmaxResult<f32>> {

  pub fn argmax_backpropagation(&mut self, argmax: &Argmax) -> ArgmaxResult<f32> {
    use rayon::prelude::*;

    match (self.children.as_deref_mut(), self.is_inside) {
      (Some(children), false) => // backpropagate
        self.data = children
          .par_iter_mut()
          .map(|x| x.argmax_backpropagation(argmax))
          .max_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap())
          .unwrap(),
      (None, false) => // find argmax for this node
        self.data = argmax
          .find_max_domain(self.rect.into()),
      (_, true) => self.data = ArgmaxResult::default()
    }
    self.data
  }

}