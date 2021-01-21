#![feature(try_trait)]
#![feature(box_syntax)]
#![feature(type_ascription)]
#![feature(array_map)]
#![allow(dead_code)]

use std::{thread};
use lib::{Result, Point};
use quadtree::{sdf, Quadtree};
use std::convert::TryInto;
use crate::quadtree::sdf::{SDF, Circle};

mod quadtree;
mod drawing;
#[path = "util.rs"]
mod lib;

const WORLD_SIZE: f32 = 1024.0;
/// final image resolution is `WORLD_SIZE` * `IMG_SCALE`
const IMG_SCALE: f32 = 8.0;

fn main() -> Result<()> {
  thread::Builder::new()
    .spawn(||{
      img_test_parallel(
        sdf_argmax_bruteforce_test()?
      )?;
      Ok(())
    })?
    .join()
    .unwrap()
}

/// single circle in the middle, draw tree layout
fn basic_test() -> Result<()> {
  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    10
  );
  profile!("tree", {
    let c = sdf::Circle {
        xy: Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
        r: WORLD_SIZE / 4.0 - 1e-4
    };
    tree.insert_sdf(&|sample| c.sdf(sample), c);
  });
  tree.print_stats();
  profile!("draw", { drawing::tree_test(&tree, "out.png".into())?; });
  open::that("out.png")?;
  Ok(())
}

/// 1'000'000 random non-intersecting circles
fn test_1000000() -> Result<()> {
  unimplemented!()
}

fn img_test() -> Result<()> {
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    9
  );

  let mut circles = vec![];
  profile!("tree", {
    for _ in 0..10 {
      //println!("{}", i);
      let rect = loop {
        let pt = Point { x: rng.gen_range(0.0..WORLD_SIZE), y: rng.gen_range(0.0..WORLD_SIZE)};
        //let pt = match tree.find_empty_pt(&mut rng)
        let path = tree.path_to_pt(pt);
        let node = path.last()?;
        if !node.data {
          break node.rect;
        }
      };
      //let rect = node.last()?.rect;
      let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(0.5) * rect.size / (3.0 + 1e-3)).max(0.5 - 1e-3);
      let delta = rect.size / 2.0 - r;
      let c = sdf::Circle {
        xy: Point {
          x: rng.gen_range(rect.center.x - delta..rect.center.x + delta),
          y: rng.gen_range(rect.center.y - delta..rect.center.y + delta)
        },
        r
      };
      tree.insert_sdf(&|sample| c.sdf(sample), c);
      circles.push(c);
    };
  });
  tree.print_stats();
  //profile!("draw", drawing::exec_img(circles, "out.png".into())?);
  //open::that("out.png")?;
  Ok(())
}

fn img_test_parallel(circles: Vec<Circle>) -> Result<()> {
  use image::ImageBuffer;
  use walkdir::{WalkDir, DirEntry};
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let files = WalkDir::new("H:\\Temp\\export\\bottle-fairy")
    .sort_by(|a, b| {
      let [a, b] = [a, b].map(|x| x.file_name().to_string_lossy().to_string());
      lexical_sort::natural_cmp(&b, &a) // reversed
    })
    .into_iter()
    .filter_map(std::result::Result::ok)
    .map(|file: DirEntry| file.path().to_owned())
    .filter(|file| {
      let f = file.to_string_lossy();
      f.ends_with(".png") || f.ends_with(".jpg")
    });

  let mut draw_data = circles
    .into_iter()
    .zip(files)
    .collect::<Vec<_>>();
  // will distribute the load between threads [statistically] evenly
  draw_data.shuffle(&mut rng);

  const NUM_THREADS: usize = 4;

  let draw_data_chunks = draw_data
    .chunks((draw_data.len() as f32 / NUM_THREADS as f32).ceil() as usize)
    .map(|chunk| chunk.to_vec())
    .collect::<Vec<_>>();

  assert_eq!(draw_data_chunks.len(), NUM_THREADS);

  let partial_buffers: [_; NUM_THREADS] = draw_data_chunks.into_iter().map(|chunk| {
    thread::spawn(move || {
      let mut framebuffer: ImageBuffer<image::Rgba<u8>, _> =
        ImageBuffer::new((WORLD_SIZE * IMG_SCALE) as u32, (WORLD_SIZE * IMG_SCALE) as u32);

      drawing::exec_img(chunk.into_iter(), &mut framebuffer).ok();

      framebuffer
    })
  }).collect::<Vec<_>>() // thread handles
    .into_iter()
    .map(|thread| thread.join().unwrap())
    .collect::<Vec<_>>() // image buffers
    .try_into().unwrap();

  let mut final_buffer = partial_buffers[0].clone();

  // merge partial buffers
  partial_buffers[1..]
    .iter()
    .for_each(|buffer|
      image::imageops::overlay(&mut final_buffer, buffer, 0, 0)
    );

  std::io::stdin().read_line(&mut String::new())?;

  final_buffer.save("out_parallel.png")?;
  open::that("out_parallel.png")?;

  Ok(())
}

fn expand_area_test() -> Result<()> {
  use rand::prelude::*;

  let mut tree = Quadtree::new(
    WORLD_SIZE,
    Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    10
  );
  let big_circle = sdf::Circle {
    xy: Point { x: WORLD_SIZE / 2.0, y: WORLD_SIZE / 2.0 },
    r: WORLD_SIZE / 4.0 - 1e-4
  };
  tree.insert_sdf(&|sample| big_circle.sdf(sample), big_circle);

  /*let c = sdf::Circle {
    xy: Point { x: 165.0, y: 858.5 },
    r: 155.16904 - 1.0
  };
  tree.insert_sdf(&|sample| sdf::circle(sample, c), c);*/

  let path = format!("anim/#{:04}.png", 0);
  let img = drawing::tree_test(&tree, &path)?;
  drop(img);

  let mut rng = rand_pcg::Pcg64::seed_from_u64(2);

  for frame in 1..10 {
    //for _ in 1..16 {
      if let Some(pt) = tree.find_empty_pt(&mut rng) {
    //if let pt = { Point { x: 128.0, y: 128.0 } } {
        let path = tree.path_to_pt(pt);
        let (trbl, points) = path.get(path.len() - 2)?.find_max_free_area_attempt_7(pt)?;
        //let rect = tree.path_to_pt(pt).last()?.rect;
        //let trbl: sdf::TLBR = rect.into();
        let rect: sdf::Rect = trbl.into();
        println!("{:?}", rect);
        let c = sdf::Circle {
          xy: rect.center,
          r: rect.size / 2.0 - 1e-4
        };
        tree.insert_sdf(&|sample| c.sdf(sample), c);

        use plotters::prelude::*;
        let path = format!("anim/#{:04}.png", frame);
        let mut img = drawing::tree_test(&tree, &path)?;
        {
          img.draw_rect(
            (trbl.tl.x as i32, trbl.tl.y as i32),
            (trbl.br.x as i32, trbl.br.y as i32),
            &WHITE,
            false
          ).ok()?;
          img
            .draw_circle((pt.x as i32, pt.y as i32), 6, &RED, true)
            .ok()?;
          for (i, pt) in points.iter().enumerate() {
            img
              .draw_circle((pt.x as i32, pt.y as i32), 6, &[
                WHITE, // TL
                GREEN, // TR
                BLUE,  // BL
                YELLOW // BR
              ][i], true)
              .ok()?;
            //img.draw_line((0, pt.y as i32), (WORLD_SIZE as i32, pt.y as i32), &WHITE).ok()?;
            //img.draw_line((pt.x as i32, 0), (pt.x as i32, WORLD_SIZE as i32), &WHITE).ok()?;
          }

        };
      }
  }
  open::that("anim\\#0001.png")?;
  Ok(())
}

fn sdf_argmax_bruteforce_test() -> Result<Vec<Circle>> {
  use rayon::prelude::*;
  use rand::prelude::*;

  let mut rng = rand_pcg::Pcg64::seed_from_u64(0);

  let mut circles: Vec<Circle> = vec![];

  let mut max_dist_map = image::ImageBuffer::<image::Luma<f32>, _>::from_fn(
    WORLD_SIZE as u32,
    WORLD_SIZE as u32,
    |_, _| {
      image::Luma([f32::MAX / 2.0])
    });

  // insert boundary rect SDF
  max_dist_map
    .as_mut()
    .par_iter_mut()
    .enumerate()
    .for_each(|(i, pixel)| {
      let (x, y) = ((i % WORLD_SIZE as usize) as f32, (i / WORLD_SIZE as usize) as f32);
      let size = WORLD_SIZE;
      let sdf = sdf::Rect { center: Point {x: size / 2.0, y: size / 2.0}, size }
        .sdf(Point { x, y } );
      *pixel = pixel.min(sdf * -1.0)
    });

  profile! ("argmax", {
    'argmax: for i in 0..6153 {
      #[derive(Copy, Clone, Debug)]
      struct Argmax {
        distance: f32,
        point: Point
      }
      let argmax: Argmax = max_dist_map
        .as_ref()
        .par_iter()
        .enumerate()
        .map(|(i, distance)| {
          let (x, y) = ((i % WORLD_SIZE as usize) as f32, (i / WORLD_SIZE as usize) as f32);
          Argmax { distance: *distance, point: Point { x, y } }
        })
        .reduce(
          || Argmax { distance: -f32::MAX / 2.0, point: Point { x: 0.0, y: 0.0 } }, // identity element
          |a, b| if a.distance < b.distance { b } else { a }
        );

      const MIN_DISTANCE: f32 = 1.2;

      if argmax.distance < MIN_DISTANCE {
        println!("#{}: reached minimum, breaking: {:?}", i, argmax);
        break 'argmax;
      }

      let circle = {
        use std::f32::consts::PI;

        let angle = rng.gen_range::<f32, _>(-PI..=PI);
        let r = (rng.gen_range::<f32, _>(0.0..1.0).powf(0.5) * argmax.distance)
          .min(WORLD_SIZE / 6.0)
          .max(MIN_DISTANCE);
        let delta = argmax.distance - r;
        let offset = Point { x: delta * angle.sin(), y: delta * angle.cos() };

        sdf::Circle {
          xy: argmax.point.translate(offset), r
        }
      };

      if i % 100 == 0 { println!("argmax #{}", i); }

      max_dist_map
        .as_mut()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, pixel)| {
          let (x, y) = ((i % WORLD_SIZE as usize) as f32, (i / WORLD_SIZE as usize) as f32);
          let sdf = circle.sdf(Point { x: x as f32, y: y as f32 });
          *pixel = pixel.min(sdf)
        });
      circles.push(circle);

      /*exr::image::write::write_rgb_f32_file(
        format!("anim/#{:04}.exr", i),
        (WORLD_SIZE as usize, WORLD_SIZE as usize),
        |x, y| {
          let color = max_dist_map.get_pixel(x as u32, y as u32).0[0] / 128.0;
          let mut color = if color > 0.0 {
            (color.abs(), color.abs(), color.abs())
          } else {
            (color.abs(), 1.0 / 32.0, 1.0 / 32.0)
          };
          if {
            sdf::Circle { xy: argmax.point, r: 8.0 }
              .sdf(Point { x: x as f32, y: y as f32 }) < 0.0
          } {
            color = (0.0, 0.0, 1.0)
          }
          color
        })?;*/
    };
  });

  Ok(circles)
}