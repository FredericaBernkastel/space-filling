//! This is a library for generalized space filling in ℝ² (do not mix with _packing_).
//!
//! It is split into two main modules: [`solver`] for generating a distribution of shapes,
//! and [`drawing`] for displaying it (requires `draw` feature).
//! Here, "shape" denotes one or multiple regions of ℝ² space, that can be represented by a
//! signed distance function.
//!
//! # Basic usage, Argmax2D solver
//! ```no_run
//! # use {
//! #   space_filling::{
//! #     geometry::{Shape, Circle, Scale, Translation},
//! #     sdf::{self, SDF},
//! #     solver::Argmax2D,
//! #     drawing::Draw,
//! #     util
//! #   },
//! #   anyhow::Result,
//! #   image::{Luma, Pixel}
//! # };
//! #
//! # fn main() -> Result<()> {
    //! let path = "out.png";
    //!
    //! /**
    //!   * Initialize instance of `Argmax2D`, which will store the discrete distance field in
    //!   * a 1024x1024 bitmap, and 16x16 chunk size.
    //!   * Resolution must be divisible by chunk size.
    //!   * Resolution affects the precision of solver, and is not related to final picture size.
    //!   * Chunk size is only important for optimization. Best values depend on the actual
    //!   * system configuration, but typically `chunk = resolution.sqrt() / 2`
    //!   **/
    //! let mut representation = Argmax2D::new(1024, 16)?;
    //! // prevent shapes from escaping image
    //! representation.insert_sdf(sdf::boundary_rect);
    //!
    //! // Initialize image buffer, which will hold final image.
    //! let mut image = image::RgbaImage::new(2048, 2048);
    //!
    //! // Generate the distribution of shapes:
    //! for _ in 0..1000 {
    //!   // find global maxima of the field
    //!   let global_max = representation.find_max();
    //!   // Make a new circle at the location with highest distance to all other circles.
    //!   let circle = Circle
    //!     .translate(global_max.point.to_vector())
    //!     .scale(global_max.distance / 4.0);
    //!   /** Update the field.
    //!    * `Circle` impletemens the `SDF` trait. Additionally, it has been concluded that
    //!    * only a certain part of the field is being changed every time, depending on the
    //!    * current maximum distance. To be exact - a square region with a side of
    //!    * max_dist * 4.0 * sqrt(2.0), which made possible achieving
    //!    * greater speed of computation. **/
    //!   representation.insert_sdf_domain(
    //!     util::domain_empirical(global_max),
    //!     |v| circle.sdf(v)
    //!   );
    //!   circle
    //!     .texture(Luma([255u8]).to_rgba()) // fill with white color
    //!     .draw(&mut image); // draw the shape on image
    //! }
    //! image.save(path)?; // save image
//! #   Ok(())
//! # }
//! ```
//! # GD-ADF solver
//! ```no_run
//! # use {
//! #   space_filling::{
//! #     geometry::{Shape, Circle, Translation, Scale, P2},
//! #     sdf::{self, SDF},
//! #     solver::{line_search::LineSearch, adf::ADF},
//! #     drawing::Draw,
//! #     util
//! #   },
//! #   image::{Luma, Pixel},
//! #   anyhow::Result,
//! #   rand::prelude::*,
//! #   std::sync::{Arc, RwLock}
//! # };
//! #
//! # fn main() -> Result<()> {
    //! let path = "out.png";
    //! let mut representation = RwLock::new(ADF::<f64>::new(5, vec![Arc::new(sdf::boundary_rect)]));
    //! let mut image = image::RgbaImage::new(2048, 2048);
    //! // In case of GD-ADF, it is adviced to use `util::local_maxima_iter`,
    //! // as it is capable of finding multiple local maxima in parallel.
    //! // By default, this is an infinite iterator.
    //! util::local_maxima_iter(
    //!   // provide a closure for sampling distance field
    //!   Box::new(|p| representation.read().unwrap().sdf(p)),
    //!   32, 0, LineSearch::default()
    //! ).filter_map(|local_max| {
    //!   let circle = Circle
    //!     .translate(local_max.point.to_vector())
    //!     .scale(local_max.distance / 4.0);
    //!   // Update distance field. Since the precision is not perfect, sometimes update may fail -
    //!   // thus Option is returned
    //!   representation.write().unwrap().insert_sdf_domain(
    //!     util::domain_empirical(local_max),
    //!     Arc::new(move |p| circle.sdf(p))
    //!   ).then(|| circle)
    //! }).take(1000) // stop, once 1000 circles were successfully added
    //!   .for_each(|shape| shape
    //!     .texture(Luma([255u8]).to_rgba())
    //!     .draw(&mut image)
    //!   );
    //!
    //! image.save(path)?;
  //! # Ok(())
//! # }
//!
//! ```
//! <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/fractal_distribution.png">
//!
//! # On dynamic dispatch and parallelism
//! There are three main traits related to drawing:
//! - `trait `[`Shape`](`geometry::Shape`)`: `[`SDF`](`sdf::SDF`)` + `[`BoundingBox`](`geometry::BoundingBox`)
//! - `trait `[`Draw`](`drawing::Draw`)`:`[`Shape`](`geometry::Shape`)`
//!
//! Draw is primarily implemented on [`Texture`](`drawing::Texture`):
//! ```text
//! .texture(Rgba(...)) -> Texture<T, Rgba<u8>>
//! .texture(image) -> Texture<T, image::DynamicImage>
//! .texture(|pixel| { ... }) -> Texture<T, Fn(Point2D) -> Rgba<u8>>
//! ```
//!
//! At first, you could think writing:
//! ```ignore
//! let shapes: Vec<Box<dyn Shape>> = vec![
//!   Box::new(Circle.translate(...).scale(...)),
//!   Box::new(Square.translate(...).scale(...))
//! ];
//! for shape in shapes {
//!   shape.texture(...)
//!     .draw(...);
//! }
//! ```
//! But this won't work, because all of `Shape` methods require `Sized`.
//! Correct way is:
//! ```ignore
//! let shapes: Vec<Box<dyn Draw<RgbaImage>>> = ...
//! ```
//!
//! Lastly, there are is: [`draw_parallel`](drawing::draw_parallel), that accept an iterator on
//! `dyn Draw<RgbaImage> + Send + Sync`. It is constructed via trait object casting, exactly as above.
//! See `examples/polymorphic.rs` and `drawing/tests::polymorphic_*` for more examples.
//!
//! This way, both distribution generation and drawing are guaranteed to evenly load all available
//! cores, as long as enough memory bandwidth is available.
//!
//! Have a good day, `nyaa~ =^_^=`
//!
//! <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/neko.gif">

#![cfg_attr(doc, feature(doc_cfg))]
#![allow(rustdoc::private_intra_doc_links)]

pub mod util;
pub mod sdf;
pub mod solver;
pub mod geometry;
#[cfg(feature = "drawing")]
#[cfg_attr(doc, doc(cfg(feature = "drawing")))]
pub mod drawing;