//! This is a library for generalized space filling in ℝ².
//!
//! It is split into two main modules: [`solver`] for generating a distribution of shapes,
//! and [`drawing`] for displaying it (requires `draw` feature).
//! Here, "shape" denotes one or multiple regions of ℝ² space, that can be represented by a
//! signed distance function.
//!
//! # Basic usage
//! ```no_run
//! # use {
//! #   space_filling::{
//! #     error::Result,
//! #     solver::Argmax2D,
//! #     geometry::{Shape, Circle, Scale, Translation},
//! #     drawing::Draw
//! #   },
//! #   image::{Luma, Pixel}
//! # };
//! # type AffineT<T> = Scale<Translation<T, f32>, f32>;
//! # fn distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = AffineT<Circle>> + '_ {
//! #   [].iter().cloned()
//! # }
//! # fn main() -> Result<()> {
    //! let path = "out.png";
    //!
    //! /**
    //!   * Initialize instance of `Argmax2D`, which will store the discrete distance field in
    //!   * a 1024x1024 bitmap, and 16x16 chunk size.
    //!   * Resolution must be divisible by chunk size.
    //!   * Resolution affects the precision of solver, and is not related to final picture size.
    //!   * Chunk size is only important for optimization. Best values depend on the actual
    //!   * system configuration, but typically (resolution -> chunk):
    //!   * - 1024 -> 16
    //!   * - 4096 -> 32
    //!   * - 16384 -> 64
    //!   **/
    //! let mut argmax = Argmax2D::new(1024, 16)?;
    //!
    //! // Initialize image buffer, which will hold final image.
    //! let mut image = image::RgbaImage::new(2048, 2048);
    //! // Generate the distribution of shapes:
    //! distribution(&mut argmax)
    //!   .take(1000)
    //!   .for_each(|shape| shape
    //!     .texture(Luma([255u8]).to_rgba()) // fill with white color
    //!     .draw(&mut image)); // draw the shape on image
    //! image.save(path)?; // save image
//! #   Ok(())
//! # }
//! ```
//! The distribution function can be defined as follows:
//! ```
//! # use space_filling::{
//! #   geometry::{Circle, Shape, Translation, Scale},
//! #   error::Result,
//! #   sdf::{self, SDF},
//! #   solver::Argmax2D,
//! # };
//! # use euclid::Vector2D;
//! // A set with an affine morphism on it
//! type AffineT<T> = Scale<Translation<T, f32>, f32>;
//!
//! fn distribution(argmax: &mut Argmax2D) -> impl Iterator<Item = AffineT<Circle>> + '_ {
//!   argmax.insert_sdf(sdf::boundary_rect); // all shapes must be *inside* the image
//!
//!   argmax.iter() // Returns an iterator builder. See `argmax2d::ArgmaxIter` for more options.
//!      // Finish build. By default, this is an infinite iterator, which will break
//!      // only after reaching a certain distance threshold (i.e. no more space).
//!     .build()
//!     .map(|(argmax_ret, argmax)| {
//!      /** Here, on each iteration, `argmax_ret` contains largest value of the
//!        * distance field, and its location;
//!        * `&mut Argmax2D` is passed here too. Using the one from outer scope is impossible,
//!        * because only one mutable reference must be active at any given time. **/
//!
//!       // Make a new circle at the location with highest distance to all other circles.
//!       let circle = Circle
//!         .translate(argmax_ret.point.to_vector())
//!         .scale(argmax_ret.distance / 4.0);
//!
//!       /** Update the field.
//!         * `Circle` impletemens the `SDF` trait. Additionally, it has been concluded that
//!         * only a certain part of the field is being changed every time, depending on the
//!         * current maximum distance. To be exact - a square region with a side of
//!         * max_dist * 4.0 * sqrt(2.0), which made possible achieving
//!         * greater speed of computation. **/
//!       argmax.insert_sdf_domain(
//!         Argmax2D::domain_empirical(argmax_ret.point, argmax_ret.distance),
//!         |pixel| circle.sdf(pixel)
//!       );
//!
//!       // Return the generated circle
//!       circle
//!     })
//! }
//! ```
//! <img src="https://raw.githubusercontent.com/FredericaBernkastel/space-filling/master/doc/fractal_distribution.png">
//!
//! # On dynamic dispatch and parallelism
//! There are three main traits related to drawing:
//! - `trait `[`Shape`](`geometry::Shape`)`: `[`SDF`](`sdf::SDF`)` + `[`BoundingBox`](`geometry::BoundingBox`)
//! - `trait `[`Draw`](`drawing::Draw`)`:`[`Shape`](`geometry::Shape`)`
//! - `trait `[`DrawSync`](`drawing::DrawSync`)`: `[`Draw`](`drawing::Draw`)` + Send + Sync`
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
//! Lastly, there are two functions: [`draw_parallel`](drawing::draw_parallel),
//! [`draw_parallel_unsafe`](drawing::draw_parallel_unsafe) that accept an iterator on
//! `dyn DrawSync<RgbaImage>`. It is constructed via trait object casting, exactly as above.
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

pub mod error;
pub mod sdf;
pub mod solver;
pub mod geometry;
#[cfg(feature = "drawing")]
#[cfg_attr(doc, doc(cfg(feature = "drawing")))]
pub mod drawing;