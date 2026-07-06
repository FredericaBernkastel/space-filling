//! Verification harness for Scene 4 (`scene04_quadtree.py`): builds a real ADF
//! over the scene's exact shape set and prints the leaf layout in the scene's
//! field coordinates ([-2.6, 2.6]^2). The python mirror of `insert_where` must
//! produce the same tree (uniform scaling changes no `>=` decision).
//!
//!   cargo run --release --bin adf_layout

use {
  space_filling::{
    geometry::{Shape, Circle, Rect as RectShape, NGonC, Line, P2},
    sdf::{self, SDF},
    solver::{ADF, Primitive},
  },
  euclid::{Point2D, Rect, Size2D, Vector2D as V2},
  std::sync::Arc,
};

const B: f64 = 2.6;
const K: f64 = 1.0 / (2.0 * B); // field coords -> unit domain

fn u(x: f64, y: f64) -> V2<f64, space_filling::geometry::WorldSpace> {
  V2::new(0.5 + x * K, 0.5 + y * K)
}

fn main() {
  let mut adf = ADF::new(5, vec![Primitive::new(sdf::boundary_rect)])
    .with_prune_subdiv(8);
  let domain = Rect::from_size(Size2D::splat(1.0));
  let mut add = |f: Arc<dyn Fn(P2<f64>) -> f64 + Send + Sync>| {
    adf.insert_sdf_domain(domain, f);
  };

  // scene04's PRIMS, in ORDER (field coords scaled by K into the unit domain)
  add(Arc::new(|p| Circle.scale(0.62 * K).translate(u(-1.3, 1.0)).sdf(p))); // circle
  add(Arc::new(|p| RectShape { size: Point2D::new(1.0 * K, 1.0 * K) }
    .translate(u(1.25, 1.05)).sdf(p))); // box (half-extents 0.5)
  add(Arc::new(|p| NGonC::<3>.scale(0.62 * K).translate(u(1.3, -1.2)).sdf(p))); // tri
  add(Arc::new(|p| Circle.scale(0.52 * K).translate(u(-1.25, -1.15)).sdf(p))); // circle2
  add(Arc::new(|p| Line {
    a: u(-0.35, -0.15).to_point(),
    b: u(0.55, 0.45).to_point(),
    thickness: 2.0 * 0.13 * K, // fields.segment's th is a radius
  }.sdf(p))); // seg

  let mut leaves = vec![];
  adf.tree.visit_leaves(|_| false, |leaf| {
    let r = leaf.rect; // unit -> field coords: v / K - B
    let ax = r.origin.x / K - B;
    let ay = r.origin.y / K - B;
    let bx = (r.origin.x + r.size.width) / K - B;
    let by = (r.origin.y + r.size.height) / K - B;
    leaves.push((leaf.depth, ax, ay, bx, by, leaf.data.len()));
  });
  leaves.sort_by(|a, b| a.partial_cmp(b).unwrap());
  println!("{} leaves", leaves.len());
  for (d, ax, ay, bx, by, n) in leaves {
    println!("  d{d} ({ax:+.2},{ay:+.2})..({bx:+.2},{by:+.2})  |B|={n}");
  }
  println!("{adf:#?}");
}
