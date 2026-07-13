//! The scene of `03_embedded`, recreated with the ADF solver, in 3D: one
//! distribution embedded inside another. First a blob-shaped domain is filled
//! with host spheres; then the field is *inverted*, so its maxima lie deep
//! inside the hosts, and a second generation of spheres fills their interiors.
//! Only the second generation is exported — swarms of small spheres tracing
//! the invisible hosts.
//!
//! Where `Argmax2D` inverts its discrete bitmap in place, the ADF version
//! expresses the same trick through a primitive: the second field is seeded
//! with the *negated* union of the first generation, positive exactly inside
//! the hosts.
//!
//! ```text
//! cargo run --release --bin 08_embedded_3d -- --help
//! cargo run --release --bin 08_embedded_3d                      # out.vol
//! cargo run --release --bin 08_embedded_3d -- --format obj      # out.obj
//! blender --python src/08_embedded_3d_vdb.py -- out.vol
//! ```

mod format;

use {
    anyhow::Result,
    clap::{Parser, ValueEnum},
    rand::prelude::*,
    space_filling::{
        geometry::{Circle, Point, Shape, Vector, VectorExt},
        sdf::SDF,
        solver::{LineSearch, Primitive, ADF},
        util
    },
    std::{path::PathBuf, sync::RwLock, time::Instant}
};
use format::{obj, vol};

type P3 = Point<f64, 3>;
type V3 = Vector<f64, 3>;

#[derive(Copy, Clone, PartialEq, ValueEnum)]
enum Format {
    /// volumetric signed-distance grid (`SFVD`, for `src/08_embedded_3d_vdb.py`)
    Vol,
    /// Wavefront OBJ: one UV-sphere mesh per sphere, smooth normals
    Obj,
}

/// Two-level embedded space filling in 3D (the `03_embedded` scene, via ADF),
/// exported as a volumetric distance grid or an OBJ mesh.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// First-generation host spheres (never exported)
    #[arg(long, default_value_t = 64)]
    hosts: usize,
    /// Second-generation spheres, embedded inside the hosts
    #[arg(long, default_value_t = 4096)]
    embedded: usize,
    /// Output format
    #[arg(short, long, value_enum, default_value_t = Format::Vol)]
    format: Format,
    /// Output path [default: out.vol | out.obj, by format]
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Grid resolution per axis (vol)
    #[arg(long, default_value_t = 256)]
    resolution: usize,
    /// Longitude segments of the largest sphere mesh (obj)
    #[arg(long, default_value_t = 48)]
    segments: u32,
    /// Spheres smaller than this radius are placed, but not exported
    #[arg(long, default_value_t = 0.005)]
    min_radius: f64,
    /// RNG seed
    #[arg(long, default_value_t = 3)]
    seed: u64,
}

/// `min_i (|p − cᵢ| − rᵢ)` — the exact union SDF of a set of spheres.
fn sphere_union(spheres: &[(P3, f64)]) -> impl Fn(P3) -> f64 + Clone + Send + Sync + use<> {
    let spheres = spheres.to_vec();
    move |p| spheres.iter()
        .map(|&(c, r)| (p - c).length() - r)
        .fold(f64::MAX, f64::min)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = rand_pcg::Pcg64::seed_from_u64(args.seed);

    // ---- the domain ----
    let domain_balls: Vec<(P3, f64)> = {
        vec![(P3::new(0.5, 0.5, 0.5), 0.5)]
    };
    let cluster = sphere_union(&domain_balls);

    // ---- generation 1: host spheres fill the cluster ----
    let t0 = Instant::now();
    let representation = RwLock::new(ADF::<f64, 3>::new(
        6,
        // positive inside the cluster — the same role `boundary_rect` plays for
        // the unit square
        vec![Primitive::new(move |p| -cluster(p))],
    ));
    let mut hosts: Vec<(P3, f64)> = vec![];

    for local_max in util::local_maxima_iter(
        Box::new(|p: P3| representation.read().unwrap().sdf(p)),
        32, 0, LineSearch::default()
    ) {
        // uniform random direction on the unit sphere
        let dir = {
            use std::f64::consts::TAU;
            let z = rng.random_range(-1.0..1.0f64);
            let theta = rng.random_range(0.0..TAU);
            let xy = (1.0 - z * z).sqrt();
            V3::new(xy * theta.cos(), xy * theta.sin(), z)
        };
        let r = (rng.random_range(0.0..1.0f64) * local_max.distance)
            .min(1.0 / 4.0);
        // anywhere inside the free ball, pushed `delta` away from its center
        let center = local_max.point - dir * (local_max.distance - r);
        let sphere = Circle.translate(center.coords).scale(r);
        if representation.write().unwrap()
            .insert_at_maximum(local_max, Primitive::from_shape(sphere)) {
            hosts.push((center, r));
            if hosts.len() >= args.hosts { break; }
        }
    }
    println!("generation 1: {} hosts in {:?}", hosts.len(), t0.elapsed());

    // ---- generation 2: invert the field, fill the hosts' interiors ----
    // `Argmax2D::invert` negates its bitmap; here the negated union of the
    // hosts seeds a fresh field, positive exactly inside them. Its maxima are
    // the hosts' deepest interior points, so the second generation lands
    // embedded — largest hosts first.
    let t1 = Instant::now();
    let inverted = {
        let union = sphere_union(&hosts);
        move |p: P3| -union(p)
    };
    let representation = RwLock::new(ADF::<f64, 3>::new(
        7, vec![Primitive::new(inverted)]));
    let mut spheres: Vec<(P3, f64)> = vec![];
    let mut placed = 0usize;

    for local_max in util::local_maxima_iter(
        Box::new(|p: P3| representation.read().unwrap().sdf(p)),
        32, 0, LineSearch::default()
    ) {
        // the `03_embedded` rule: centered at the maximum, a third of the free ball
        let r = local_max.distance / 3.0;
        let sphere = Circle.translate(local_max.point.coords).scale(r);
        if representation.write().unwrap()
            .insert_at_maximum(local_max, Primitive::from_shape(sphere)) {
            // sub-threshold spheres still block their spot, but are not exported
            if r >= args.min_radius {
                spheres.push((local_max.point, r));
            }
            placed += 1;
            if placed % 500 == 0 { println!("#{placed}"); }
            if placed >= args.embedded { break; }
        }
    }
    let adf = representation.into_inner().unwrap();
    println!("generation 2: {} embedded ({} exported) in {:?}",
             placed, spheres.len(), t1.elapsed());
    println!("{adf:#?}");

    // ---- export the embedded generation only — the domain and the hosts are
    // scaffolding, visible solely as the swarm shapes ----
    let output = args.output.unwrap_or_else(|| match args.format {
        Format::Vol => PathBuf::from("out.vol"),
        Format::Obj => PathBuf::from("out.obj"),
    });
    match args.format {
        Format::Vol => {
            vol::export(&output, args.resolution, &spheres)?;
            println!("now run:\n  blender --python src/08_embedded_3d_vdb.py -- {}",
                output.display());
        }
        Format::Obj => obj::export(&output, &spheres, args.segments)?,
    }
    Ok(())
}
