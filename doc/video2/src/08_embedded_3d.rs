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
//! cargo run --release --bin 08_embedded_3d -- [n_hosts] [n_embedded] [resolution] [out.vol]
//! blender --python src/08_volumetric_3d.py -- out.vol
//! ```
//!
//! File format (`.vol`): magic `SFVD`, three `u32` LE dimensions (x, y, z),
//! then `x·y·z` signed distances as `f32` LE, x-fastest; the grid covers
//! `[0, 1]³`, sampled at cell centers. Distances are exact near the surface
//! and clamped at `+0.12` in the far field (far beyond the shader's cutoff).

use {
    space_filling::{
        geometry::{Shape, Circle, Point, Vector, VectorExt},
        sdf::SDF,
        solver::{LineSearch, ADF, Primitive},
        util
    },
    anyhow::Result,
    rand::prelude::*,
    rayon::prelude::*,
    std::{fs::File, io::{BufWriter, Write}, sync::RwLock, time::Instant}
};

type P3 = Point<f64, 3>;
type V3 = Vector<f64, 3>;

/// Far-field clamp of the exported distances; the shader cuts off at +0.012.
const FAR: f64 = 0.12;

/// `min_i (|p − cᵢ| − rᵢ)` — the exact union SDF of a set of spheres.
fn sphere_union(spheres: &[(P3, f64)]) -> impl Fn(P3) -> f64 + Clone + Send + Sync + use<> {
    let spheres = spheres.to_vec();
    move |p| spheres.iter()
        .map(|&(c, r)| (p - c).length() - r)
        .fold(f64::MAX, f64::min)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let n_hosts: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(64);
    let n_embedded: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(4096);
    let res: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(256);
    let path = args.next().unwrap_or_else(|| "out.vol".to_string());

    let mut rng = rand_pcg::Pcg64::seed_from_u64(3);

    // ---- the domain: an irregular cluster — a union of balls drifting +x ----
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
            if hosts.len() >= n_hosts { break; }
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
            // sub-voxel spheres still block their spot, but are not exported
            if r >= 1.25 / res as f64 {
                spheres.push((local_max.point, r));
            }
            placed += 1;
            if placed % 500 == 0 { println!("#{placed}"); }
            if placed >= n_embedded { break; }
        }
    }
    let adf = representation.into_inner().unwrap();
    println!("generation 2: {} embedded ({} exported) in {:?}",
             placed, spheres.len(), t1.elapsed());
    println!("{adf:#?}");

    // ---- sample the exact sphere-union SDF on a res³ grid (cell centers) ----
    // The grid stores the distance to the *embedded generation only* — the
    // domain and the hosts are scaffolding, visible solely as the swarm shapes.
    // Per z-slab, only spheres that can reach below `FAR` are tested.
    let t2 = Instant::now();
    let mut data = vec![0f32; res * res * res];
    data.par_chunks_mut(res * res).enumerate().for_each(|(z, slab)| {
        let pz = (z as f64 + 0.5) / res as f64;
        let nearby: Vec<(P3, f64)> = spheres.iter()
            .filter(|&&(c, r)| (c.z - pz).abs() - r < FAR)
            .copied()
            .collect();
        for y in 0..res {
            for x in 0..res {
                let p = P3::new(
                    (x as f64 + 0.5) / res as f64,
                    (y as f64 + 0.5) / res as f64,
                    pz);
                let d = nearby.iter()
                    .map(|&(c, r)| (p - c).length() - r)
                    .fold(FAR, f64::min);
                slab[y * res + x] = d as f32;
            }
        }
    });
    println!("sampled {res}³ voxels: {:?}", t2.elapsed());

    // ---- export ----
    let mut w = BufWriter::new(File::create(&path)?);
    w.write_all(b"SFVD")?;
    for dim in [res, res, res] {
        w.write_all(&(dim as u32).to_le_bytes())?;
    }
    for v in &data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    println!("wrote {path} ({} MiB); now run:", (16 + data.len() * 4) >> 20);
    println!("  blender --python examples/gd_adf/08_volumetric_3d.py -- {path}");
    Ok(())
}
