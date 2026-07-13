//! Volumetric export: the exact sphere-union SDF, sampled on a dense grid.
//!
//! File format (`.vol`): magic `SFVD`, three `u32` LE dimensions (x, y, z),
//! then `x·y·z` signed distances as `f32` LE, x-fastest; the grid covers
//! `[0, 1]³`, sampled at cell centers. Distances are exact near the surface
//! and clamped at [`FAR`] in the far field (far beyond the shader's cutoff).
//!
//! Rendered by `src/08_embedded_3d_vdb.py`, which converts the grid to
//! OpenVDB and builds the Blender scene; the VDB background must equal
//! [`FAR`], or the cube's boundary faces glow (see the note there).

use {
    anyhow::Result,
    rayon::prelude::*,
    space_filling::geometry::{Point, VectorExt},
    std::{fs::File, io::{BufWriter, Write}, path::Path, time::Instant},
};

type P3 = Point<f64, 3>;

/// Far-field clamp of the exported distances; the shader cuts off at +0.012.
pub const FAR: f64 = 0.12;

/// Sample the union SDF of `spheres` on a `res`³ grid and write it to `path`.
pub fn export(path: &Path, res: usize, spheres: &[(P3, f64)]) -> Result<()> {
    // Per z-slab, only spheres that can reach below `FAR` are tested.
    let t0 = Instant::now();
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
    println!("sampled {res}³ voxels: {:?}", t0.elapsed());

    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(b"SFVD")?;
    for dim in [res, res, res] {
        w.write_all(&(dim as u32).to_le_bytes())?;
    }
    for v in &data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    println!("wrote {} ({} MiB)", path.display(), (16 + data.len() * 4) >> 20);
    Ok(())
}
