//! Wavefront OBJ export: every sphere becomes a UV-sphere mesh with smooth
//! per-vertex normals, in its own `o sphere_N` group — so a Blender import
//! can keep the spheres as separate objects, or merge them into one mesh.
//!
//! Tessellation adapts to size: the largest sphere gets the requested segment
//! count, smaller ones proportionally fewer (`∝ √(r/r_max)`, at least 8), so
//! dust-sized spheres do not dominate the file.

use {
    anyhow::Result,
    space_filling::geometry::{Point, Vector},
    std::{fs::File, io::{BufWriter, Write}, path::Path, time::Instant},
};

type P3 = Point<f64, 3>;
type V3 = Vector<f64, 3>;

/// Write `spheres` as UV-sphere meshes; `segments` is the longitude count of
/// the largest sphere.
pub fn export(path: &Path, spheres: &[(P3, f64)], segments: u32) -> Result<()> {
    use std::f64::consts::{PI, TAU};

    let t0 = Instant::now();
    let r_max = spheres.iter().map(|s| s.1).fold(0.0, f64::max);
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "# space-filling embedded distribution: {} spheres", spheres.len())?;

    let mut v_base = 1usize; // OBJ indices are global and 1-based
    let mut total_faces = 0usize;
    for (i, &(c, r)) in spheres.iter().enumerate() {
        let seg = ((segments as f64 * (r / r_max).sqrt()) as u32).clamp(8, segments) as usize;
        let rings = (seg / 2).max(4);
        writeln!(w, "o sphere_{i}")?;

        // vertices and their outward normals: top pole, `rings - 1` interior
        // rings of `seg` vertices, bottom pole — normal index == vertex index
        let mut emit = |n: V3| -> std::io::Result<()> {
            let v = c + n * r;
            writeln!(w, "v {:.6} {:.6} {:.6}", v.x, v.y, v.z)?;
            writeln!(w, "vn {:.4} {:.4} {:.4}", n.x, n.y, n.z)
        };
        emit(V3::new(0.0, 0.0, 1.0))?;
        for ring in 1..rings {
            let (st, ct) = (PI * ring as f64 / rings as f64).sin_cos();
            for s in 0..seg {
                let (sp, cp) = (TAU * s as f64 / seg as f64).sin_cos();
                emit(V3::new(st * cp, st * sp, ct))?;
            }
        }
        emit(V3::new(0.0, 0.0, -1.0))?;

        // faces, wound counter-clockwise seen from outside
        let top = v_base;
        let ring = |j: usize, s: usize| v_base + 1 + j * seg + s % seg;
        let bottom = v_base + 1 + (rings - 1) * seg;
        let f3 = |w: &mut BufWriter<File>, a: usize, b: usize, c: usize|
            writeln!(w, "f {a}//{a} {b}//{b} {c}//{c}");
        for s in 0..seg {
            f3(&mut w, top, ring(0, s), ring(0, s + 1))?;
            f3(&mut w, bottom, ring(rings - 2, s + 1), ring(rings - 2, s))?;
        }
        for j in 0..rings - 2 {
            for s in 0..seg {
                let (a, b) = (ring(j, s), ring(j, s + 1));
                let (d, c2) = (ring(j + 1, s), ring(j + 1, s + 1));
                writeln!(w, "f {a}//{a} {d}//{d} {c2}//{c2} {b}//{b}")?;
            }
        }
        v_base = bottom + 1;
        total_faces += seg * rings;
    }
    w.flush()?;
    println!("wrote {} ({} spheres, {} vertices, ~{} faces) in {:?}",
        path.display(), spheres.len(), v_base - 1, total_faces, t0.elapsed());
    Ok(())
}
