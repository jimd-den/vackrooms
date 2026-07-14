use crate::application::ports::{POSITION_FIXED_SCALE, SurfaceChunk};
use crate::core::ports::artifact_sink::ArtifactSinkPort;
use crate::core::ports::reference_renderer::RenderedImage;
use std::fs;
use std::io::Write;
use std::path::Path;

pub struct FsArtifactSink;

impl ArtifactSinkPort for FsArtifactSink {
    fn write_png(&self, path: &Path, image: &RenderedImage) -> Result<(), std::io::Error> {
        // Encode raw RGBA bytes into a PNG file.
        image::save_buffer(
            path,
            &image.rgba,
            image.width,
            image.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    fn write_obj(&self, path: &Path, chunks: &[SurfaceChunk]) -> Result<(), std::io::Error> {
        let mut file = fs::File::create(path)?;

        writeln!(file, "o backrooms_scene")?;

        let mut vertex_offset = 1; // OBJ indices are 1-based

        for chunk in chunks {
            writeln!(file, "g chunk_{}_{}", chunk.key.0, chunk.key.1)?;

            for v in &chunk.mesh.vertices {
                let px = (v.position[0] as f32 / POSITION_FIXED_SCALE) + chunk.origin[0];
                let py = (v.position[1] as f32 / POSITION_FIXED_SCALE) + chunk.origin[1];
                let pz = (v.position[2] as f32 / POSITION_FIXED_SCALE) + chunk.origin[2];
                writeln!(file, "v {:.6} {:.6} {:.6}", px, py, pz)?;
            }

            let mut i = 0;
            while i + 2 < chunk.mesh.indices.len() {
                let i1 = (chunk.mesh.indices[i] as usize) + vertex_offset;
                let i2 = (chunk.mesh.indices[i + 1] as usize) + vertex_offset;
                let i3 = (chunk.mesh.indices[i + 2] as usize) + vertex_offset;
                writeln!(file, "f {} {} {}", i1, i2, i3)?;
                i += 3;
            }

            vertex_offset += chunk.mesh.vertices.len();
        }

        Ok(())
    }
}
