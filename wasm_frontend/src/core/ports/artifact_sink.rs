use crate::application::ports::ChunkDraw;
use crate::core::ports::reference_renderer::RenderedImage;
use std::path::Path; // Or a specific type that contains the mesh

pub trait ArtifactSinkPort {
    fn write_png(&self, path: &Path, image: &RenderedImage) -> Result<(), std::io::Error>;
    fn write_obj(
        &self,
        path: &Path,
        scene: &[crate::application::ports::SurfaceChunk],
    ) -> Result<(), std::io::Error>;
    // fn write_json<T: serde::Serialize>(&self, path: &Path, value: &T) -> Result<(), std::io::Error>;
}
