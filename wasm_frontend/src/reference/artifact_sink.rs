//! Explicit artifact-export boundary for reference-rendering tools.

use crate::application::ports::SurfaceChunk;
use crate::reference::renderer::RenderedImage;
use std::path::Path;

/// Explicit, opt-in artifact export. Regression tests compare semantics in
/// memory and never call this port, so an ordinary test run cannot rewrite a
/// checked-in baseline.
pub trait ArtifactSinkPort {
    fn write_png(&self, path: &Path, image: &RenderedImage) -> Result<(), std::io::Error>;
    fn write_obj(&self, path: &Path, scene: &[SurfaceChunk<'_>]) -> Result<(), std::io::Error>;
}
