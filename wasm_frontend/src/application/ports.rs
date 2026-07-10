//! Ports (Dependency Inversion boundaries) owned by the application layer.
//!
//! The application layer *defines* these interfaces; the outer layers
//! (adapters / drivers) *implement* them. This keeps the frame loop and
//! streaming logic testable without a GPU or a browser.

use crate::application::collision::Aabb;

/// Per-chunk data the renderer needs to raymarch one chunk of the SVO atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkDraw {
    /// World-space origin (min corner) of the chunk's SVO cube.
    pub origin: [f32; 3],
    /// Root node index *within the merged atlas texture*.
    pub root_index: i32,
    /// Side length of the chunk's SVO cube in world units.
    pub world_size: f32,
}

/// Camera state for one frame, in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameParams {
    pub camera_pos: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub flashlight: bool,
}

/// Abstraction over the actual rasterizer/raymarcher back end.
/// Implemented by `drivers::webgl::WebGl2Renderer` in the browser and by
/// test doubles in native unit tests.
pub trait RendererPort {
    /// Uploads the merged SVO node atlas (RGBA32UI texel stream, 4 u32 per
    /// node, rows of 1024 texels — see `OctreeGpuSerializer` in the core).
    fn upload_atlas(&mut self, texels: &[u32]);

    /// Overwrites whole atlas rows starting at `first_row` (1024 texels per
    /// row) without reallocating or re-uploading the rest of the atlas.
    /// Returns `false` if the back end can't do partial updates (or has no
    /// atlas yet), in which case the caller must fall back to `upload_atlas`.
    fn upload_atlas_rows(&mut self, _first_row: u32, _texels: &[u32]) -> bool {
        false
    }

    /// Draws one frame: fullscreen raymarch of every chunk in `chunks`.
    fn draw(&mut self, frame: &FrameParams, chunks: &[ChunkDraw]);
}

/// A fully prepared chunk as the application consumes it.
#[derive(Debug, Clone)]
pub struct ChunkPayload {
    /// Root node index local to this chunk's `nodes` array.
    pub root: u32,
    /// GPU-serialized SVO nodes (row-padded, 4 u32 per node).
    pub nodes: Vec<u32>,
    /// Side length of the SVO cube in world units.
    pub world_size: f32,
    /// Solid-voxel bounding boxes in world space, for player collision.
    pub collision: Vec<Aabb>,
}

/// Abstraction over where chunks come from. The browser build implements
/// this with in-wasm procedural generation (`adapters::local_chunk_source`);
/// a networked build could implement it with HTTP fetches instead.
pub trait ChunkSourcePort {
    /// Loads the chunk at `origin` for the given Backrooms level
    /// (0 = backrooms, 34 = grassland; see the core's `level_generator`).
    ///
    /// `lod` selects the level of detail: 0 is full resolution and each
    /// step doubles the voxel size, costing ~1/8 as much to produce. Any
    /// LOD of a chunk covers the same world cube (`world_size` invariant),
    /// so payloads are interchangeable to the renderer.
    fn load(&self, origin_x: f32, origin_z: f32, level: u32, lod: u8) -> ChunkPayload;
}
