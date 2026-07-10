//! Ports (Dependency Inversion boundaries) owned by the application layer.
//!
//! The application layer *defines* these interfaces; the outer layers
//! (adapters / drivers) *implement* them. This keeps the frame loop and
//! streaming logic testable without a GPU or a browser.

use crate::application::collision::Aabb;

/// Chunk-local fixed-point scale used by [`PackedVertex::position`]. A 20 u
/// high-spec chunk occupies only 20,480 units, comfortably inside `u16`.
pub const POSITION_FIXED_SCALE: f32 = 1024.0;

/// Compact vertex consumed by the default surface renderer. Position stays
/// chunk-local so it has stable precision even far from the origin; normal,
/// material, baked light, and AO remain compact integer attributes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedVertex {
    pub position: [u16; 3],
    pub normal_axis: u8,
    pub material: u8,
    pub light: u8,
    pub ao: u8,
}

/// Indexed greedy-mesh payload for one chunk. The SVO payload remains
/// authoritative for collision, storage, and ray queries; this exists only
/// to make visible surfaces cheap to rasterize.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMeshPayload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    pub bounds: Aabb,
    pub lod: u8,
}

impl SurfaceMeshPayload {
    pub fn empty(lod: u8) -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            bounds: Aabb::new([0.0; 3], [0.0; 3]),
            lod,
        }
    }
}

/// Stable mesh identity, quantized from a chunk origin like the streaming
/// store's key. The renderer uses it to replace only refined/changed meshes.
pub type SurfaceChunkKey = (i64, i64);

/// Borrowed incremental surface update. Mesh bytes are uploaded immediately;
/// no extra clone of a chunk's geometry is needed in the frame loop.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceChunk<'a> {
    pub key: SurfaceChunkKey,
    pub origin: [f32; 3],
    pub mesh: &'a SurfaceMeshPayload,
}

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

/// Abstraction over the actual rasterizer back end.
/// Implemented by `drivers::webgl::WebGl2Renderer` in the browser and by
/// test doubles in native unit tests.
pub trait RendererPort {
    /// True for the indexed surface renderer. The engine then skips SVO-atlas
    /// upload work while preserving the SVO in the chunk payload for collision
    /// and debug/reference traversal.
    fn uses_surface_meshes(&self) -> bool {
        false
    }

    /// Incrementally uploads newly loaded or refined chunk meshes.
    fn upload_surfaces(&mut self, _chunks: &[SurfaceChunk<'_>]) {}

    /// Releases meshes for chunks that left the streaming footprint.
    fn remove_surfaces(&mut self, _keys: &[SurfaceChunkKey]) {}

    /// Drops all resident meshes (used when noclipping to another level).
    fn clear_surfaces(&mut self) {}

    /// Last asynchronous GPU timing sample, when the driver supports
    /// `EXT_disjoint_timer_query_webgl2`.
    fn gpu_frame_ms(&self) -> Option<f32> {
        None
    }

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
    /// Greedy surface mesh used by the default raster path.
    pub surface: SurfaceMeshPayload,
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
