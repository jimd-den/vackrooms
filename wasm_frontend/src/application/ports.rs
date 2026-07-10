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
    pub static_indirect: u8,
    pub ao: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LightSource {
    pub id: u64,
    pub position: [f32; 3],
    pub half_size: [f32; 2],
    pub color: [f32; 3],
    pub radius: f32,
    /// Authored luminous strength. Kept independent from radius so a large
    /// atrium light can be brighter without relying on a driver heuristic.
    pub intensity: f32,
    pub flicker_mode: u8,
    pub enabled: bool,
}

/// One error-selected, axis-aligned visible surface rectangle for the splat
/// renderer: a greedy-merged face at the chunk's current LOD, extent-capped
/// so one flat light/shadow sample never smears across a whole wall. 16
/// bytes, uploaded verbatim as per-instance vertex attributes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedFaceInstance {
    /// Face-center, chunk-local fixed point ([`POSITION_FIXED_SCALE`]).
    pub position: [u16; 3],
    /// Extents in voxel cells along the face's U axis (X for Y/Z faces,
    /// Y for X faces) and V axis (Z for Y/X faces, Y for Z faces).
    pub extent_u: u8,
    pub extent_v: u8,
    /// Same encoding as [`PackedVertex::normal_axis`].
    pub normal_axis: u8,
    pub material: u8,
    /// Scalar baked voxel light, 0–15.
    pub baked_light: u8,
    /// Directional face occlusion bit (0 or 1).
    pub ao: u8,
    /// Bit 0: emissive material.
    pub flags: u8,
    pub reserved: [u8; 3],
}

pub const FACE_INSTANCE_FLAG_EMISSIVE: u8 = 1;

/// Contiguous run of face instances inside one culling cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceCellRange {
    /// Cell coordinate, chunk-local, in units of [`FaceInstanceSet::cell_size`].
    pub cell: [u8; 3],
    pub offset: u32,
    pub count: u32,
}

/// Per-chunk face-instance page: instances sorted by culling cell so the
/// renderer can draw or skip contiguous ranges without per-frame rebuilds.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceInstanceSet {
    pub instances: Vec<PackedFaceInstance>,
    pub cells: Vec<FaceCellRange>,
    /// World size of one culling cell in units.
    pub cell_size: f32,
}

impl FaceInstanceSet {
    pub fn empty() -> Self {
        Self {
            instances: Vec::new(),
            cells: Vec::new(),
            cell_size: 1.0,
        }
    }
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
    pub light_volume: Vec<u8>,
    pub light_volume_size: [u32; 3],
    pub lights: Vec<LightSource>,
    /// Face-instance page for the splat renderer, derived from the same
    /// greedy quads as `vertices`, so both paths describe the same boundary.
    pub faces: FaceInstanceSet,
    /// World size of one voxel cell at this payload's LOD.
    pub voxel_scale: f32,
}

impl SurfaceMeshPayload {
    pub fn empty(lod: u8) -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            bounds: Aabb::new([0.0; 3], [0.0; 3]),
            lod,
            light_volume: Vec::new(),
            light_volume_size: [0, 0, 0],
            lights: Vec::new(),
            faces: FaceInstanceSet::empty(),
            voxel_scale: 1.0,
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

    /// Telemetry data from the CPU renderer fallback, if applicable.
    fn cpu_telemetry_string(&self) -> Option<String> {
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

/// One background chunk-load order, echoed back verbatim with its result so
/// the engine can reject stale work (wrong level, chunk no longer desired,
/// or already refined past this LOD).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkRequest {
    pub origin_x: f32,
    pub origin_z: f32,
    pub level: u32,
    pub lod: u8,
}

/// A finished background load.
#[derive(Debug)]
pub struct CompletedChunk {
    pub request: ChunkRequest,
    pub payload: ChunkPayload,
}

/// Abstraction over where chunks come from. The browser build implements
/// this with in-wasm procedural generation (`adapters::local_chunk_source`)
/// or a Web Worker pool (`drivers::worker_source`); a networked build could
/// implement it with HTTP fetches instead.
pub trait ChunkSourcePort {
    /// Loads the chunk at `origin` for the given Backrooms level
    /// (0 = backrooms, 34 = grassland; see the core's `level_generator`).
    ///
    /// `lod` selects the level of detail: 0 is full resolution and each
    /// step doubles the voxel size, costing ~1/8 as much to produce. Any
    /// LOD of a chunk covers the same world cube (`world_size` invariant),
    /// so payloads are interchangeable to the renderer.
    fn load(&self, origin_x: f32, origin_z: f32, level: u32, lod: u8) -> ChunkPayload;

    /// True when the source generates in the background. The engine then
    /// drives it with [`Self::request`]/[`Self::poll_completed`] instead of
    /// the blocking [`Self::load`], keeping the frame loop responsive.
    fn is_async(&self) -> bool {
        false
    }

    /// Queues a background load. Deduplication is the caller's concern.
    fn request(&mut self, _request: ChunkRequest) {}

    /// Takes every finished background load. Results may arrive in any
    /// order and may be stale; the caller validates against its own state.
    fn poll_completed(&mut self) -> Vec<CompletedChunk> {
        Vec::new()
    }
}
