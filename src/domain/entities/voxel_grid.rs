/// Represents a dense 3D grid of voxels.
/// This is the foundation for our "smaller voxels" requirement.
pub struct VoxelGrid {
    width: usize,
    height: usize,
    depth: usize,
    data: Vec<u8>,
    /// Scalar light level 0-15 (the max of the RGB channels), kept for
    /// presenters and bakes that only need brightness.
    light_data: Vec<u8>,
    /// Colored flood-fill light, 0-15 per channel.
    light_rgb: Vec<[u8; 3]>,
    pub runtime_lights: Vec<crate::domain::entities::architecture::RuntimeLight>,
    /// Semantic threshold planes emitted by the same architectural pass as
    /// geometry. The application consumes them to advance deterministic
    /// anomaly epochs; render/collision never invent a separate reality.
    pub traversal_gates: Vec<crate::domain::entities::anomaly::TraversalGate>,
    /// Real floor openings that relocate the player when entered.
    pub pit_hazards: Vec<crate::domain::entities::anomaly::PitHazard>,
    /// Consumable pickups placed by the same generation pass as their
    /// marker voxels. Consumption is reality state, not grid state.
    pub supply_items: Vec<crate::domain::entities::supplies::SupplyItem>,
    /// Physical doorways between Backrooms levels. The application flushes
    /// and re-streams the world when the player steps through one.
    pub level_exits: Vec<crate::domain::entities::anomaly::LevelExit>,
    /// Six neighbor-occupancy bits prepared by the geometry source. A set bit
    /// means that face touches solid matter; clearing it means the face is
    /// exposed to air. Streaming generation computes this from its halo, so
    /// chunk-boundary answers do not depend on which neighbor is resident.
    face_occlusion: Vec<u8>,
}

/// Neighbor-occupancy encoding shared by meshing, SVO serialization, and CPU
/// surface reconstruction. The order intentionally matches the historical
/// `FaceDirection::occlusion_bit` contract.
pub const FACE_OCCLUDED_POSITIVE_X: u8 = 1;
pub const FACE_OCCLUDED_NEGATIVE_X: u8 = 2;
pub const FACE_OCCLUDED_POSITIVE_Y: u8 = 4;
pub const FACE_OCCLUDED_NEGATIVE_Y: u8 = 8;
pub const FACE_OCCLUDED_POSITIVE_Z: u8 = 16;
pub const FACE_OCCLUDED_NEGATIVE_Z: u8 = 32;
pub const FACE_OCCLUSION_MASK: u8 = 0x3f;

pub const VOXEL_AIR: u8 = 0;
pub const VOXEL_WALL: u8 = 1;
pub const VOXEL_FLOOR: u8 = 2;
pub const VOXEL_CEILING: u8 = 3;
pub const VOXEL_LIGHT: u8 = 4;
pub const VOXEL_RED_WALL: u8 = 5;
/// Walkable ground cover (grassland level). Visual-only, like FLOOR.
pub const VOXEL_GRASS: u8 = 6;
/// Walkable shallow water (grassland lakes). Visual-only, like FLOOR.
pub const VOXEL_WATER: u8 = 7;
/// Tree trunk (grassland). Solid: blocks the player like WALL.
pub const VOXEL_TREE: u8 = 8;
pub const VOXEL_RED_LIGHT: u8 = 9;
/// Pale bone-tinted arch-room wall. Solid like WALL.
pub const VOXEL_PALE_WALL: u8 = 10;
/// Rough, damaged blackout-expanse wall. Solid like WALL.
pub const VOXEL_DAMAGED_WALL: u8 = 11;
/// Shallow, dry, desaturated pillar-expanse carpet. Walkable, like FLOOR.
pub const VOXEL_DRY_CARPET: u8 = 12;
/// Deep, wet arch-room carpet. Walkable, like FLOOR.
pub const VOXEL_DEEP_CARPET: u8 = 13;
/// Thick, sticky, discolored red-room carpet. Walkable, like FLOOR.
pub const VOXEL_STICKY_CARPET: u8 = 14;
/// Recessed basin holding dark ankle-deep fluid. Walkable, like FLOOR.
pub const VOXEL_FLUID: u8 = 15;
/// Tiny cool emergency glimmer fixture (blackout recovery skeleton).
/// Emissive like LIGHT, far dimmer and colder.
pub const VOXEL_GLIMMER: u8 = 16;
/// Bare structural concrete (Level 1 walls and pillars). Solid like WALL.
pub const VOXEL_CONCRETE_WALL: u8 = 17;
/// Institutional tile floor (Level 1). Walkable, like FLOOR.
pub const VOXEL_TILE_FLOOR: u8 = 18;
/// Poured concrete slab floor (Level 1 parking sectors). Walkable.
pub const VOXEL_CONCRETE_FLOOR: u8 = 19;
/// Wooden supply crate (Level 1 Gild sector). Solid: blocks like WALL.
pub const VOXEL_CRATE: u8 = 20;
/// Exposed ceiling pipe / rebar (Level 1). Solid, but authored overhead
/// or as thin obstructions the player walks around.
pub const VOXEL_PIPE: u8 = 21;
/// Metal fire-exit door panel. Non-solid: crossing it is a level transit,
/// so the panel must admit the player who pushes into it.
pub const VOXEL_METAL_DOOR: u8 = 22;
/// Almond water bottle marker. Non-solid, faintly emissive so bottles
/// glint in dim fabric; the pickup itself is the exported `SupplyItem`.
pub const VOXEL_ALMOND_WATER: u8 = 23;

/// Number of voxel material ids (the palette table length).
pub const VOXEL_MATERIAL_COUNT: usize = 24;

/// The one authoritative material palette, `0xRRGGBB` per voxel id. Every
/// consumer — octree bake, greedy-mesh debug colors, CPU splatter, and the
/// generated GLSL tables — derives from this table so a new material can
/// never render differently across paths.
pub const MATERIAL_COLORS: [u32; VOXEL_MATERIAL_COUNT] = [
    0x000000, // 0 air (never drawn)
    0xDDCC66, // 1 yellow wallpaper wall
    0x998811, // 2 office carpet floor
    0xCCCCCC, // 3 dropped ceiling
    0xFFF8D6, // 4 fluorescent panel
    0x880000, // 5 crimson / peeled red wall
    0x4F9A3D, // 6 grass
    0x3A6FB8, // 7 water
    0x6B4A2F, // 8 tree / dark pillar
    0xFF4433, // 9 red light
    0xD8D2C0, // 10 pale arch wall
    0x8A7F5C, // 11 rough damaged wall
    0xC2B76B, // 12 dry shallow carpet
    0x6B5E22, // 13 deep wet carpet
    0x7A4A26, // 14 sticky red-room carpet
    0x2E2A22, // 15 dark pooled fluid
    0x9FC4E8, // 16 cool emergency glimmer
    0x8F8D88, // 17 bare structural concrete
    0xBDBBB0, // 18 institutional tile floor
    0x6E6C66, // 19 poured concrete slab
    0x9C7B4A, // 20 wooden supply crate
    0x3E4348, // 21 exposed pipe / rebar
    0x4A5A6A, // 22 metal fire-exit door
    0xEDE6D0, // 23 almond water bottle
];

/// Palette lookup as normalized linear-ish RGB for shader tables.
pub fn material_color_f32(voxel: u8) -> [f32; 3] {
    let c = MATERIAL_COLORS
        .get(voxel as usize)
        .copied()
        .unwrap_or(0x000000);
    [
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
    ]
}

/// Materials that block the player and produce collision boxes. Everything
/// else is walkable or decorative.
pub const SOLID_MATERIALS: [u8; 8] = [
    VOXEL_WALL,
    VOXEL_TREE,
    VOXEL_RED_WALL,
    VOXEL_PALE_WALL,
    VOXEL_DAMAGED_WALL,
    VOXEL_CONCRETE_WALL,
    VOXEL_CRATE,
    VOXEL_PIPE,
];

/// Materials that emit light in the baked flood fill and render emissive.
pub const EMISSIVE_MATERIALS: [u8; 4] = [
    VOXEL_LIGHT,
    VOXEL_RED_LIGHT,
    VOXEL_GLIMMER,
    VOXEL_ALMOND_WATER,
];

/// Authored emitted-radiance scale shared by fixture extraction and every
/// renderer's visible-emission path.
pub const fn material_emission_strength(material: u8) -> Option<f32> {
    match material {
        VOXEL_LIGHT => Some(10.0),
        VOXEL_RED_LIGHT => Some(8.0),
        VOXEL_GLIMMER => Some(0.9),
        // A bottle is a glint, not a lamp: just enough to catch the eye.
        VOXEL_ALMOND_WATER => Some(0.35),
        _ => None,
    }
}

impl VoxelGrid {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let size = width * height * depth;
        Self {
            width,
            height,
            depth,
            data: vec![VOXEL_AIR; size],
            light_data: vec![0; size],
            light_rgb: vec![[0; 3]; size],
            runtime_lights: Vec::new(),
            traversal_gates: Vec::new(),
            pit_hazards: Vec::new(),
            supply_items: Vec::new(),
            level_exits: Vec::new(),
            face_occlusion: vec![0; size],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn depth(&self) -> usize {
        self.depth
    }

    fn index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some(y * (self.width * self.depth) + z * self.width + x)
        } else {
            None
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, voxel_type: u8) {
        if let Some(idx) = self.index(x, y, z) {
            self.data[idx] = voxel_type;
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.data[idx]
        } else {
            VOXEL_AIR
        }
    }

    /// Sets a neutral (white) light level: all three channels get `level`.
    pub fn set_light(&mut self, x: usize, y: usize, z: usize, level: u8) {
        self.set_light_rgb(x, y, z, [level; 3]);
    }

    pub fn get_light(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.light_data[idx]
        } else {
            0
        }
    }

    /// Sets colored light (0-15 per channel); the scalar level becomes the
    /// max of the channels.
    pub fn set_light_rgb(&mut self, x: usize, y: usize, z: usize, rgb: [u8; 3]) {
        if let Some(idx) = self.index(x, y, z) {
            self.light_rgb[idx] = rgb;
            self.light_data[idx] = rgb[0].max(rgb[1]).max(rgb[2]);
        }
    }

    pub fn get_light_rgb(&self, x: usize, y: usize, z: usize) -> [u8; 3] {
        if let Some(idx) = self.index(x, y, z) {
            self.light_rgb[idx]
        } else {
            [0; 3]
        }
    }

    /// Stores the six-bit neighbor-occupancy mask for one voxel.
    pub fn set_face_occlusion(&mut self, x: usize, y: usize, z: usize, mask: u8) {
        if let Some(idx) = self.index(x, y, z) {
            self.face_occlusion[idx] = mask & FACE_OCCLUSION_MASK;
        }
    }

    pub fn get_face_occlusion(&self, x: usize, y: usize, z: usize) -> u8 {
        if let Some(idx) = self.index(x, y, z) {
            self.face_occlusion[idx]
        } else {
            0
        }
    }
}
