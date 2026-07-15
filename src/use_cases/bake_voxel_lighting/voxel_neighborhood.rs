use crate::domain::entities::voxel_grid::VoxelGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxelCoord {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) z: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GridDimensions {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
}

impl GridDimensions {
    pub(crate) fn of(grid: &VoxelGrid) -> Self {
        Self {
            width: grid.width(),
            height: grid.height(),
            depth: grid.depth(),
        }
    }

    pub(crate) fn volume(self) -> usize {
        self.width * self.height * self.depth
    }

    /// Matches `VoxelGrid`'s y-major, then z, then x storage order without
    /// exposing the entity's backing vectors to the use case.
    pub(crate) fn index(self, coord: VoxelCoord) -> usize {
        coord.y * (self.width * self.depth) + coord.z * self.width + coord.x
    }
}

pub(crate) fn for_each_voxel(dimensions: GridDimensions, mut visit: impl FnMut(VoxelCoord)) {
    for z in 0..dimensions.depth {
        for y in 0..dimensions.height {
            for x in 0..dimensions.width {
                visit(VoxelCoord { x, y, z });
            }
        }
    }
}

/// Visits the six face-sharing cells in a fixed order. The order makes the
/// implementation reproducible, while shortest integer distances make the
/// result independent of which equally short route was visited first.
pub(crate) fn for_each_face_neighbor(
    coord: VoxelCoord,
    dimensions: GridDimensions,
    mut visit: impl FnMut(VoxelCoord),
) {
    if coord.x + 1 < dimensions.width {
        visit(VoxelCoord {
            x: coord.x + 1,
            ..coord
        });
    }
    if coord.x > 0 {
        visit(VoxelCoord {
            x: coord.x - 1,
            ..coord
        });
    }
    if coord.y + 1 < dimensions.height {
        visit(VoxelCoord {
            y: coord.y + 1,
            ..coord
        });
    }
    if coord.y > 0 {
        visit(VoxelCoord {
            y: coord.y - 1,
            ..coord
        });
    }
    if coord.z + 1 < dimensions.depth {
        visit(VoxelCoord {
            z: coord.z + 1,
            ..coord
        });
    }
    if coord.z > 0 {
        visit(VoxelCoord {
            z: coord.z - 1,
            ..coord
        });
    }
}
