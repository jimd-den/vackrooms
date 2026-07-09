#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub enum MicrobiomeZone {
    Standard,
    Arch,
    PillarField,
    Holes,
    Blackout,
    RedRoom,
    Atrium,
}

impl Default for MicrobiomeZone {
    fn default() -> Self {
        Self::Standard
    }
}

/// Represents a single room or cell in the Backrooms maze.
/// We use simple booleans for walls (North, East, South, West) to adhere to KISS principle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub visited: bool,
    pub walls: [bool; 4], // [North, East, South, West]
    pub zone: MicrobiomeZone,
    pub light_level: u8,   // Pre-calculated baked light (0-15)
    pub is_corridor: bool, // Part 1: Hallways
}

impl Cell {
    /// Creates a new, unvisited cell with all walls intact and no anomalies.
    pub fn new() -> Self {
        Self {
            visited: false,
            walls: [true, true, true, true],
            zone: MicrobiomeZone::Standard,
            light_level: 0,
            is_corridor: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cell_has_all_walls_and_unvisited() {
        let cell = Cell::new();
        assert_eq!(cell.visited, false);
        assert_eq!(cell.walls, [true, true, true, true]);
    }
}
