/// Represents a single room or cell in the Backrooms maze.
/// We use simple booleans for walls (North, East, South, West) to adhere to KISS principle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell {
    pub visited: bool,
    pub walls: [bool; 4], // [North, East, South, West]
    pub is_red: bool,     // Rare anomaly: Red room
    pub is_dark: bool,    // Rare anomaly: Pitch-black room
    pub light_level: u8,  // Pre-calculated baked light (0-15)
}

impl Cell {
    /// Creates a new, unvisited cell with all walls intact and no anomalies.
    pub fn new() -> Self {
        Self {
            visited: false,
            walls: [true, true, true, true],
            is_red: false,
            is_dark: false,
            light_level: 0,
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
