use crate::domain::entities::grid::Grid;

/// Interface for rendering the grid.
pub trait GridRenderer {
    fn render(&self, grid: &Grid) -> String;
}

/// A concrete adapter that renders the Grid as an ASCII string.
/// This fulfills the Clean Architecture requirement of decoupling the domain logic
/// from the presentation layer.
pub struct AsciiRenderer;

impl GridRenderer for AsciiRenderer {
    fn render(&self, grid: &Grid) -> String {
        let mut output = String::new();
        
        for y in 0..grid.depth() {
            // Top walls for this row
            for x in 0..grid.width() {
                let cell = grid.get(x, y).unwrap();
                output.push('+');
                if cell.walls[0] { // North
                    output.push_str("---");
                } else {
                    output.push_str("   ");
                }
            }
            output.push_str("+\n");
            
            // Side walls for this row
            for x in 0..grid.width() {
                let cell = grid.get(x, y).unwrap();
                if cell.walls[3] { // West
                    output.push('|');
                } else {
                    output.push(' ');
                }
                output.push_str("   "); // Inside of the room
            }
            // East wall of the last cell in the row
            let last_cell = grid.get(grid.width() - 1, y).unwrap();
            if last_cell.walls[1] { // East
                output.push('|');
            } else {
                output.push(' ');
            }
            output.push('\n');
        }
        
        // Bottom walls for the last row
        for x in 0..grid.width() {
            let cell = grid.get(x, grid.depth() - 1).unwrap();
            output.push('+');
            if cell.walls[2] { // South
                output.push_str("---");
            } else {
                output.push_str("   ");
            }
        }
        output.push_str("+\n");
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::grid::Grid;

    #[test]
    fn test_ascii_renderer() {
        let mut grid = Grid::new(2, 2);
        // Modify some walls to test rendering
        if let Some(cell) = grid.get_mut(0, 0) {
            cell.walls[1] = false; // Remove East wall of (0,0)
        }
        if let Some(cell) = grid.get_mut(1, 0) {
            cell.walls[3] = false; // Remove West wall of (1,0)
        }

        let renderer = AsciiRenderer;
        let output = renderer.render(&grid);
        
        // Very basic validation, actual visual check is better in integration tests
        assert!(output.contains("+---+---+"));
    }
}
