use super::cell::Cell;

/// Represents the overall 2D layout of the Backrooms level.
/// Contains a flat vector of cells mapped to 2D coordinates for O(1) access.
#[derive(Debug, Clone)]
pub struct Grid {
    width: usize,
    depth: usize,
    cells: Vec<Cell>,
}

impl Grid {
    /// Creates a new grid of dimensions width x depth, filled with default unvisited cells.
    /// Complexity: O(w * d) to allocate and initialize.
    pub fn new(width: usize, depth: usize) -> Self {
        let size = width * depth;
        let cells = vec![Cell::new(); size];
        
        Self {
            width,
            depth,
            cells,
        }
    }

    /// Returns the width of the grid.
    /// Complexity: O(1)
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the depth (length along the Z axis, or Y in 2D array terms) of the grid.
    /// Complexity: O(1)
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Retrieves an immutable reference to a cell at (x, y).
    /// Returns None if out of bounds.
    /// Complexity: O(1)
    pub fn get(&self, x: usize, y: usize) -> Option<&Cell> {
        if x >= self.width || y >= self.depth {
            None
        } else {
            Some(&self.cells[y * self.width + x])
        }
    }

    /// Retrieves a mutable reference to a cell at (x, y).
    /// Returns None if out of bounds.
    /// Complexity: O(1)
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Cell> {
        if x >= self.width || y >= self.depth {
            None
        } else {
            Some(&mut self.cells[y * self.width + x])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_initialization() {
        let width = 10;
        let depth = 10;
        let grid = Grid::new(width, depth);
        
        assert_eq!(grid.width(), width);
        assert_eq!(grid.depth(), depth);
        
        // Ensure all cells are initialized correctly
        for y in 0..depth {
            for x in 0..width {
                let cell = grid.get(x, y).expect("Cell should exist");
                assert_eq!(cell.visited, false);
                assert_eq!(cell.walls, [true, true, true, true]);
            }
        }
    }
    
    #[test]
    fn test_grid_out_of_bounds() {
        let mut grid = Grid::new(5, 5);
        assert!(grid.get(5, 5).is_none());
        assert!(grid.get_mut(5, 5).is_none());
    }
}
