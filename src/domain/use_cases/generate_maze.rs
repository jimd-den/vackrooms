use crate::domain::entities::grid::Grid;
use rand::Rng;
#[allow(unused_imports)]
use rand::RngExt;

/// Design Pattern: Strategy
/// We use the Strategy pattern here to encapsulate the maze generation algorithm.
/// This is the superior choice because different Backrooms levels (or even different regions of Level 0)
/// might require different procedural generation rules. By decoupling the generation logic from the Grid entity,
/// we ensure the Open/Closed Principle (workable iʃɛ́) — we can add new generators without modifying the Grid.
pub trait MazeGenerator {
    /// Generates the maze on the provided grid using the given random number generator.
    fn generate<R: Rng + ?Sized>(&self, grid: &mut Grid, rng: &mut R);
}

/// A concrete strategy for generating a maze using the Growing Tree algorithm.
/// This algorithm produces non-linear, organic layouts suitable for the Backrooms,
/// unlike standard depth-first search which produces long winding paths.
pub struct GrowingTreeGenerator {
    pub junction_density: f32,
}

impl MazeGenerator for GrowingTreeGenerator {
    fn generate<R: Rng + ?Sized>(&self, grid: &mut Grid, rng: &mut R) {
        let width = grid.width();
        let depth = grid.depth();
        if width == 0 || depth == 0 {
            return;
        }

        // Starting point
        let start_x = rng.random_range(0..width);
        let start_y = rng.random_range(0..depth);

        if let Some(cell) = grid.get_mut(start_x, start_y) {
            cell.visited = true;
        }

        let mut frontier = vec![(start_x, start_y)];

        while !frontier.is_empty() {
            // Pick a random cell from the frontier (this randomness defines the Growing Tree behavior)
            let idx = rng.random_range(0..frontier.len());
            let (cx, cy) = frontier[idx];

            // Define neighbors: (x, y, wall_here_idx, wall_there_idx)
            // 0: North, 1: East, 2: South, 3: West
            let neighbors: [(isize, isize, usize, usize); 4] = [
                (cx as isize, (cy as isize) - 1, 0, 2), // North
                ((cx as isize) + 1, cy as isize, 1, 3), // East
                (cx as isize, (cy as isize) + 1, 2, 0), // South
                ((cx as isize) - 1, cy as isize, 3, 1), // West
            ];

            let mut unvisited = Vec::new();
            for &(nx, ny, wall_here, wall_there) in &neighbors {
                if nx >= 0 && nx < width as isize && ny >= 0 && ny < depth as isize {
                    let (nx_u, ny_u) = (nx as usize, ny as usize);
                    if let Some(cell) = grid.get(nx_u, ny_u) {
                        if !cell.visited {
                            unvisited.push((nx_u, ny_u, wall_here, wall_there));
                        }
                    }
                }
            }

            if unvisited.is_empty() {
                frontier.swap_remove(idx);
            } else {
                let unvisited_idx = rng.random_range(0..unvisited.len());
                let (nx, ny, wall_here, wall_there) = unvisited[unvisited_idx];

                // Knock down walls
                if let Some(cell) = grid.get_mut(cx, cy) {
                    cell.walls[wall_here] = false;
                }
                if let Some(cell) = grid.get_mut(nx, ny) {
                    cell.walls[wall_there] = false;
                    cell.visited = true;
                }
                
                frontier.push((nx, ny));
            }
        }
        
        // Add loops to break the "perfect maze" property and make it feel like the Backrooms
        let base_openings = (width * depth) as f32 / 4.0;
        let extra_openings = (base_openings * self.junction_density) as usize;
        for _ in 0..extra_openings {
            let x = rng.random_range(0..width);
            let y = rng.random_range(0..depth);
            let dir = rng.random_range(0..4);
            
            if let Some(cell) = grid.get_mut(x, y) {
                cell.walls[dir] = false;
            }
            
            // Note: In a perfect implementation we would also knock down the corresponding wall
            // of the adjacent cell. But for simplicity and abstract logic, knocking down one side 
            // is effectively "no wall" in this direction. Let's do it properly though.
            let (nx, ny) = match dir {
                0 => (x as isize, y as isize - 1),
                1 => (x as isize + 1, y as isize),
                2 => (x as isize, y as isize + 1),
                3 => (x as isize - 1, y as isize),
                _ => unreachable!(),
            };
            
            if nx >= 0 && nx < width as isize && ny >= 0 && ny < depth as isize {
                let (nx, ny) = (nx as usize, ny as usize);
                let opposite_dir = (dir + 2) % 4;
                if let Some(n_cell) = grid.get_mut(nx, ny) {
                    n_cell.walls[opposite_dir] = false;
                }
            }
        }

        // Pass 3: Anomalies (Red rooms and dark rooms)
        for y in 0..depth {
            for x in 0..width {
                if let Some(cell) = grid.get_mut(x, y) {
                    // 5% chance of red room
                    if rng.random_bool(0.05) {
                        cell.is_red = true;
                    }
                    // 10% chance of dark room (no lights)
                    if rng.random_bool(0.10) {
                        cell.is_dark = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::grid::Grid;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_generate_maze_visits_all_cells() {
        let mut grid = Grid::new(10, 10);
        let mut rng = StdRng::seed_from_u64(42);
        
        let generator = GrowingTreeGenerator { junction_density: 1.0 };
        generator.generate(&mut grid, &mut rng);
        
        for y in 0..grid.depth() {
            for x in 0..grid.width() {
                let cell = grid.get(x, y).unwrap();
                assert!(cell.visited, "Cell at ({}, {}) was not visited", x, y);
            }
        }
    }
}
