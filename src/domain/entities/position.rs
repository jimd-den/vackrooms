/// Position represents a 2D coordinate in the continuous noise space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f32,
    pub z: f32,
}

impl Position {
    pub fn new(x: f32, z: f32) -> Self {
        Self { x, z }
    }

    pub fn distance(&self, other: &Position) -> f32 {
        ((self.x - other.x).powi(2) + (self.z - other.z).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let p1 = Position::new(0.0, 0.0);
        let p2 = Position::new(3.0, 4.0);
        assert_eq!(p1.distance(&p2), 5.0);
    }
}
