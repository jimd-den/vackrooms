use crate::entities::models::Position;
use crate::use_cases::ports::NoiseProvider;

/// SimpleNoiseProvider is a concrete implementation of the NoiseProvider port.
/// It resides in the Frameworks/Drivers layer.
/// In a production scenario, this might wrap `noise-rs` or SIMD-accelerated noise.
/// For Dependency Minimalism, we implement a basic seeded value noise.
pub struct SimpleNoiseProvider;

impl SimpleNoiseProvider {
    pub fn new() -> Self {
        Self
    }

    /// A deterministic pseudo-random hash function for 2D integer coordinates.
    fn hash2d(seed: u32, x: i32, y: i32) -> f32 {
        // Simple hash mixing
        let mut h = seed
            .wrapping_add(x as u32 ^ 0x9E3779B9)
            .wrapping_add(y as u32 ^ 0x85EBCA6B);
        h ^= h >> 16;
        h = h.wrapping_mul(0x85EBCA6B);
        h ^= h >> 13;
        h = h.wrapping_mul(0xC2B2AE35);
        h ^= h >> 16;
        
        // Normalize to [-1.0, 1.0]
        ((h as f32) / (std::u32::MAX as f32)) * 2.0 - 1.0
    }

    /// Linear interpolation
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    /// Smoothstep for softer transitions
    fn smoothstep(t: f32) -> f32 {
        t * t * (3.0 - 2.0 * t)
    }
}

impl NoiseProvider for SimpleNoiseProvider {
    fn evaluate_2d(&self, seed: u32, position: Position) -> f32 {
        // Scale frequency to make the fields low-frequency
        let freq = 0.05;
        let x = position.x * freq;
        let y = position.z * freq;

        let x0 = x.floor() as i32;
        let x1 = x0 + 1;
        let y0 = y.floor() as i32;
        let y1 = y0 + 1;

        let tx = x - (x0 as f32);
        let ty = y - (y0 as f32);
        
        let u = Self::smoothstep(tx);
        let v = Self::smoothstep(ty);

        let v00 = Self::hash2d(seed, x0, y0);
        let v10 = Self::hash2d(seed, x1, y0);
        let v01 = Self::hash2d(seed, x0, y1);
        let v11 = Self::hash2d(seed, x1, y1);

        let nx0 = Self::lerp(v00, v10, u);
        let nx1 = Self::lerp(v01, v11, u);

        Self::lerp(nx0, nx1, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_determinism() {
        let provider = SimpleNoiseProvider::new();
        let pos = Position::new(42.5, -17.2);
        
        let val1 = provider.evaluate_2d(123, pos);
        let val2 = provider.evaluate_2d(123, pos);
        let val3 = provider.evaluate_2d(999, pos);
        
        assert_eq!(val1, val2);
        assert_ne!(val1, val3);
    }
}
