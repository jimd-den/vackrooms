//! Presentation metadata for domain material ids.
//!
//! Generation owns what a material *is* (solid, emissive, walkable). This
//! adapter owns how that id is named and colored when it crosses a rendering
//! boundary. Keeping each visual in one row makes adding a material an
//! extension to this table instead of a set of unrelated renderer edits.

use crate::domain::entities::voxel_grid::VOXEL_MATERIAL_COUNT;
use crate::use_cases::ports::MaterialPalette;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialVisual {
    pub name: &'static str,
    pub color: u32,
}

const fn visual(name: &'static str, color: u32) -> MaterialVisual {
    MaterialVisual { name, color }
}

pub const MATERIAL_VISUALS: [MaterialVisual; VOXEL_MATERIAL_COUNT] = [
    visual("air", 0x000000),
    visual("wall", 0xDDCC66),
    visual("floor", 0x998811),
    visual("ceiling", 0xCCCCCC),
    visual("light", 0xFFF8D6),
    visual("redwall", 0x880000),
    visual("grass", 0x4F9A3D),
    visual("water", 0x3A6FB8),
    visual("tree", 0x6B4A2F),
    visual("redlight", 0xFF4433),
    visual("palewall", 0xD8D2C0),
    visual("damagedwall", 0x8A7F5C),
    visual("drycarpet", 0xC2B76B),
    visual("deepcarpet", 0x6B5E22),
    visual("stickycarpet", 0x7A4A26),
    visual("fluid", 0x2E2A22),
    visual("glimmer", 0x9FC4E8),
    visual("concretewall", 0x8F8D88),
    visual("tilefloor", 0xBDBBB0),
    visual("concretefloor", 0x6E6C66),
    visual("crate", 0x9C7B4A),
    visual("pipe", 0x3E4348),
    visual("metaldoor", 0x4A5A6A),
    visual("almondwater", 0xEDE6D0),
    visual("agedwallpaper", 0xA69456),
    visual("stainedcarpet", 0x746021),
];

const UNKNOWN_VISUAL: MaterialVisual = MaterialVisual {
    name: "unknown",
    color: 0x000000,
};

pub fn material_visual(material: u8) -> MaterialVisual {
    MATERIAL_VISUALS
        .get(usize::from(material))
        .copied()
        .unwrap_or(UNKNOWN_VISUAL)
}

pub fn material_color_f32(material: u8) -> [f32; 3] {
    let color = material_visual(material).color;
    [
        ((color >> 16) & 0xFF) as f32 / 255.0,
        ((color >> 8) & 0xFF) as f32 / 255.0,
        (color & 0xFF) as f32 / 255.0,
    ]
}

/// Stateless production palette, shared by every adapter composition root.
pub struct DefaultMaterialPalette;

impl MaterialPalette for DefaultMaterialPalette {
    fn color(&self, material: u8) -> u32 {
        material_visual(material).color
    }
}

pub static DEFAULT_MATERIAL_PALETTE: DefaultMaterialPalette = DefaultMaterialPalette;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::entities::voxel_grid::{VOXEL_AGED_WALLPAPER, VOXEL_STAINED_CARPET};

    #[test]
    fn every_domain_material_has_one_visual() {
        assert_eq!(MATERIAL_VISUALS.len(), VOXEL_MATERIAL_COUNT);
        assert_eq!(material_visual(VOXEL_AGED_WALLPAPER).name, "agedwallpaper");
        assert_eq!(material_visual(VOXEL_STAINED_CARPET).name, "stainedcarpet");
    }

    #[test]
    fn unknown_materials_have_an_explicit_fallback() {
        assert_eq!(material_visual(u8::MAX), UNKNOWN_VISUAL);
    }
}
