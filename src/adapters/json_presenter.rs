use crate::adapters::voxel_mapper::{FaceDirection, MergedQuad, VoxelType};

/// Interface Adapter to convert MergedQuads to JSON format for the web presentation layer.
pub struct JsonPresenter;

impl JsonPresenter {
    pub fn render_voxels(quads: &[MergedQuad]) -> String {
        let mut json = String::with_capacity(quads.len() * 100);
        json.push('[');

        for (i, quad) in quads.iter().enumerate() {
            let dir_str = match quad.dir {
                FaceDirection::Up => "\"up\"",
                FaceDirection::Down => "\"down\"",
                FaceDirection::North => "\"north\"",
                FaceDirection::South => "\"south\"",
                FaceDirection::East => "\"east\"",
                FaceDirection::West => "\"west\"",
            };

            let type_str = match quad.v_type {
                VoxelType::Wall => "\"wall\"",
                VoxelType::Floor => "\"floor\"",
                VoxelType::Ceiling => "\"ceiling\"",
                VoxelType::Light => "\"light\"",
                VoxelType::RedWall => "\"redwall\"",
                VoxelType::Grass => "\"grass\"",
                VoxelType::Water => "\"water\"",
                VoxelType::Tree => "\"tree\"",
                VoxelType::RedLight => "\"redlight\"",
            };

            let obj = format!(
                "{{\"x\":{},\"y\":{},\"z\":{},\"w\":{},\"h\":{},\"dir\":{},\"type\":{},\"color\":{}}}",
                quad.x, quad.y, quad.z, quad.w, quad.h, dir_str, type_str, quad.color
            );

            json.push_str(&obj);
            if i < quads.len() - 1 {
                json.push(',');
            }
        }

        json.push(']');
        json
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rendering() {
        let quads = vec![MergedQuad {
            x: 0.0,
            y: 1.0,
            z: 2.0,
            w: 4.0,
            h: 1.0,
            dir: FaceDirection::Up,
            v_type: VoxelType::Wall,
            color: 1441813,
            light: 0,
            ao: 0,
        }];

        let json = JsonPresenter::render_voxels(&quads);
        assert_eq!(
            json,
            r#"[{"x":0,"y":1,"z":2,"w":4,"h":1,"dir":"up","type":"wall","color":1441813}]"#
        );
    }
}
