use wasm_bindgen::prelude::*;

const VOXEL_AIR: u8 = 0;
const VOXEL_LIGHT: u8 = 4;

#[wasm_bindgen]
pub struct CPURaycaster {
    grid_data: Vec<u8>,
    light_data: Vec<u8>,
    width: usize,
    height: usize,
    depth: usize,
    frame_buffer: Vec<u8>,
    screen_width: usize,
    screen_height: usize,
}

#[wasm_bindgen]
impl CPURaycaster {
    #[wasm_bindgen(constructor)]
    pub fn new(screen_width: usize, screen_height: usize) -> Self {
        Self {
            grid_data: Vec::new(),
            light_data: Vec::new(),
            width: 0,
            height: 0,
            depth: 0,
            frame_buffer: vec![255; screen_width * screen_height * 4],
            screen_width,
            screen_height,
        }
    }

    pub fn set_grid(&mut self, width: usize, height: usize, depth: usize, data: &[u8], light: &[u8]) {
        self.width = width;
        self.height = height;
        self.depth = depth;
        self.grid_data = data.to_vec();
        self.light_data = light.to_vec();
    }
    
    pub fn get_frame_buffer(&self) -> *const u8 {
        self.frame_buffer.as_ptr()
    }

    fn get_voxel(&self, x: isize, y: isize, z: isize) -> u8 {
        if x < 0 || y < 0 || z < 0 || x >= self.width as isize || y >= self.height as isize || z >= self.depth as isize {
            return VOXEL_AIR;
        }
        self.grid_data[(y as usize * self.width * self.depth) + (z as usize * self.width) + (x as usize)]
    }

    fn get_light(&self, x: isize, y: isize, z: isize) -> u8 {
        if x < 0 || y < 0 || z < 0 || x >= self.width as isize || y >= self.height as isize || z >= self.depth as isize {
            return 0;
        }
        self.light_data[(y as usize * self.width * self.depth) + (z as usize * self.width) + (x as usize)]
    }

    pub fn render(&mut self, px: f32, py: f32, pz: f32, yaw: f32, pitch: f32, fov_deg: f32) {
        // Clear background to dark fog
        for i in (0..self.frame_buffer.len()).step_by(4) {
            self.frame_buffer[i] = 10;     // R
            self.frame_buffer[i + 1] = 10; // G
            self.frame_buffer[i + 2] = 10; // B
            self.frame_buffer[i + 3] = 255; // A
        }

        if self.width == 0 {
            return;
        }

        let fov_rad = fov_deg * std::f32::consts::PI / 180.0;
        let half_fov = fov_rad / 2.0;

        let dir_x = -yaw.sin();
        let dir_z = -yaw.cos();
        let plane_x = yaw.cos() * half_fov.tan();
        let plane_z = -yaw.sin() * half_fov.tan();

        for x in 0..self.screen_width {
            let camera_x = 2.0 * (x as f32) / (self.screen_width as f32) - 1.0;
            let ray_dir_x = dir_x + plane_x * camera_x;
            let ray_dir_z = dir_z + plane_z * camera_x;
            
            let mut map_x = px.floor() as isize;
            let mut map_z = pz.floor() as isize;
            
            let delta_dist_x = if ray_dir_x == 0.0 { f32::INFINITY } else { (1.0 / ray_dir_x).abs() };
            let delta_dist_z = if ray_dir_z == 0.0 { f32::INFINITY } else { (1.0 / ray_dir_z).abs() };
            
            let mut side_dist_x;
            let mut side_dist_z;
            let step_x: isize;
            let step_z: isize;
            
            if ray_dir_x < 0.0 {
                step_x = -1;
                side_dist_x = (px - map_x as f32) * delta_dist_x;
            } else {
                step_x = 1;
                side_dist_x = (map_x as f32 + 1.0 - px) * delta_dist_x;
            }
            
            if ray_dir_z < 0.0 {
                step_z = -1;
                side_dist_z = (pz - map_z as f32) * delta_dist_z;
            } else {
                step_z = 1;
                side_dist_z = (map_z as f32 + 1.0 - pz) * delta_dist_z;
            }
            
            let mut hit = 0;
            let mut side = 0;
            let mut perp_wall_dist = 0.0;
            let mut hit_y = 0;
            
            let map_y = py.floor() as isize;
            let max_steps = 200;
            let mut steps = 0;

            while hit == 0 && steps < max_steps {
                if side_dist_x < side_dist_z {
                    side_dist_x += delta_dist_x;
                    map_x += step_x;
                    side = 0;
                } else {
                    side_dist_z += delta_dist_z;
                    map_z += step_z;
                    side = 1;
                }
                
                let v = self.get_voxel(map_x, map_y, map_z);
                if v != VOXEL_AIR {
                    hit = v;
                    hit_y = map_y;
                }
                steps += 1;
            }
            
            if hit > 0 {
                if side == 0 {
                    perp_wall_dist = side_dist_x - delta_dist_x;
                } else {
                    perp_wall_dist = side_dist_z - delta_dist_z;
                }
                
                let line_height = (self.screen_height as f32 / perp_wall_dist).abs() as isize;
                let pitch_offset = (pitch * self.screen_height as f32) as isize;
                
                let mut draw_start = -line_height / 2 + self.screen_height as isize / 2 + pitch_offset;
                let draw_start_clamped = if draw_start < 0 { 0 } else { draw_start as usize };
                let mut draw_end = line_height / 2 + self.screen_height as isize / 2 + pitch_offset;
                if draw_end >= self.screen_height as isize {
                    draw_end = self.screen_height as isize - 1;
                }
                let draw_end_clamped = draw_end as usize;
                
                let mut r = 200;
                let mut g = 200;
                let mut b = 200;
                
                if hit == VOXEL_LIGHT {
                    r = 255; g = 255; b = 200;
                } else if hit == 5 {
                    r = 150; g = 30; b = 30;
                } else if hit == 1 {
                    r = 180; g = 160; b = 100;
                }
                
                let light = self.get_light(map_x, hit_y, map_z);
                let light_factor = if hit == VOXEL_LIGHT { 1.0 } else { light as f32 / 15.0 };
                let fog_factor = (-perp_wall_dist * 0.05).exp();
                
                r = (r as f32 * light_factor * fog_factor) as u8;
                g = (g as f32 * light_factor * fog_factor) as u8;
                b = (b as f32 * light_factor * fog_factor) as u8;
                
                if side == 1 && hit != VOXEL_LIGHT {
                    r = (r as f32 * 0.7) as u8;
                    g = (g as f32 * 0.7) as u8;
                    b = (b as f32 * 0.7) as u8;
                }
                
                for y in draw_start_clamped..=draw_end_clamped {
                    let pixel_idx = (y * self.screen_width + x) * 4;
                    self.frame_buffer[pixel_idx] = r;
                    self.frame_buffer[pixel_idx + 1] = g;
                    self.frame_buffer[pixel_idx + 2] = b;
                    self.frame_buffer[pixel_idx + 3] = 255;
                }
                
                let floor_fog = (-perp_wall_dist * 0.01).exp(); // less fog on floor to see shape
                for y in (draw_end_clamped + 1)..self.screen_height {
                    let pixel_idx = (y * self.screen_width + x) * 4;
                    self.frame_buffer[pixel_idx] = (40.0 * floor_fog) as u8;
                    self.frame_buffer[pixel_idx + 1] = (40.0 * floor_fog) as u8;
                    self.frame_buffer[pixel_idx + 2] = (40.0 * floor_fog) as u8;
                    self.frame_buffer[pixel_idx + 3] = 255;
                }
                
                for y in 0..draw_start_clamped {
                    let pixel_idx = (y * self.screen_width + x) * 4;
                    self.frame_buffer[pixel_idx] = (60.0 * floor_fog) as u8;
                    self.frame_buffer[pixel_idx + 1] = (55.0 * floor_fog) as u8;
                    self.frame_buffer[pixel_idx + 2] = (50.0 * floor_fog) as u8;
                    self.frame_buffer[pixel_idx + 3] = 255;
                }
            }
        }
    }
}
