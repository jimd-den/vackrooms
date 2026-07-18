//! Browser-owned field-of-view preference, sampled once at the frame boundary.

use std::cell::Cell;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::application::rendering::DEFAULT_FOV_TAN;

thread_local! {
    static FOV_TAN: Cell<f32> = const { Cell::new(DEFAULT_FOV_TAN) };
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub fn set_fov(degrees: f32) {
    let degrees = if degrees.is_finite() { degrees } else { 75.0 };
    FOV_TAN.with(|fov| fov.set((degrees.clamp(40.0, 110.0).to_radians() * 0.5).tan()));
}

pub fn fov_tan() -> f32 {
    FOV_TAN.with(Cell::get)
}
