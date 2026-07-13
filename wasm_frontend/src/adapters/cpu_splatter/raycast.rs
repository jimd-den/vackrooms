//! Stack-based SVO ray traversal — the CPU twin of the GLSL `raymarchSVO`.
//!
//! Used for *secondary* rays only (flashlight-beam occlusion and hero-light
//! shadow rays); primary visibility comes from splatting, not marching.
//! The algorithm matches the raymarch shader step for step:
//!
//! 1. Slab-test every chunk AABB and optionally sort hits near-to-far
//!    ([`trace_svo`], controlled by `rt_f2b`).
//! 2. Inside a chunk, descend toward the octant containing the current
//!    point; **empty-space skip** jumps a whole empty leaf/octant in one
//!    step via its AABB exit plane instead of voxel-by-voxel DDA.
//! 3. After a skip, pop every ancestor the point exited (a skip through a
//!    shared plane can leave several boxes at once).

use crate::application::ports::ChunkDraw;

use super::atlas::decode_node;

/// A solid-leaf intersection: ray parameter and the voxel's material type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub t: f32,
    pub voxel_type: u32,
}

/// Keeps slab divisions finite for axis-aligned rays. `f32::signum()` is
/// zero for `0.0`, so multiplying it by epsilon would leave the divisor at
/// zero; choose the sign explicitly instead.
fn nonzero_direction(component: f32) -> f32 {
    if component.abs() >= 1.0e-4 {
        component
    } else if component.is_sign_negative() {
        -1.0e-4
    } else {
        1.0e-4
    }
}

/// One saved traversal level: the parent node and its bounds.
#[derive(Clone, Copy)]
struct TraversalFrame {
    node_idx: usize,
    b_min: [f32; 3],
    b_max: [f32; 3],
}

/// The mutable cursor of one chunk march: current node, its bounds, and the
/// ancestor stack (an SVO chunk is at most 8 levels deep, +1 slack).
struct TraversalCursor {
    stack: [TraversalFrame; 9],
    stack_ptr: usize,
    node: usize,
    b_min: [f32; 3],
    b_max: [f32; 3],
}

impl TraversalCursor {
    fn at_root(root: usize, world_size: f32) -> Self {
        Self {
            stack: [TraversalFrame {
                node_idx: 0,
                b_min: [0.0; 3],
                b_max: [0.0; 3],
            }; 9],
            stack_ptr: 0,
            node: root,
            b_min: [0.0; 3],
            b_max: [world_size; 3],
        }
    }

    fn push(&mut self) {
        if self.stack_ptr < 8 {
            self.stack[self.stack_ptr] = TraversalFrame {
                node_idx: self.node,
                b_min: self.b_min,
                b_max: self.b_max,
            };
            self.stack_ptr += 1;
        }
    }

    /// Pops EVERY level `p` exited: a skip through a plane shared by several
    /// ancestor boxes leaves `p` outside more than one of them, and
    /// descending from a box that no longer contains `p` corrupts the
    /// traversal (wall cracks). Returns `false` when `p` left the chunk.
    fn pop_exited(&mut self, p: [f32; 3]) -> bool {
        while p[0] < self.b_min[0]
            || p[0] > self.b_max[0]
            || p[1] < self.b_min[1]
            || p[1] > self.b_max[1]
            || p[2] < self.b_min[2]
            || p[2] > self.b_max[2]
        {
            if self.stack_ptr == 0 {
                return false;
            }
            self.stack_ptr -= 1;
            let frame = self.stack[self.stack_ptr];
            self.node = frame.node_idx;
            self.b_min = frame.b_min;
            self.b_max = frame.b_max;
        }
        true
    }
}

/// Advances the ray to `bounds`' exit and nudges every tied axis just past
/// its plane. Leaving an axis exactly *on* its plane stalls the march (t
/// stops advancing), so the epsilon push is required for termination.
fn skip_to_exit(ro: [f32; 3], rd: [f32; 3], b_min: [f32; 3], b_max: [f32; 3]) -> (f32, [f32; 3]) {
    let exit_plane = |axis: usize| {
        if rd[axis] > 0.0 {
            b_max[axis]
        } else {
            b_min[axis]
        }
    };
    let t_max_planes = [
        (exit_plane(0) - ro[0]) / rd[0],
        (exit_plane(1) - ro[1]) / rd[1],
        (exit_plane(2) - ro[2]) / rd[2],
    ];
    let t_exit = t_max_planes[0].min(t_max_planes[1]).min(t_max_planes[2]);
    let mut p = [
        ro[0] + t_exit * rd[0],
        ro[1] + t_exit * rd[1],
        ro[2] + t_exit * rd[2],
    ];
    for axis in 0..3 {
        if (t_exit - t_max_planes[axis]).abs() < 0.0001 {
            p[axis] = exit_plane(axis) + if rd[axis] > 0.0 { 0.001 } else { -0.001 };
        }
    }
    (t_exit, p)
}

/// Marches one chunk's SVO from `t_entry`. `ro`/`rd` are chunk-local.
/// Returns the first solid leaf hit, or `None`.
fn raymarch_svo_single(
    atlas: &[u32],
    ro: [f32; 3],
    rd: [f32; 3],
    chunk_root_idx: usize,
    t_entry: f32,
    world_size: f32,
) -> Option<RayHit> {
    let mut t = t_entry;
    let mut p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
    let mut cursor = TraversalCursor::at_root(chunk_root_idx, world_size);

    // Step budget: a bounded march can never hang a frame even on a
    // malformed atlas. 160 comfortably covers the deepest real traversals.
    const MAX_STEPS: i32 = 160;
    for _ in 0..MAX_STEPS {
        let node = decode_node(atlas, cursor.node)?;

        if node.is_leaf {
            if node.voxel_type != 0 {
                return Some(RayHit {
                    t,
                    voxel_type: node.voxel_type,
                });
            }
            // TRAVERSAL: empty-space skip — one step to this empty
            // leaf's exit plane instead of voxel-by-voxel stepping.
            (t, p) = skip_to_exit(ro, rd, cursor.b_min, cursor.b_max);
            if !cursor.pop_exited(p) {
                return None;
            }
            continue;
        }

        let center = [
            (cursor.b_min[0] + cursor.b_max[0]) * 0.5,
            (cursor.b_min[1] + cursor.b_max[1]) * 0.5,
            (cursor.b_min[2] + cursor.b_max[2]) * 0.5,
        ];
        let ox = (p[0] >= center[0]) as usize;
        let oy = (p[1] >= center[1]) as usize;
        let oz = (p[2] >= center[2]) as usize;
        let child_idx = (oz << 2) | (oy << 1) | ox;

        // The octant's own bounds, picked axis-by-axis from the parent box
        // and its center plane.
        //
        // BUG FIX (cone light): `oct_min[1]` used to read `center[1]` in
        // both branches. Every secondary ray crossing an empty lower-Y
        // octant computed a too-near exit plane, stalled until the step
        // budget ran out, and reported "no hit" — so the flashlight beam
        // shone through floors and hero-shadow rays never found occluders.
        let oct_min = [
            if ox == 1 { center[0] } else { cursor.b_min[0] },
            if oy == 1 { center[1] } else { cursor.b_min[1] },
            if oz == 1 { center[2] } else { cursor.b_min[2] },
        ];
        let oct_max = [
            if ox == 1 { cursor.b_max[0] } else { center[0] },
            if oy == 1 { cursor.b_max[1] } else { center[1] },
            if oz == 1 { cursor.b_max[2] } else { center[2] },
        ];

        if (node.child_mask & (1 << child_idx)) != 0 {
            // Descend into the occupied octant.
            cursor.push();
            cursor.b_min = oct_min;
            cursor.b_max = oct_max;
            cursor.node = node.child_base + child_idx;
        } else {
            // TRAVERSAL: masked-out (air) octant skipped in one step.
            (t, p) = skip_to_exit(ro, rd, oct_min, oct_max);
            if !cursor.pop_exited(p) {
                return None;
            }
        }
    }
    None
}

/// Traces a world-space ray through every resident chunk. Returns the first
/// solid hit within `max_t`.
pub fn trace_svo(
    atlas: &[u32],
    chunks: &[ChunkDraw],
    origin: [f32; 3],
    direction: [f32; 3],
    max_t: f32,
    front_to_back: bool,
) -> Option<RayHit> {
    struct ChunkHit {
        idx: usize,
        t_min: f32,
    }

    let mut hits = Vec::with_capacity(chunks.len());

    // Axis-aligned rays divide by the direction components; clamp near-zero
    // components so the slab test stays finite.
    let safe_rd = direction.map(nonzero_direction);
    let inv_rd = [1.0 / safe_rd[0], 1.0 / safe_rd[1], 1.0 / safe_rd[2]];

    for (i, chunk) in chunks.iter().enumerate() {
        let local_ro = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];

        // Slab test against the chunk's [0, world_size]^3 cube.
        let mut t_entry = f32::NEG_INFINITY;
        let mut t_exit = f32::INFINITY;
        for axis in 0..3 {
            let t1 = (0.0 - local_ro[axis]) * inv_rd[axis];
            let t2 = (chunk.world_size - local_ro[axis]) * inv_rd[axis];
            t_entry = t_entry.max(t1.min(t2));
            t_exit = t_exit.min(t1.max(t2));
        }

        if t_entry < t_exit && t_exit > 0.0 && t_entry < max_t {
            hits.push(ChunkHit {
                idx: i,
                t_min: t_entry.max(0.0),
            });
        }
    }

    // OPTIMIZATION (rt_f2b): nearest chunk first permits the first valid hit
    // to return immediately. The disabled reference path visits input order
    // and explicitly retains the nearest hit, producing the same answer.
    if front_to_back {
        hits.sort_by(|a, b| a.t_min.total_cmp(&b.t_min));
    }

    let mut closest = None;
    for hit in hits {
        let chunk = &chunks[hit.idx];
        let local_ro = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];

        if let Some(ray_hit) = raymarch_svo_single(
            atlas,
            local_ro,
            safe_rd,
            chunk.root_index as usize,
            hit.t_min,
            chunk.world_size,
        ) {
            // BUG FIX (cone light): the old code ignored `max_t` once inside
            // a chunk, so a wall *behind* the lit surface could "occlude"
            // the flashlight ray and randomly black out beam splats.
            if ray_hit.t <= max_t {
                if front_to_back {
                    return Some(ray_hit);
                }
                if closest.is_none_or(|current: RayHit| ray_hit.t < current.t) {
                    closest = Some(ray_hit);
                }
            }
        }
    }

    closest
}

#[cfg(test)]
mod tests {
    use super::nonzero_direction;

    #[test]
    fn axis_aligned_zero_gets_a_finite_slab_divisor() {
        for component in [0.0, -0.0, 1.0e-8, -1.0e-8] {
            let safe = nonzero_direction(component);
            assert!(safe != 0.0);
            assert!((1.0 / safe).is_finite());
        }
    }
}
