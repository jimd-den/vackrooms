//! Stack-based SVO ray traversal — the CPU twin of the GLSL `raymarchSVO`.
//!
//! Primary visibility comes from splatting, not marching. This module offers
//! two explicit query contracts: [`trace_svo`] returns the nearest hit for
//! reference/debug callers, while [`svo_segment_is_occluded`] is the
//! allocation-free early-out used by flashlight and hero visibility.
//! Both share the same bounded single-chunk marcher. Its work budget is
//! derived from the chunk's real voxel resolution, so selecting 0.05-unit
//! voxels cannot silently inherit the old 160-step coarse-scene limit:
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

/// The parameter interval over which a ray lies inside an axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RayInterval {
    pub entry: f32,
    pub exit: f32,
}

/// Exact ray/AABB slab intersection shared by chunk traversal and flashlight
/// receiver geometry.
///
/// A zero direction component means that axis is parallel: the ray either
/// remains inside that slab forever or can never enter it. Tiny *non-zero*
/// components remain untouched. Replacing them with an epsilon changes the
/// ray itself and made CPU flashlight shadows jump when yaw or pitch crossed
/// a cardinal direction.
pub(super) fn ray_box_interval(
    origin: [f32; 3],
    direction: [f32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
) -> Option<RayInterval> {
    let mut entry = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    let mut has_direction = false;

    for axis in 0..3 {
        let origin_axis = origin[axis];
        let direction_axis = direction[axis];
        if !origin_axis.is_finite()
            || !direction_axis.is_finite()
            || !bounds_min[axis].is_finite()
            || !bounds_max[axis].is_finite()
        {
            return None;
        }

        if direction_axis == 0.0 {
            if origin_axis < bounds_min[axis] || origin_axis > bounds_max[axis] {
                return None;
            }
            continue;
        }

        has_direction = true;
        let first = (bounds_min[axis] - origin_axis) / direction_axis;
        let second = (bounds_max[axis] - origin_axis) / direction_axis;
        entry = entry.max(first.min(second));
        exit = exit.min(first.max(second));

        if entry > exit {
            return None;
        }
    }

    has_direction.then_some(RayInterval { entry, exit })
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

/// Result of marching one intersected chunk.
///
/// `Indeterminate` is deliberately distinct from a clear segment. A corrupt
/// atlas or exhausted safety budget must not become a light leak: visibility
/// queries treat it as blocked, while the nearest-hit diagnostic API (which
/// cannot represent an unknown material) simply continues to another chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChunkMarch {
    Hit(RayHit),
    Clear,
    Indeterminate,
}

const MIN_TRAVERSAL_STEPS: usize = 160;
const MAX_TRAVERSAL_STEPS: usize = 32_768;

/// Conservative upper work bound for a valid octree traversal.
///
/// A line crosses at most roughly three times the leaf count of a cubic grid.
/// In the pessimistic sparse-tree case each crossing can descend every SVO
/// level again. The absolute ceiling still protects a frame from malformed or
/// hostile metadata; reaching it produces [`ChunkMarch::Indeterminate`].
fn traversal_step_budget(world_size: f32, voxel_size: f32, svo_depth: u8) -> usize {
    let geometry_cells =
        if world_size.is_finite() && voxel_size.is_finite() && world_size > 0.0 && voxel_size > 0.0
        {
            (world_size / voxel_size).ceil().clamp(1.0, 1_048_576.0) as u64
        } else {
            1
        };
    let depth_cells = 1_u64
        .checked_shl(u32::from(svo_depth.min(20)))
        .unwrap_or(1_048_576);
    let cells_per_axis = geometry_cells.max(depth_cells);
    let levels = u64::from(svo_depth).saturating_add(1);
    let estimated = cells_per_axis
        .saturating_mul(3)
        .saturating_add(1)
        .saturating_mul(levels)
        .saturating_add(16);
    estimated.clamp(MIN_TRAVERSAL_STEPS as u64, MAX_TRAVERSAL_STEPS as u64) as usize
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

/// Advances the ray to `bounds`' exit and nudges every tied axis by one
/// representable float just past its plane.
///
/// This is scale independent. The old fixed `0.001` world-unit push could
/// jump over geometry when the user selected 0.05-unit voxels, while its
/// fixed tie epsilon merged unrelated planes on long rays.
fn skip_to_exit(
    ro: [f32; 3],
    rd: [f32; 3],
    b_min: [f32; 3],
    b_max: [f32; 3],
) -> Option<(f32, [f32; 3])> {
    let exit_plane = |axis: usize| match rd[axis].total_cmp(&0.0) {
        std::cmp::Ordering::Greater => b_max[axis],
        std::cmp::Ordering::Less => b_min[axis],
        std::cmp::Ordering::Equal => ro[axis],
    };
    let axis_exit = |axis: usize| {
        if rd[axis] == 0.0 {
            f32::INFINITY
        } else {
            (exit_plane(axis) - ro[axis]) / rd[axis]
        }
    };
    let t_max_planes = [axis_exit(0), axis_exit(1), axis_exit(2)];
    let t_exit = t_max_planes[0].min(t_max_planes[1]).min(t_max_planes[2]);
    if !t_exit.is_finite() {
        return None;
    }
    let mut p = [
        ro[0] + t_exit * rd[0],
        ro[1] + t_exit * rd[1],
        ro[2] + t_exit * rd[2],
    ];
    for axis in 0..3 {
        let axis_t = t_max_planes[axis];
        if !axis_t.is_finite() {
            continue;
        }
        let tie_tolerance = f32::EPSILON * 4.0 * t_exit.abs().max(axis_t.abs()).max(1.0);
        if (t_exit - axis_t).abs() <= tie_tolerance {
            let plane = exit_plane(axis);
            p[axis] = if rd[axis] > 0.0 {
                plane.next_up()
            } else {
                plane.next_down()
            };
        }
    }
    Some((t_exit, p))
}

/// Marches one chunk's SVO from `t_entry`. `ro`/`rd` are chunk-local.
/// Distinguishes a proven-clear segment from an invalid or exhausted march.
fn raymarch_svo_single(
    atlas: &[u32],
    ro: [f32; 3],
    rd: [f32; 3],
    chunk_root_idx: usize,
    t_entry: f32,
    max_t: f32,
    world_size: f32,
    voxel_size: f32,
    svo_depth: u8,
) -> ChunkMarch {
    let mut t = t_entry;
    let mut p = [ro[0] + t * rd[0], ro[1] + t * rd[1], ro[2] + t * rd[2]];
    let mut cursor = TraversalCursor::at_root(chunk_root_idx, world_size);

    let max_steps = traversal_step_budget(world_size, voxel_size, svo_depth);
    for _ in 0..max_steps {
        // A finite visibility segment must not spend the rest of its step
        // budget walking geometry beyond the receiver. Equality remains a
        // valid endpoint hit, matching `trace_svo`'s public contract.
        if t > max_t {
            return ChunkMarch::Clear;
        }
        let Some(node) = decode_node(atlas, cursor.node) else {
            return ChunkMarch::Indeterminate;
        };

        if node.is_leaf {
            if node.voxel_type != 0 {
                return ChunkMarch::Hit(RayHit {
                    t,
                    voxel_type: node.voxel_type,
                });
            }
            // TRAVERSAL: empty-space skip — one step to this empty
            // leaf's exit plane instead of voxel-by-voxel stepping.
            let Some(next) = skip_to_exit(ro, rd, cursor.b_min, cursor.b_max) else {
                return ChunkMarch::Indeterminate;
            };
            (t, p) = next;
            if !cursor.pop_exited(p) {
                return ChunkMarch::Clear;
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
            let Some(next) = skip_to_exit(ro, rd, oct_min, oct_max) else {
                return ChunkMarch::Indeterminate;
            };
            (t, p) = next;
            if !cursor.pop_exited(p) {
                return ChunkMarch::Clear;
            }
        }
    }
    ChunkMarch::Indeterminate
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

    for (i, chunk) in chunks.iter().enumerate() {
        let local_ro = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];

        let Some(interval) = ray_box_interval(local_ro, direction, [0.0; 3], [chunk.world_size; 3])
        else {
            continue;
        };

        if interval.entry < interval.exit && interval.exit > 0.0 && interval.entry < max_t {
            hits.push(ChunkHit {
                idx: i,
                t_min: interval.entry.max(0.0),
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
        // OPTIMIZATION (rt_f2b): sorting permits a safe early exit once the
        // next chunk does not begin before the closest solid already found.
        // Returning after the first intersected chunk is not equivalent:
        // padded chunk AABBs overlap, so a nearer box entry can contain a
        // farther solid than the following box.
        if front_to_back && closest.is_some_and(|current: RayHit| current.t <= hit.t_min) {
            break;
        }

        let chunk = &chunks[hit.idx];
        let local_ro = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];

        if let ChunkMarch::Hit(ray_hit) = raymarch_svo_single(
            atlas,
            local_ro,
            direction,
            chunk.root_index as usize,
            hit.t_min,
            max_t,
            chunk.world_size,
            chunk.voxel_size,
            chunk.svo_depth,
        ) {
            // BUG FIX (cone light): the old code ignored `max_t` once inside
            // a chunk, so a wall *behind* the lit surface could "occlude"
            // the flashlight ray and randomly black out beam splats.
            if ray_hit.t <= max_t {
                if closest.is_none_or(|current: RayHit| ray_hit.t < current.t) {
                    closest = Some(ray_hit);
                }
            }
        }
    }

    closest
}

/// Returns `true` as soon as any solid voxel intersects the finite ray
/// segment `[origin, origin + direction * max_t]`.
///
/// Visibility queries do not need the nearest material hit. Keeping that
/// contract separate from [`trace_svo`] avoids its per-query chunk-hit
/// allocation and sort: each resident chunk is slab-tested in place and the
/// first solid hit inside the segment terminates the scan. The SVO march is
/// still resolution-bounded and absolutely capped, so malformed atlas data
/// cannot hang a flashlight or fixture-shadow query.
pub fn svo_segment_is_occluded(
    atlas: &[u32],
    chunks: &[ChunkDraw],
    origin: [f32; 3],
    direction: [f32; 3],
    max_t: f32,
) -> bool {
    if !max_t.is_finite() || max_t <= 0.0 {
        return false;
    }

    for chunk in chunks {
        let local_origin = [
            origin[0] - chunk.origin[0],
            origin[1] - chunk.origin[1],
            origin[2] - chunk.origin[2],
        ];
        let Some(interval) =
            ray_box_interval(local_origin, direction, [0.0; 3], [chunk.world_size; 3])
        else {
            continue;
        };
        if interval.entry >= interval.exit || interval.exit <= 0.0 || interval.entry >= max_t {
            continue;
        }

        let entry = interval.entry.max(0.0);
        match raymarch_svo_single(
            atlas,
            local_origin,
            direction,
            chunk.root_index as usize,
            entry,
            max_t,
            chunk.world_size,
            chunk.voxel_size,
            chunk.svo_depth,
        ) {
            ChunkMarch::Hit(hit) if hit.t <= max_t => return true,
            // Unknown traversal is conservatively shadowed. Treating it as
            // clear created bright leaks exactly where fine-voxel scenes
            // exceeded the former fixed step budget.
            ChunkMarch::Indeterminate => return true,
            ChunkMarch::Hit(_) | ChunkMarch::Clear => {}
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::{
        ray_box_interval, skip_to_exit, svo_segment_is_occluded, trace_svo, traversal_step_budget,
    };
    use crate::application::ports::ChunkDraw;

    #[test]
    fn parallel_slab_axes_do_not_change_the_ray() {
        let bounds_min = [0.0; 3];
        let bounds_max = [2.0; 3];

        let exact = ray_box_interval([1.0, 1.0, -1.0], [0.0, 0.0, 1.0], bounds_min, bounds_max)
            .expect("axis-aligned ray crosses the box");
        assert_eq!(exact.entry, 1.0);
        assert_eq!(exact.exit, 3.0);

        let tiny = ray_box_interval(
            [1.0, 1.0, -1.0],
            [5.0e-7, -5.0e-7, 1.0],
            bounds_min,
            bounds_max,
        )
        .expect("tiny real components still cross the box");
        assert_eq!(tiny.entry, 1.0);
        assert_eq!(tiny.exit, 3.0);
    }

    #[test]
    fn parallel_ray_outside_a_slab_misses() {
        assert!(ray_box_interval([3.0, 1.0, -1.0], [0.0, 0.0, 1.0], [0.0; 3], [2.0; 3],).is_none());
    }

    #[test]
    fn skip_ignores_parallel_planes() {
        let (t, point) = skip_to_exit([1.0, 1.0, -1.0], [0.0, 0.0, 1.0], [0.0; 3], [2.0; 3])
            .expect("the moving z axis has an exit");
        assert_eq!(t, 3.0);
        assert_eq!(point, [1.0, 1.0, 2.0_f32.next_up()]);
    }

    #[test]
    fn traversal_budget_scales_with_user_selected_voxel_resolution() {
        let coarse = traversal_step_budget(16.0, 1.0, 4);
        let fine = traversal_step_budget(12.8, 0.05, 8);

        assert_eq!(coarse, 261);
        assert!(fine > 6_000, "fine budget was only {fine} steps");
        assert!(fine > coarse);
    }

    #[test]
    fn front_to_back_is_exact_for_overlapping_chunk_bounds() {
        // Root A starts first but contains its solid only in the far-z
        // octant. Root B starts later and is solid immediately. This is the
        // padded/overlapping-chunk case where returning the first chunk hit
        // reports t=3 instead of the true nearest t=2.
        let mut atlas = Vec::new();
        atlas.extend([0, 1, 1 << 4, 0]); // root A, child 4 occupied
        for child in 0..8 {
            let material = if child == 4 { 1 } else { 0 };
            atlas.extend([1, material, 0x00FF_FFFF, 0]);
        }
        let root_b = atlas.len() / 4;
        atlas.extend([1, 1, 0x00FF_FFFF, 0]);

        let chunks = [
            ChunkDraw {
                origin: [0.0, 0.0, 0.0],
                root_index: 0,
                world_size: 4.0,
                voxel_size: 2.0,
                svo_depth: 1,
            },
            ChunkDraw {
                origin: [0.0, 0.0, 1.0],
                root_index: root_b as i32,
                world_size: 1.0,
                voxel_size: 1.0,
                svo_depth: 0,
            },
        ];
        let origin = [0.5, 0.5, -1.0];
        let direction = [0.0, 0.0, 1.0];
        let reference = trace_svo(&atlas, &chunks, origin, direction, 10.0, false)
            .expect("reference must hit root B");
        let optimized = trace_svo(&atlas, &chunks, origin, direction, 10.0, true)
            .expect("optimized path must hit root B");
        assert_eq!(reference, optimized);
        assert!((optimized.t - 2.0).abs() < 1.0e-6, "hit={optimized:?}");
    }

    #[test]
    fn allocation_free_segment_query_matches_nearest_hit_visibility() {
        let mut atlas = Vec::new();
        atlas.extend([1, 1, 0x00FF_FFFF, 0]);
        let chunks = [
            ChunkDraw {
                origin: [0.0, 0.0, 8.0],
                root_index: 0,
                world_size: 2.0,
                voxel_size: 2.0,
                svo_depth: 0,
            },
            ChunkDraw {
                origin: [0.0, 0.0, 2.0],
                root_index: 0,
                world_size: 2.0,
                voxel_size: 2.0,
                svo_depth: 0,
            },
        ];
        let origin = [1.0, 1.0, 0.0];
        let direction = [0.0, 0.0, 1.0];

        for max_t in [1.0, 2.0, 2.5, 20.0] {
            let reference = trace_svo(&atlas, &chunks, origin, direction, max_t, true).is_some();
            assert_eq!(
                svo_segment_is_occluded(&atlas, &chunks, origin, direction, max_t),
                reference,
                "visibility differs at max_t={max_t}"
            );
        }
    }

    #[test]
    fn segment_query_rejects_invalid_lengths_and_fails_closed_on_malformed_atlas() {
        let chunk = ChunkDraw {
            origin: [0.0; 3],
            root_index: 99,
            world_size: 4.0,
            voxel_size: 1.0,
            svo_depth: 2,
        };
        let origin = [1.0, 1.0, -1.0];
        let direction = [0.0, 0.0, 1.0];

        for max_t in [-1.0, 0.0, f32::NAN, f32::INFINITY] {
            assert!(!svo_segment_is_occluded(
                &[],
                &[chunk],
                origin,
                direction,
                max_t,
            ));
        }
        assert!(svo_segment_is_occluded(
            &[],
            &[chunk],
            origin,
            direction,
            10.0,
        ));
    }
}
