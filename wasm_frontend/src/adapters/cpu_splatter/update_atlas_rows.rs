//! Validate and apply one independent pooled-atlas row update.
//!
//! Streaming supplies a complete row-aligned chunk slot whose child pointers
//! were already rebased by [`crate::application::atlas::AtlasPool`]. This
//! boundary validates that independence before mutating persistent renderer
//! state, then rebuilds only the corresponding prepared MIP slice.

use std::ops::Range;

use crate::application::atlas::ROW_TEXELS;

use super::atlas::{MipNode, rebuild_mips_in_node_range};

const WORDS_PER_NODE: usize = 4;
const WORDS_PER_ROW: usize = ROW_TEXELS * WORDS_PER_NODE;

#[derive(Debug, Clone)]
struct ValidatedRowUpdate {
    word_range: Range<usize>,
    node_range: Range<usize>,
}

/// Applies a complete independent slot update without reallocating the atlas.
pub(super) fn update_prepared_atlas_rows(
    atlas: &mut [u32],
    mips: &mut [MipNode],
    mip_visit_scratch: &mut Vec<bool>,
    first_row: u32,
    texels: &[u32],
) -> bool {
    let Some(update) = validate_row_update(atlas, mips, first_row, texels) else {
        return false;
    };

    atlas[update.word_range].copy_from_slice(texels);
    let rebuilt = rebuild_mips_in_node_range(atlas, mips, update.node_range, mip_visit_scratch);
    debug_assert!(rebuilt, "validated row range must always rebuild");
    rebuilt
}

/// Validation is deliberately complete before the first persistent write.
fn validate_row_update(
    atlas: &[u32],
    mips: &[MipNode],
    first_row: u32,
    texels: &[u32],
) -> Option<ValidatedRowUpdate> {
    if atlas.is_empty()
        || texels.is_empty()
        || atlas.len() % WORDS_PER_ROW != 0
        || texels.len() % WORDS_PER_ROW != 0
        || mips.len().checked_mul(WORDS_PER_NODE)? != atlas.len()
    {
        return None;
    }

    let first_row = usize::try_from(first_row).ok()?;
    let word_start = first_row.checked_mul(WORDS_PER_ROW)?;
    let word_end = word_start.checked_add(texels.len())?;
    if word_end > atlas.len() {
        return None;
    }

    let node_range = word_start / WORDS_PER_NODE..word_end / WORDS_PER_NODE;
    validate_independent_node_block(texels, node_range.clone())?;
    Some(ValidatedRowUpdate {
        word_range: word_start..word_end,
        node_range,
    })
}

/// Every internal node in a streamed slot must keep all eight child slots
/// inside that same update. This rejects partial-slot rows and corrupted or
/// unrebased pointers before they can invalidate an unrelated slot's MIPs.
fn validate_independent_node_block(texels: &[u32], global_nodes: Range<usize>) -> Option<()> {
    for node in texels.chunks_exact(WORDS_PER_NODE) {
        match node[0] {
            1 => {}
            0 => {
                if node[2] & !0xff != 0 {
                    return None;
                }
                let child_base = usize::try_from(node[1]).ok()?;
                let child_end = child_base.checked_add(8)?;
                if child_base < global_nodes.start || child_end > global_nodes.end {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::cpu_splatter::atlas::build_mips;

    const AIR: [u32; 4] = [1, 0, 0, 0];

    fn empty_rows(row_count: usize) -> Vec<u32> {
        AIR.repeat(ROW_TEXELS * row_count)
    }

    /// One root-last tree in a row-sized pool slot. All pointers are global,
    /// exactly like `AtlasPool::rebased_block` produces.
    fn tree_slot(first_node: usize, color: u32, light: u8) -> Vec<u32> {
        let mut slot = empty_rows(1);
        for child in 0..8 {
            let word = child * WORDS_PER_NODE;
            slot[word] = 1;
            slot[word + 1] = 1;
            slot[word + 2] = color;
            slot[word + 3] = u32::from(light)
                | (u32::from(light) << 16)
                | (u32::from(light) << 20)
                | (u32::from(light) << 24);
        }
        let root_word = 8 * WORDS_PER_NODE;
        slot[root_word] = 0;
        slot[root_word + 1] = first_node as u32;
        slot[root_word + 2] = 0xff;
        slot
    }

    #[test]
    fn partial_slot_rebuild_matches_the_full_reference_without_touching_other_rows() {
        let first_slot = tree_slot(0, 0xff0000, 3);
        let old_second = tree_slot(ROW_TEXELS, 0x0000ff, 4);
        let new_second = tree_slot(ROW_TEXELS, 0x00ff00, 11);
        let mut atlas = [first_slot.as_slice(), old_second.as_slice()].concat();
        let mut mips = build_mips(&atlas);
        let untouched_atlas = atlas[..WORDS_PER_ROW].to_vec();
        let untouched_mips = mips[..ROW_TEXELS].to_vec();
        let mut scratch = Vec::new();

        assert!(update_prepared_atlas_rows(
            &mut atlas,
            &mut mips,
            &mut scratch,
            1,
            &new_second,
        ));

        let expected_atlas = [first_slot.as_slice(), new_second.as_slice()].concat();
        assert_eq!(atlas, expected_atlas);
        assert_eq!(mips, build_mips(&expected_atlas));
        assert_eq!(&atlas[..WORDS_PER_ROW], untouched_atlas);
        assert_eq!(&mips[..ROW_TEXELS], untouched_mips);
        assert_eq!(scratch.len(), ROW_TEXELS);
    }

    #[test]
    fn malformed_or_non_independent_updates_are_rejected_before_mutation() {
        let slot = tree_slot(0, 0x112233, 5);
        let mut atlas = slot.clone();
        let mut mips = build_mips(&atlas);
        let original_atlas = atlas.clone();
        let original_mips = mips.clone();
        let mut scratch = Vec::new();

        let mut invalid_type = slot.clone();
        invalid_type[0] = 7;
        let mut cross_slot_pointer = slot.clone();
        cross_slot_pointer[8 * WORDS_PER_NODE + 1] = ROW_TEXELS as u32;

        for (first_row, update) in [
            (0, &slot[..slot.len() - 1]),
            (1, slot.as_slice()),
            (0, invalid_type.as_slice()),
            (0, cross_slot_pointer.as_slice()),
        ] {
            assert!(!update_prepared_atlas_rows(
                &mut atlas,
                &mut mips,
                &mut scratch,
                first_row,
                update,
            ));
            assert_eq!(atlas, original_atlas);
            assert_eq!(mips, original_mips);
        }
    }

    #[test]
    fn no_atlas_and_non_row_aligned_atlases_fail_safely() {
        let slot = tree_slot(0, 0x445566, 6);
        let mut scratch = Vec::new();
        assert!(!update_prepared_atlas_rows(
            &mut [],
            &mut [],
            &mut scratch,
            0,
            &slot,
        ));

        let mut short_atlas = vec![1, 0, 0, 0];
        let mut short_mips = build_mips(&short_atlas);
        assert!(!update_prepared_atlas_rows(
            &mut short_atlas,
            &mut short_mips,
            &mut scratch,
            0,
            &slot,
        ));

        let mut atlas_without_prepared_mips = slot.clone();
        let unchanged = atlas_without_prepared_mips.clone();
        assert!(!update_prepared_atlas_rows(
            &mut atlas_without_prepared_mips,
            &mut [],
            &mut scratch,
            0,
            &slot,
        ));
        assert_eq!(atlas_without_prepared_mips, unchanged);

        let mut valid_atlas = slot.clone();
        let mut valid_mips = build_mips(&valid_atlas);
        assert!(!update_prepared_atlas_rows(
            &mut valid_atlas,
            &mut valid_mips,
            &mut scratch,
            0,
            &[],
        ));
    }
}
