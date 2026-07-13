//! A sampled Level 0 area plus the one-column halo needed for clean caps.

use crate::use_cases::backrooms_level::ColumnPlan;

/// Resolution-dependent samples of a resolution-independent world plan.
///
/// The halo is part of this type's invariant.  Callers cannot accidentally
/// voxelize a chunk without knowing its neighbours' ceiling heights, which is
/// what keeps separately generated requests sealed at their shared edge.
pub(crate) struct ColumnField {
    width: usize,
    depth: usize,
    columns: Vec<ColumnPlan>,
}

impl ColumnField {
    pub(crate) fn sample(
        width: usize,
        depth: usize,
        mut plan_at: impl FnMut(i64, i64) -> ColumnPlan,
    ) -> Self {
        let mut columns = Vec::with_capacity((width + 2) * (depth + 2));
        for z in -1..=depth as i64 {
            for x in -1..=width as i64 {
                columns.push(plan_at(x, z));
            }
        }
        Self {
            width,
            depth,
            columns,
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }

    pub(crate) fn depth(&self) -> usize {
        self.depth
    }

    /// Local coordinates include the halo, so `x` and `z` may be `-1` or
    /// exactly the corresponding output dimension.
    pub(crate) fn get(&self, x: i64, z: i64) -> &ColumnPlan {
        debug_assert!((-1..=self.width as i64).contains(&x));
        debug_assert!((-1..=self.depth as i64).contains(&z));
        &self.columns[((z + 1) as usize) * (self.width + 2) + (x + 1) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_owns_a_one_column_halo() {
        let field = ColumnField::sample(2, 3, |x, z| ColumnPlan::open((x + z + 10) as f32));
        assert_eq!(field.get(-1, -1).ceiling_units, 8.0);
        assert_eq!(field.get(2, 3).ceiling_units, 15.0);
    }
}
