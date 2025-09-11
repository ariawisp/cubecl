//! Stride compatibility helpers for preflight checks and host I/O planning.
//!
//! These utilities describe common stride patterns independently of any backend,
//! so hosts and higher layers can make informed choices and surface clearer errors.
//!
//! Strides are expressed in element units (not bytes). Element size may be used
//! by callers to convert to/from byte pitches as needed.

use alloc::vec;
use alloc::vec::Vec;

/// Canonical contiguous row-major strides for a given shape (in elements).
///
/// Example: shape [R, C] -> strides [C, 1]
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0; shape.len()];
    let mut s = 1usize;
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = s;
        s = s.saturating_mul(*dim.max(&1));
    }
    strides
}

/// A coarse description of a stride pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StridePattern {
    /// Fully contiguous row-major layout.
    Contiguous,
    /// 2D with inner-most contiguous axis and a row pitch (in elements) on the outer axis.
    /// `row_pitch_elems >= cols` is required.
    InnerContiguous2D {
        /// Pitch between consecutive rows in elements (not bytes).
        row_pitch_elems: usize,
    },
    /// Any other non-supported or irregular stride pattern.
    Other,
}

/// Describe the given shape/strides pair.
pub fn describe(shape: &[usize], strides: &[usize]) -> StridePattern {
    if shape.len() != strides.len() {
        return StridePattern::Other;
    }

    if strides == contiguous_strides(shape).as_slice() {
        return StridePattern::Contiguous;
    }

    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        let row_pitch = strides[0];
        let inner = strides[1];

        // Accept inner-most contiguous 2D with row pitch >= cols.
        if inner == 1 && row_pitch >= cols && rows > 0 && cols > 0 {
            return StridePattern::InnerContiguous2D {
                row_pitch_elems: row_pitch,
            };
        }
    }

    StridePattern::Other
}

/// Whether the given shape/strides is fully contiguous.
#[inline]
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    matches!(describe(shape, strides), StridePattern::Contiguous)
}

/// Whether the given shape/strides is rank-2 with inner-most contiguous axis and a row pitch.
#[inline]
pub fn is_inner_contiguous_2d(shape: &[usize], strides: &[usize]) -> bool {
    matches!(
        describe(shape, strides),
        StridePattern::InnerContiguous2D { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_for_1d_and_2d() {
        assert_eq!(contiguous_strides(&[8]), vec![1]);
        assert_eq!(contiguous_strides(&[2, 3]), vec![3, 1]);
        assert!(is_contiguous(&[8], &[1]));
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(!is_contiguous(&[2, 3], &[4, 1]));
    }

    #[test]
    fn inner_contiguous_2d_detection() {
        // 2D pitched: rows=4, cols=5, pitch=8 (in elems)
        let shape = [4, 5];
        let strides = [8, 1];
        assert!(is_inner_contiguous_2d(&shape, &strides));
        match describe(&shape, &strides) {
            StridePattern::InnerContiguous2D { row_pitch_elems } => assert_eq!(row_pitch_elems, 8),
            other => panic!("unexpected: {other:?}"),
        }
        // Not inner-contiguous
        assert!(!is_inner_contiguous_2d(&shape, &[8, 2]));
        // Pitch less than cols should not be accepted
        assert!(!is_inner_contiguous_2d(&shape, &[4, 1]));
    }

    #[test]
    fn describe_other() {
        // Rank 3 non-contiguous pattern should be Other
        assert!(matches!(
            describe(&[2, 3, 4], &[10, 4, 1]),
            StridePattern::Other
        ));
        // Mismatched lengths
        assert!(matches!(describe(&[2], &[]), StridePattern::Other));
    }
}
