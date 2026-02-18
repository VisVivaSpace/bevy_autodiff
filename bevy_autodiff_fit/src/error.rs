//! Error types for the fitting crate.

use std::fmt;

/// Errors that can occur during fitting and evaluation.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum FitError {
    /// Data arrays have different lengths.
    #[error("data arrays must have the same length (x: {x_len}, y: {y_len})")]
    LengthMismatch { x_len: usize, y_len: usize },

    /// Not enough data points for the requested fit.
    #[error("need at least {min} data points, got {got}")]
    InsufficientData { min: usize, got: usize },

    /// Breakpoints are not strictly increasing.
    #[error("breakpoints must be strictly increasing")]
    InvalidBreakpoints,

    /// Polynomial degree is too small.
    #[error("degree must be at least 1, got {0}")]
    InvalidDegree(usize),

    /// Segment index is out of range.
    #[error("segment index {index} out of range (have {count} segments)")]
    SegmentOutOfRange { index: usize, count: usize },

    /// Query point is outside the fitted domain.
    #[error("x = {x} is outside the fitted domain [{a}, {b}]")]
    OutOfDomain { x: String, a: String, b: String },

    /// 2D grid data has wrong dimensions.
    #[error("grid data has {rows} rows and {cols} columns, expected {expected_rows}×{expected_cols}")]
    GridDimensionMismatch {
        rows: usize,
        cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },

    /// An error from the underlying autodiff library.
    #[error("autodiff error: {0}")]
    AutoDiff(#[from] bevy_autodiff::AutoDiffError),
}

impl FitError {
    /// Create an OutOfDomain error from displayable values.
    pub(crate) fn out_of_domain(
        x: impl fmt::Display,
        a: impl fmt::Display,
        b: impl fmt::Display,
    ) -> Self {
        Self::OutOfDomain {
            x: x.to_string(),
            a: a.to_string(),
            b: b.to_string(),
        }
    }
}
