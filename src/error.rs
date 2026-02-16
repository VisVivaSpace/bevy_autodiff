//! Error types for bevy_autodiff.

use thiserror::Error;

/// Errors that can occur during autodiff operations.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum AutoDiffError {
    /// Tried to create more than 64 input variables.
    #[error("cannot create more than 64 input variables (attempted input #{count})")]
    InputLimitExceeded { count: usize },

    /// `set_input()` called on a non-input variable.
    #[error("set_input called on a non-input variable")]
    NotAnInput,

    /// Variable missing its `Value` component (entity may be from a different context).
    #[error("variable missing Value component (entity may be from a different context)")]
    MissingValue,

    /// Input array count does not match the compiled graph.
    #[error("expected {expected} inputs, got {got}")]
    InputCountMismatch { expected: usize, got: usize },

    /// Requested partial derivative was not compiled.
    #[error("partial {requested:?} was not compiled (available: {available:?})")]
    PartialNotCompiled {
        requested: Vec<usize>,
        available: Vec<Vec<usize>>,
    },

    /// Cycle detected in computation graph.
    #[error("cycle detected in computation graph")]
    CycleDetected,

    /// Non-finite value cannot be emitted as a WGSL literal.
    #[error("cannot emit WGSL literal for non-finite value: {value}")]
    NonFiniteWgsl { value: f64 },

    /// Multi-index length does not match the number of inputs.
    #[error("multi-index length {got} does not match input count {expected}")]
    MultiIndexLengthMismatch { expected: usize, got: usize },
}
