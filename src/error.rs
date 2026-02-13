//! Error types for automatic differentiation.

use thiserror::Error;

/// Errors that can occur during automatic differentiation.
#[derive(Debug, Clone, Error)]
pub enum AutoDiffError {
    /// Division by zero.
    #[error("Division by zero: divisor value is {0}")]
    DivisionByZero(f64),

    /// Logarithm domain error: input must be positive.
    #[error("Logarithm undefined for non-positive value: {0}")]
    LogDomainError(f64),

    /// Square root domain error: input must be non-negative.
    #[error("Square root undefined for negative value: {0}")]
    SqrtDomainError(f64),

    /// Power domain error: base must be positive for non-integer exponents.
    #[error("Power undefined for base {base} with exponent {exponent}")]
    PowDomainError {
        /// The base value
        base: f64,
        /// The exponent value
        exponent: f64,
    },
}

/// Alias for Result with AutoDiffError.
pub type AutoDiffResult<T> = Result<T, AutoDiffError>;
