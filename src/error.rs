//! Error types for Taylor coefficient computation.
//!
//! This module provides error types for mathematical domain errors that can occur
//! during Taylor coefficient computation, such as division by zero or logarithm
//! of non-positive values.

use thiserror::Error;

/// Errors that can occur during Taylor coefficient computation.
#[derive(Debug, Clone, Error)]
pub enum TaylorError {
    /// Division by zero: the denominator's leading coefficient is zero.
    #[error("Division by zero: divisor's leading coefficient is {0}")]
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

/// Alias for Result with TaylorError.
pub type TaylorResult<T> = Result<T, TaylorError>;
