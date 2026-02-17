//! Numeric traits for automatic differentiation.
//!
//! Two traits form the hierarchy:
//!
//! - [`DiffNum`] — base trait for dual-use functions. Implemented for `f32`, `f64`
//!   (direct evaluation) and [`Var`](crate::Var) (graph construction). Functions
//!   decorated with `#[autodiff]` are generic over `T: DiffNum`.
//!
//! - [`Float`] — extends `DiffNum` for types that can be stored in computation graphs.
//!   Implemented for `f32` and `f64`. Users can implement this for custom numeric types.
//!   `Var` does **not** implement `Float` (it's a graph handle, not a number).

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::var::Var;

/// A numeric type that supports the operations needed for automatic differentiation.
///
/// Implemented for `f32`, `f64` (direct evaluation), and [`Var`] (graph construction).
///
/// # Example
///
/// ```
/// use bevy_autodiff::DiffNum;
///
/// // Works with any DiffNum type
/// fn quadratic<T: DiffNum>(x: T) -> T {
///     x * x + T::from_f64(2.0) * x + T::from_f64(1.0)
/// }
///
/// // Direct evaluation with f64
/// assert_eq!(quadratic(3.0_f64), 16.0);
///
/// // Direct evaluation with f32
/// assert_eq!(quadratic(3.0_f32), 16.0);
/// ```
pub trait DiffNum:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sized
{
    /// Create a value from an f64 literal.
    fn from_f64(v: f64) -> Self;

    /// The additive identity (0).
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// The multiplicative identity (1).
    fn one() -> Self {
        Self::from_f64(1.0)
    }

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn square(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn powf(self, exp: Self) -> Self;
    fn powi(self, n: i32) -> Self;

    /// Power using logarithmic differentiation (default: delegates to `powf`).
    ///
    /// For `Var`, this routes to [`pow_log`](crate::ops::pow_log) which avoids
    /// catastrophic cancellation in second-order derivatives.
    fn pow_log(self, exp: Self) -> Self {
        self.powf(exp)
    }

    /// Integer power using logarithmic differentiation (default: delegates to `powi`).
    fn powi_log(self, n: i32) -> Self {
        self.powi(n)
    }

    /// Float power using logarithmic differentiation (default: delegates to `powf`).
    fn powf_log(self, exp: Self) -> Self {
        self.powf(exp)
    }

    /// Division using logarithmic differentiation (default: delegates to `/`).
    ///
    /// For `Var`, this routes to [`div_log`](crate::ops::div_log) which avoids
    /// catastrophic cancellation in second-order derivatives.
    fn div_log(self, rhs: Self) -> Self {
        self / rhs
    }
}

/// A numeric type that can be stored in computation graphs.
///
/// Extends [`DiffNum`] with conversion and validation methods needed by
/// [`AutoDiff`](crate::AutoDiff) and [`CompiledGraph`](crate::CompiledGraph).
///
/// Implemented for `f32` and `f64`. Users can implement this for custom numeric
/// types (e.g., interval arithmetic, extended precision).
///
/// [`Var`](crate::Var) does **not** implement `Float` — it's a graph handle, not a number.
pub trait Float: DiffNum + PartialEq + std::fmt::Display + Debug + Send + Sync + 'static {
    /// Convert to f64 (for WGSL code generation and GPU constant conversion).
    fn to_f64(self) -> f64;

    /// Returns true if the value is finite (not NaN or infinity).
    fn is_finite(self) -> bool;
}

impl Float for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl Float for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

// =============================================================================
// f64 implementation
// =============================================================================

impl DiffNum for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn square(self) -> Self {
        self * self
    }
    #[inline]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f64::atan(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        f64::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f64::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f64::atanh(self)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        f64::powf(self, exp)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
}

// =============================================================================
// f32 implementation
// =============================================================================

impl DiffNum for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f32::tan(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f32::ln(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn square(self) -> Self {
        self * self
    }
    #[inline]
    fn sinh(self) -> Self {
        f32::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f32::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f32::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f32::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f32::atan(self)
    }
    #[inline]
    fn asinh(self) -> Self {
        f32::asinh(self)
    }
    #[inline]
    fn acosh(self) -> Self {
        f32::acosh(self)
    }
    #[inline]
    fn atanh(self) -> Self {
        f32::atanh(self)
    }
    #[inline]
    fn powf(self, exp: Self) -> Self {
        f32::powf(self, exp)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
}

// =============================================================================
// Var implementation — delegates to ops.rs free functions
//
// Var is always f64-only. All operations require an active `with_context` scope.
// Using Var DiffNum methods outside `with_context` will panic.
// =============================================================================

impl DiffNum for Var {
    fn from_f64(v: f64) -> Self {
        crate::ops::with_current_context(|ctx| ctx.constant(v))
    }
    fn sin(self) -> Self {
        crate::ops::sin(self)
    }
    fn cos(self) -> Self {
        crate::ops::cos(self)
    }
    fn tan(self) -> Self {
        crate::ops::tan(self)
    }
    fn exp(self) -> Self {
        crate::ops::exp(self)
    }
    fn ln(self) -> Self {
        crate::ops::ln(self)
    }
    fn sqrt(self) -> Self {
        crate::ops::sqrt(self)
    }
    fn square(self) -> Self {
        crate::ops::square(self)
    }
    fn sinh(self) -> Self {
        crate::ops::sinh(self)
    }
    fn cosh(self) -> Self {
        crate::ops::cosh(self)
    }
    fn tanh(self) -> Self {
        crate::ops::tanh(self)
    }
    fn asin(self) -> Self {
        crate::ops::asin(self)
    }
    fn acos(self) -> Self {
        crate::ops::acos(self)
    }
    fn atan(self) -> Self {
        crate::ops::atan(self)
    }
    fn asinh(self) -> Self {
        crate::ops::asinh(self)
    }
    fn acosh(self) -> Self {
        crate::ops::acosh(self)
    }
    fn atanh(self) -> Self {
        crate::ops::atanh(self)
    }
    fn powf(self, exp: Self) -> Self {
        crate::ops::pow(self, exp)
    }
    fn powi(self, n: i32) -> Self {
        crate::ops::powi(self, n)
    }
    fn pow_log(self, exp: Self) -> Self {
        crate::ops::pow_log(self, exp)
    }
    fn powi_log(self, n: i32) -> Self {
        crate::ops::powi_log(self, n)
    }
    fn powf_log(self, exp: Self) -> Self {
        crate::ops::pow_log(self, exp)
    }
    fn div_log(self, rhs: Self) -> Self {
        crate::ops::div_log(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_basic_ops() {
        assert_eq!(<f64 as DiffNum>::from_f64(3.14), 3.14_f64);
        assert_eq!(DiffNum::sin(0.0_f64), 0.0);
        assert_eq!(DiffNum::cos(0.0_f64), 1.0);
        assert_eq!(DiffNum::exp(0.0_f64), 1.0);
        assert_eq!(DiffNum::ln(1.0_f64), 0.0);
        assert_eq!(DiffNum::sqrt(4.0_f64), 2.0);
        assert_eq!(DiffNum::square(3.0_f64), 9.0);
    }

    #[test]
    fn f32_basic_ops() {
        assert_eq!(<f32 as DiffNum>::from_f64(3.14), 3.14_f32);
        assert_eq!(DiffNum::sin(0.0_f32), 0.0_f32);
        assert_eq!(DiffNum::cos(0.0_f32), 1.0_f32);
        assert_eq!(DiffNum::square(3.0_f32), 9.0_f32);
    }

    #[test]
    fn f64_pow_ops() {
        assert_eq!(DiffNum::powf(2.0_f64, 3.0), 8.0);
        assert_eq!(DiffNum::powi(2.0_f64, 4), 16.0);
    }

    #[test]
    fn f64_log_variants_match_regular() {
        let x = 2.0_f64;
        let y = 3.0_f64;
        assert_eq!(DiffNum::pow_log(x, y), DiffNum::powf(x, y));
        assert_eq!(DiffNum::powi_log(x, 4), DiffNum::powi(x, 4));
        assert_eq!(DiffNum::powf_log(x, y), DiffNum::powf(x, y));
        assert_eq!(DiffNum::div_log(x, y), x / y);
    }

    #[test]
    fn generic_function_f64() {
        fn quadratic<T: DiffNum>(x: T) -> T {
            x * x + T::from_f64(2.0) * x + T::from_f64(1.0)
        }
        assert_eq!(quadratic(3.0_f64), 16.0);
        assert_eq!(quadratic(3.0_f32), 16.0_f32);
    }

    #[test]
    fn generic_function_var() {
        use crate::AutoDiff;
        use crate::ops::with_context;

        fn quadratic<T: DiffNum>(x: T) -> T {
            x * x + T::from_f64(2.0) * x + T::from_f64(1.0)
        }

        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(3.0).unwrap();
        let f = with_context(&mut ad, || quadratic(x));
        assert_eq!(ad.eval(f).unwrap(), 16.0);
    }
}
