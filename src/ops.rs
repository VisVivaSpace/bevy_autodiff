//! Operator overloading for `Var` type.
//!
//! This module provides `std::ops` trait implementations for convenient
//! mathematical notation. Due to Rust's orphan rules and the need for
//! mutable `AutoDiff` access, these implementations use a thread-local
//! context approach.
//!
//! # Usage
//!
//! ```
//! use bevy_autodiff::AutoDiff;
//! use bevy_autodiff::ops::with_context;
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(2.0).unwrap();
//! let y = ad.var(3.0).unwrap();
//!
//! // Use operator overloading within a context
//! let result = with_context(&mut ad, || {
//!     x + y * x  // Returns a Var
//! });
//!
//! assert_eq!(ad.eval(result).unwrap(), 8.0); // 2 + 3*2 = 8
//! ```
//!
//! # Design Notes
//!
//! Operator overloading in automatic differentiation libraries is challenging
//! because operations need access to the computation graph (the `AutoDiff`
//! context). There are several approaches:
//!
//! 1. **Thread-local context** (this approach): Store a pointer to the current
//!    `AutoDiff` in thread-local storage. Safe but requires wrapping code in
//!    `with_context`.
//!
//! 2. **RefCell wrapping**: Store `Rc<RefCell<AutoDiff>>` in each `Var`. Adds
//!    overhead and complicates the API.
//!
//! 3. **Builder pattern**: Return operation builders that are evaluated lazily.
//!    More complex but safer.
//!
//! We use approach 1 for simplicity, with safety ensured by the `with_context`
//! function that properly sets and clears the thread-local context.

use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr::NonNull;

use crate::context::AutoDiff;
use crate::var::Var;

// Thread-local storage for the current AutoDiff context
thread_local! {
    static CONTEXT: RefCell<Option<NonNull<AutoDiff>>> = const { RefCell::new(None) };
}

/// Executes a closure with the given `AutoDiff` context active.
///
/// Within the closure, operator overloading on `Var` values will use
/// this context to perform operations.
///
/// # Safety
///
/// The context pointer is valid only during the closure execution.
/// The closure must not store the context or pass it to other threads.
///
/// # Panics
///
/// Panics if operator overloading is used outside of a `with_context` block.
///
/// # Example
///
/// ```
/// use bevy_autodiff::AutoDiff;
/// use bevy_autodiff::ops::with_context;
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(2.0).unwrap();
/// let y = ad.var(3.0).unwrap();
///
/// let f = with_context(&mut ad, || x + y);
/// assert_eq!(ad.eval(f).unwrap(), 5.0);
/// ```
pub fn with_context<F, R>(ad: &mut AutoDiff, f: F) -> R
where
    F: FnOnce() -> R,
{
    let ptr = NonNull::new(ad as *mut AutoDiff)
        .expect("internal: mutable reference cannot be null");
    CONTEXT.with(|ctx| {
        let old = ctx.borrow_mut().replace(ptr);
        // RAII guard restores the previous context even if f() panics,
        // preventing a dangling pointer from remaining in thread-local storage.
        let _guard = ContextGuard { old };
        f()
    })
}

/// RAII guard that restores the previous context on drop (including panic unwind).
struct ContextGuard {
    old: Option<NonNull<AutoDiff>>,
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = self.old.take();
        });
    }
}

/// Executes `f` with the current AutoDiff context, confining the `&mut` lifetime.
///
/// # Panics
///
/// Panics if called outside of a `with_context` block.
fn with_current_context<F, R>(f: F) -> R
where
    F: FnOnce(&mut AutoDiff) -> R,
{
    CONTEXT.with(|ctx| {
        let ptr = ctx
            .borrow()
            .expect("Operator overloading requires an active AutoDiff context. Use with_context().");
        // SAFETY: The pointer is valid for the duration of the with_context() call.
        // We create a short-lived &mut that does not escape this closure.
        // Only one with_current_context call executes at a time because operator
        // trait methods run sequentially within a single expression.
        f(unsafe { &mut *ptr.as_ptr() })
    })
}

// ============================================================================
// Var + Var
// ============================================================================

impl Add for Var {
    type Output = Var;

    fn add(self, rhs: Var) -> Var {
        with_current_context(|ctx| ctx.add(self, rhs))
    }
}

impl Add<f64> for Var {
    type Output = Var;

    fn add(self, rhs: f64) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(rhs);
            ctx.add(self, c)
        })
    }
}

impl Add<Var> for f64 {
    type Output = Var;

    fn add(self, rhs: Var) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(self);
            ctx.add(c, rhs)
        })
    }
}

// ============================================================================
// Var - Var
// ============================================================================

impl Sub for Var {
    type Output = Var;

    fn sub(self, rhs: Var) -> Var {
        with_current_context(|ctx| ctx.sub(self, rhs))
    }
}

impl Sub<f64> for Var {
    type Output = Var;

    fn sub(self, rhs: f64) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(rhs);
            ctx.sub(self, c)
        })
    }
}

impl Sub<Var> for f64 {
    type Output = Var;

    fn sub(self, rhs: Var) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(self);
            ctx.sub(c, rhs)
        })
    }
}

// ============================================================================
// Var * Var
// ============================================================================

impl Mul for Var {
    type Output = Var;

    fn mul(self, rhs: Var) -> Var {
        with_current_context(|ctx| ctx.mul(self, rhs))
    }
}

impl Mul<f64> for Var {
    type Output = Var;

    fn mul(self, rhs: f64) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(rhs);
            ctx.mul(self, c)
        })
    }
}

impl Mul<Var> for f64 {
    type Output = Var;

    fn mul(self, rhs: Var) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(self);
            ctx.mul(c, rhs)
        })
    }
}

// ============================================================================
// Var / Var
// ============================================================================

impl Div for Var {
    type Output = Var;

    fn div(self, rhs: Var) -> Var {
        with_current_context(|ctx| ctx.div(self, rhs))
    }
}

impl Div<f64> for Var {
    type Output = Var;

    fn div(self, rhs: f64) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(rhs);
            ctx.div(self, c)
        })
    }
}

impl Div<Var> for f64 {
    type Output = Var;

    fn div(self, rhs: Var) -> Var {
        with_current_context(|ctx| {
            let c = ctx.constant(self);
            ctx.div(c, rhs)
        })
    }
}

// ============================================================================
// -Var
// ============================================================================

impl Neg for Var {
    type Output = Var;

    fn neg(self) -> Var {
        with_current_context(|ctx| ctx.neg(self))
    }
}

// ============================================================================
// Free functions for use inside with_context blocks
// ============================================================================

/// Computes sin(x) within the current AutoDiff context.
pub fn sin(x: Var) -> Var { with_current_context(|ctx| ctx.sin(x)) }

/// Computes cos(x) within the current AutoDiff context.
pub fn cos(x: Var) -> Var { with_current_context(|ctx| ctx.cos(x)) }

/// Computes tan(x) within the current AutoDiff context.
pub fn tan(x: Var) -> Var { with_current_context(|ctx| ctx.tan(x)) }

/// Computes exp(x) within the current AutoDiff context.
pub fn exp(x: Var) -> Var { with_current_context(|ctx| ctx.exp(x)) }

/// Computes ln(x) within the current AutoDiff context.
pub fn ln(x: Var) -> Var { with_current_context(|ctx| ctx.ln(x)) }

/// Computes sqrt(x) within the current AutoDiff context.
pub fn sqrt(x: Var) -> Var { with_current_context(|ctx| ctx.sqrt(x)) }

/// Computes sinh(x) within the current AutoDiff context.
pub fn sinh(x: Var) -> Var { with_current_context(|ctx| ctx.sinh(x)) }

/// Computes cosh(x) within the current AutoDiff context.
pub fn cosh(x: Var) -> Var { with_current_context(|ctx| ctx.cosh(x)) }

/// Computes tanh(x) within the current AutoDiff context.
pub fn tanh(x: Var) -> Var { with_current_context(|ctx| ctx.tanh(x)) }

/// Computes asin(x) within the current AutoDiff context.
pub fn asin(x: Var) -> Var { with_current_context(|ctx| ctx.asin(x)) }

/// Computes acos(x) within the current AutoDiff context.
pub fn acos(x: Var) -> Var { with_current_context(|ctx| ctx.acos(x)) }

/// Computes atan(x) within the current AutoDiff context.
pub fn atan(x: Var) -> Var { with_current_context(|ctx| ctx.atan(x)) }

/// Computes asinh(x) within the current AutoDiff context.
pub fn asinh(x: Var) -> Var { with_current_context(|ctx| ctx.asinh(x)) }

/// Computes acosh(x) within the current AutoDiff context.
pub fn acosh(x: Var) -> Var { with_current_context(|ctx| ctx.acosh(x)) }

/// Computes atanh(x) within the current AutoDiff context.
pub fn atanh(x: Var) -> Var { with_current_context(|ctx| ctx.atanh(x)) }

/// Computes x^2 within the current AutoDiff context.
pub fn square(x: Var) -> Var { with_current_context(|ctx| ctx.square(x)) }

/// Computes base^exp within the current AutoDiff context (both Var).
pub fn pow(base: Var, exp: Var) -> Var { with_current_context(|ctx| ctx.pow(base, exp)) }

/// Computes x^n within the current AutoDiff context (integer exponent).
pub fn powi(x: Var, n: i32) -> Var { with_current_context(|ctx| ctx.powi(x, n)) }

/// Computes x^p within the current AutoDiff context (float exponent).
pub fn powf(x: Var, p: f64) -> Var { with_current_context(|ctx| ctx.powf(x, p)) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let result = with_context(&mut ad, || x + y);
        assert_eq!(ad.eval(result).unwrap(), 5.0);
    }

    #[test]
    fn test_add_var_scalar() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

        let result = with_context(&mut ad, || x + 3.0);
        assert_eq!(ad.eval(result).unwrap(), 5.0);
    }

    #[test]
    fn test_add_scalar_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

        let result = with_context(&mut ad, || 3.0 + x);
        assert_eq!(ad.eval(result).unwrap(), 5.0);
    }

    #[test]
    fn test_sub_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let result = with_context(&mut ad, || x - y);
        assert_eq!(ad.eval(result).unwrap(), 2.0);
    }

    #[test]
    fn test_mul_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let result = with_context(&mut ad, || x * y);
        assert_eq!(ad.eval(result).unwrap(), 6.0);
    }

    #[test]
    fn test_div_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0).unwrap();
        let y = ad.var(2.0).unwrap();

        let result = with_context(&mut ad, || x / y);
        assert_eq!(ad.eval(result).unwrap(), 3.0);
    }

    #[test]
    fn test_neg_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();

        let result = with_context(&mut ad, || -x);
        assert_eq!(ad.eval(result).unwrap(), -5.0);
    }

    #[test]
    fn test_complex_expression() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        // f = (x + y) * (x - y) = x² - y² = 4 - 9 = -5
        let result = with_context(&mut ad, || (x + y) * (x - y));
        assert_eq!(ad.eval(result).unwrap(), -5.0);
    }

    #[test]
    fn test_nested_context() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

        let outer = with_context(&mut ad, || {
            let inner = x + 1.0; // 3.0
            inner * 2.0 // 6.0
        });

        assert_eq!(ad.eval(outer).unwrap(), 6.0);
    }

    #[test]
    #[should_panic(expected = "Operator overloading requires an active AutoDiff context")]
    fn test_panic_without_context() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        // This should panic - no context active
        let _ = x + y;
    }

    // --- Phase 4: Free function tests ---

    #[test]
    fn test_free_sin_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let result = with_context(&mut ad, || sin(x) + cos(x));
        assert!((ad.eval(result).unwrap() - 1.0).abs() < 1e-10); // sin(0) + cos(0) = 0 + 1
    }

    #[test]
    fn test_free_tan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(std::f64::consts::FRAC_PI_4).unwrap();

        let result = with_context(&mut ad, || tan(x));
        assert!((ad.eval(result).unwrap() - 1.0).abs() < 1e-10); // tan(π/4) = 1
    }

    #[test]
    fn test_free_exp_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

        let result = with_context(&mut ad, || ln(exp(x)));
        assert!((ad.eval(result).unwrap() - 2.0).abs() < 1e-10); // ln(exp(2)) = 2
    }

    #[test]
    fn test_free_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(9.0).unwrap();

        let result = with_context(&mut ad, || sqrt(x));
        assert!((ad.eval(result).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_sinh_cosh_tanh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let sh = with_context(&mut ad, || sinh(x));
        let ch = with_context(&mut ad, || cosh(x));
        let th = with_context(&mut ad, || tanh(x));
        assert!((ad.eval(sh).unwrap() - 0.0).abs() < 1e-10);
        assert!((ad.eval(ch).unwrap() - 1.0).abs() < 1e-10);
        assert!((ad.eval(th).unwrap() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_asin_acos_atan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();

        let a = with_context(&mut ad, || asin(x));
        let b = with_context(&mut ad, || acos(x));
        assert!((ad.eval(a).unwrap() - 0.5_f64.asin()).abs() < 1e-10);
        assert!((ad.eval(b).unwrap() - 0.5_f64.acos()).abs() < 1e-10);

        let y = ad.var(1.0).unwrap();
        let c = with_context(&mut ad, || atan(y));
        assert!((ad.eval(c).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    }

    #[test]
    fn test_free_asinh_acosh_atanh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();
        let y = ad.var(2.0).unwrap();
        let z = ad.var(0.5).unwrap();

        let a = with_context(&mut ad, || asinh(x));
        let b = with_context(&mut ad, || acosh(y));
        let c = with_context(&mut ad, || atanh(z));
        assert!((ad.eval(a).unwrap() - 0.0).abs() < 1e-10);
        assert!((ad.eval(b).unwrap() - 2.0_f64.acosh()).abs() < 1e-10);
        assert!((ad.eval(c).unwrap() - 0.5_f64.atanh()).abs() < 1e-10);
    }

    #[test]
    fn test_free_square() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();

        let result = with_context(&mut ad, || square(x));
        assert_eq!(ad.eval(result).unwrap(), 25.0);
    }

    #[test]
    fn test_free_pow_powi_powf() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let a = with_context(&mut ad, || pow(x, y));
        assert!((ad.eval(a).unwrap() - 8.0).abs() < 1e-10); // 2^3 = 8

        let b = with_context(&mut ad, || powi(x, 4));
        assert!((ad.eval(b).unwrap() - 16.0).abs() < 1e-10); // 2^4 = 16

        let c = with_context(&mut ad, || powf(x, 0.5));
        assert!((ad.eval(c).unwrap() - std::f64::consts::SQRT_2).abs() < 1e-10); // 2^0.5
    }

    #[test]
    fn test_free_composition() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();

        // sin(x) + cos(x) + exp(-x) at x=1
        let result = with_context(&mut ad, || sin(x) + cos(x) + exp(-x));
        let expected = 1.0_f64.sin() + 1.0_f64.cos() + (-1.0_f64).exp();
        assert!((ad.eval(result).unwrap() - expected).abs() < 1e-10);
    }
}
