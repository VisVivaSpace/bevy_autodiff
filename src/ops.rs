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
//! let x = ad.var(2.0);
//! let y = ad.var(3.0);
//!
//! // Use operator overloading within a context
//! let result = with_context(&mut ad, || {
//!     x + y * x  // Returns a Var
//! });
//!
//! assert_eq!(ad.eval(result), 8.0); // 2 + 3*2 = 8
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
/// let x = ad.var(2.0);
/// let y = ad.var(3.0);
///
/// let f = with_context(&mut ad, || x + y);
/// assert_eq!(ad.eval(f), 5.0);
/// ```
pub fn with_context<F, R>(ad: &mut AutoDiff, f: F) -> R
where
    F: FnOnce() -> R,
{
    // Store the context
    let ptr = NonNull::new(ad as *mut AutoDiff).unwrap();
    CONTEXT.with(|ctx| {
        let old = ctx.borrow_mut().replace(ptr);

        // Execute the closure
        let result = f();

        // Restore the old context (or clear it)
        *ctx.borrow_mut() = old;

        result
    })
}

/// Gets the current AutoDiff context.
///
/// # Panics
///
/// Panics if called outside of a `with_context` block.
fn get_context() -> &'static mut AutoDiff {
    CONTEXT.with(|ctx| {
        ctx.borrow()
            .as_ref()
            .map(|ptr| unsafe { &mut *ptr.as_ptr() })
            .expect("Operator overloading requires an active AutoDiff context. Use with_context().")
    })
}

// ============================================================================
// Var + Var
// ============================================================================

impl Add for Var {
    type Output = Var;

    fn add(self, rhs: Var) -> Var {
        get_context().add(self, rhs)
    }
}

impl Add<f64> for Var {
    type Output = Var;

    fn add(self, rhs: f64) -> Var {
        let ctx = get_context();
        let c = ctx.constant(rhs);
        ctx.add(self, c)
    }
}

impl Add<Var> for f64 {
    type Output = Var;

    fn add(self, rhs: Var) -> Var {
        let ctx = get_context();
        let c = ctx.constant(self);
        ctx.add(c, rhs)
    }
}

// ============================================================================
// Var - Var
// ============================================================================

impl Sub for Var {
    type Output = Var;

    fn sub(self, rhs: Var) -> Var {
        get_context().sub(self, rhs)
    }
}

impl Sub<f64> for Var {
    type Output = Var;

    fn sub(self, rhs: f64) -> Var {
        let ctx = get_context();
        let c = ctx.constant(rhs);
        ctx.sub(self, c)
    }
}

impl Sub<Var> for f64 {
    type Output = Var;

    fn sub(self, rhs: Var) -> Var {
        let ctx = get_context();
        let c = ctx.constant(self);
        ctx.sub(c, rhs)
    }
}

// ============================================================================
// Var * Var
// ============================================================================

impl Mul for Var {
    type Output = Var;

    fn mul(self, rhs: Var) -> Var {
        get_context().mul(self, rhs)
    }
}

impl Mul<f64> for Var {
    type Output = Var;

    fn mul(self, rhs: f64) -> Var {
        let ctx = get_context();
        let c = ctx.constant(rhs);
        ctx.mul(self, c)
    }
}

impl Mul<Var> for f64 {
    type Output = Var;

    fn mul(self, rhs: Var) -> Var {
        let ctx = get_context();
        let c = ctx.constant(self);
        ctx.mul(c, rhs)
    }
}

// ============================================================================
// Var / Var
// ============================================================================

impl Div for Var {
    type Output = Var;

    fn div(self, rhs: Var) -> Var {
        get_context().div(self, rhs)
    }
}

impl Div<f64> for Var {
    type Output = Var;

    fn div(self, rhs: f64) -> Var {
        let ctx = get_context();
        let c = ctx.constant(rhs);
        ctx.div(self, c)
    }
}

impl Div<Var> for f64 {
    type Output = Var;

    fn div(self, rhs: Var) -> Var {
        let ctx = get_context();
        let c = ctx.constant(self);
        ctx.div(c, rhs)
    }
}

// ============================================================================
// -Var
// ============================================================================

impl Neg for Var {
    type Output = Var;

    fn neg(self) -> Var {
        get_context().neg(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        let result = with_context(&mut ad, || x + y);
        assert_eq!(ad.eval(result), 5.0);
    }

    #[test]
    fn test_add_var_scalar() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        let result = with_context(&mut ad, || x + 3.0);
        assert_eq!(ad.eval(result), 5.0);
    }

    #[test]
    fn test_add_scalar_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        let result = with_context(&mut ad, || 3.0 + x);
        assert_eq!(ad.eval(result), 5.0);
    }

    #[test]
    fn test_sub_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.var(3.0);

        let result = with_context(&mut ad, || x - y);
        assert_eq!(ad.eval(result), 2.0);
    }

    #[test]
    fn test_mul_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        let result = with_context(&mut ad, || x * y);
        assert_eq!(ad.eval(result), 6.0);
    }

    #[test]
    fn test_div_vars() {
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0);
        let y = ad.var(2.0);

        let result = with_context(&mut ad, || x / y);
        assert_eq!(ad.eval(result), 3.0);
    }

    #[test]
    fn test_neg_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);

        let result = with_context(&mut ad, || -x);
        assert_eq!(ad.eval(result), -5.0);
    }

    #[test]
    fn test_complex_expression() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // f = (x + y) * (x - y) = x² - y² = 4 - 9 = -5
        let result = with_context(&mut ad, || (x + y) * (x - y));
        assert_eq!(ad.eval(result), -5.0);
    }

    #[test]
    fn test_nested_context() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        let outer = with_context(&mut ad, || {
            let inner = x + 1.0;  // 3.0
            inner * 2.0  // 6.0
        });

        assert_eq!(ad.eval(outer), 6.0);
    }

    #[test]
    fn test_derivative_with_ops() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        // f = x² = x * x
        let f = with_context(&mut ad, || x * x);

        // f' = 2x = 4 at x=2
        let df = ad.derivative(f, x, 1);
        assert!((df - 4.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Operator overloading requires an active AutoDiff context")]
    fn test_panic_without_context() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // This should panic - no context active
        let _ = x + y;
    }
}
