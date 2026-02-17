//! Ergonomic macros for building computation graphs.
//!
//! The `expr!` macro transforms natural Rust expressions into
//! AutoDiff method calls, making graph construction more readable.
//!
//! # Example
//!
//! ```
//! use bevy_autodiff::{AutoDiff, expr};
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(2.0).unwrap();
//! let y = ad.var(3.0).unwrap();
//!
//! // Instead of: ad.add(ad.mul(x, x), ad.mul(x, y))
//! let f = expr!(ad, x * x + x * y);
//! assert_eq!(ad.eval(f).unwrap(), 10.0);  // 4 + 6 = 10
//! ```
//!
//! # Supported Operations
//!
//! - **Binary operators**: `+`, `-`, `*`, `/`
//! - **Unary negation**: `-x`
//! - **Functions**: `sin`, `cos`, `tan`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`, `tanh`,
//!   `asin`, `acos`, `atan`, `asinh`, `acosh`, `atanh`, `square`
//! - **Power functions**: `pow(base, exp)`, `powi(x, n)`, `powf(x, p)`
//! - **Logarithmic variants**: `pow_log(base, exp)`, `powi_log(x, n)`, `powf_log(x, p)`, `div_log(a, b)`
//! - **Float literals**: automatically wrapped as `ad.constant()`
//! - **Parentheses**: `(expr)` for grouping
//!
//! # Precedence
//!
//! Standard Rust precedence is followed:
//! - `*`, `/` bind tighter than `+`, `-`
//! - Unary `-` binds tighter than binary operators
//! - Function calls and parentheses bind tightest
//! - All operators are **left-associative**: `a - b - c` = `(a - b) - c`
//!
//! # Limitations
//!
//! - **f64 only**: Literals are cast to `f64` via `ad.constant(lit as f64)`,
//!   so `expr!` only works with `AutoDiff<f64>`. For `AutoDiff<f32>` or custom
//!   float types, use the builder API directly.
//! - Due to Rust macro limitations, some complex expressions may require
//!   explicit parentheses for correct parsing.

/// Transforms a natural Rust expression into AutoDiff method calls.
///
/// # Syntax
///
/// ```text
/// expr!(context, expression)
/// ```
///
/// where `context` is a mutable reference to an [`AutoDiff`](crate::AutoDiff) instance,
/// and `expression` is a mathematical expression using variables, operators, and functions.
///
/// # Examples
///
/// ## Simple Arithmetic
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(2.0).unwrap();
/// let y = ad.var(3.0).unwrap();
///
/// let sum = expr!(ad, x + y);
/// assert_eq!(ad.eval(sum).unwrap(), 5.0);
///
/// let product = expr!(ad, x * y);
/// assert_eq!(ad.eval(product).unwrap(), 6.0);
/// ```
///
/// ## With Literals
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(2.0).unwrap();
///
/// let f = expr!(ad, x * x + 3.0);
/// assert_eq!(ad.eval(f).unwrap(), 7.0);  // 4 + 3
/// ```
///
/// ## Transcendental Functions
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::<f64>::new();
/// let x = ad.var(0.0).unwrap();
///
/// let f = expr!(ad, sin(x) + cos(x));
/// assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);  // sin(0) + cos(0) = 0 + 1
/// ```
///
/// ## Complex Expressions
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(1.0).unwrap();
/// let y = ad.var(1.0).unwrap();
///
/// // Rosenbrock function at (1,1)
/// let f = expr!(ad, (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x));
/// assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
/// ```
#[macro_export]
macro_rules! expr {
    // Entry point: start at additive level with empty accumulator
    ($ad:ident, $($tokens:tt)+) => {
        $crate::expr!(@add_munch $ad, {}, $($tokens)+ @end)
    };

    // ============================================================
    // ADDITIVE LEVEL: Munch tokens looking for + or -
    // When an operator is found, evaluate the left side and enter
    // the fold loop for left-associative evaluation.
    // ============================================================

    // End marker: no + or - found, pass to multiplicative
    (@add_munch $ad:ident, {$($acc:tt)+}, @end) => {
        $crate::expr!(@mul_munch $ad, {}, $($acc)+ @end)
    };

    // Found + at additive level: evaluate left, enter fold
    (@add_munch $ad:ident, {$($left:tt)+}, + $($rest:tt)+) => {{
        let __lhs = $crate::expr!(@mul_munch $ad, {}, $($left)+ @end);
        $crate::expr!(@add_fold $ad, __lhs, +, {}, $($rest)+)
    }};

    // Found binary - (non-empty left): evaluate left, enter fold
    (@add_munch $ad:ident, {$($left:tt)+}, - $($rest:tt)+) => {{
        let __lhs = $crate::expr!(@mul_munch $ad, {}, $($left)+ @end);
        $crate::expr!(@add_fold $ad, __lhs, -, {}, $($rest)+)
    }};

    // Found unary - (empty left): pass to mul level
    (@add_munch $ad:ident, {}, - $($rest:tt)+) => {
        $crate::expr!(@mul_munch $ad, {}, - $($rest)+)
    };

    // Parenthesized group: add as single unit, continue
    (@add_munch $ad:ident, {$($acc:tt)*}, ($($inner:tt)*) $($rest:tt)*) => {
        $crate::expr!(@add_munch $ad, {$($acc)* ($($inner)*)}, $($rest)*)
    };

    // Any other token: add to accumulator, continue
    (@add_munch $ad:ident, {$($acc:tt)*}, $tok:tt $($rest:tt)*) => {
        $crate::expr!(@add_munch $ad, {$($acc)* $tok}, $($rest)*)
    };

    // ============================================================
    // ADDITIVE FOLD: Left-associative folding for + and -
    //
    // State: (@add_fold $ad, $lhs_value, $pending_op, {$rhs_accumulator}, remaining_tokens...)
    // $lhs_value is an already-evaluated expression (via let binding)
    // $pending_op is + or - (the operator to apply once we have the rhs term)
    // $rhs_accumulator collects tokens for the next multiplicative term
    // ============================================================

    // Helpers to apply the pending operator
    (@apply_add $ad:ident, $lhs:expr, +, $rhs:expr) => { $ad.add($lhs, $rhs) };
    (@apply_add $ad:ident, $lhs:expr, -, $rhs:expr) => { $ad.sub($lhs, $rhs) };

    // @end: evaluate accumulated rhs, apply pending op, done
    (@add_fold $ad:ident, $lhs:expr, $op:tt, {$($rhs:tt)+}, @end) => {{
        let __rhs = $crate::expr!(@mul_munch $ad, {}, $($rhs)+ @end);
        $crate::expr!(@apply_add $ad, $lhs, $op, __rhs)
    }};

    // Found +: evaluate accumulated rhs, apply pending op, continue with +
    (@add_fold $ad:ident, $lhs:expr, $op:tt, {$($rhs:tt)+}, + $($rest:tt)+) => {{
        let __rhs = $crate::expr!(@mul_munch $ad, {}, $($rhs)+ @end);
        let __result = $crate::expr!(@apply_add $ad, $lhs, $op, __rhs);
        $crate::expr!(@add_fold $ad, __result, +, {}, $($rest)+)
    }};

    // Found -: evaluate accumulated rhs, apply pending op, continue with -
    // Note: requires non-empty accumulator to distinguish from unary minus
    (@add_fold $ad:ident, $lhs:expr, $op:tt, {$($rhs:tt)+}, - $($rest:tt)+) => {{
        let __rhs = $crate::expr!(@mul_munch $ad, {}, $($rhs)+ @end);
        let __result = $crate::expr!(@apply_add $ad, $lhs, $op, __rhs);
        $crate::expr!(@add_fold $ad, __result, -, {}, $($rest)+)
    }};

    // Parenthesized group: accumulate as unit
    (@add_fold $ad:ident, $lhs:expr, $op:tt, {$($acc:tt)*}, ($($inner:tt)*) $($rest:tt)*) => {
        $crate::expr!(@add_fold $ad, $lhs, $op, {$($acc)* ($($inner)*)}, $($rest)*)
    };

    // Any other token: accumulate
    (@add_fold $ad:ident, $lhs:expr, $op:tt, {$($acc:tt)*}, $tok:tt $($rest:tt)*) => {
        $crate::expr!(@add_fold $ad, $lhs, $op, {$($acc)* $tok}, $($rest)*)
    };

    // ============================================================
    // MULTIPLICATIVE LEVEL: Munch tokens looking for * or /
    // Same left-associative fold pattern as additive level.
    // ============================================================

    // End marker: no * or / found, pass to unary
    (@mul_munch $ad:ident, {$($acc:tt)+}, @end) => {
        $crate::expr!(@unary $ad, $($acc)+)
    };

    // Found * at multiplicative level: evaluate left, enter fold
    (@mul_munch $ad:ident, {$($left:tt)+}, * $($rest:tt)+) => {{
        let __lhs = $crate::expr!(@unary $ad, $($left)+);
        $crate::expr!(@mul_fold $ad, __lhs, *, {}, $($rest)+)
    }};

    // Found / at multiplicative level: evaluate left, enter fold
    (@mul_munch $ad:ident, {$($left:tt)+}, / $($rest:tt)+) => {{
        let __lhs = $crate::expr!(@unary $ad, $($left)+);
        $crate::expr!(@mul_fold $ad, __lhs, /, {}, $($rest)+)
    }};

    // Parenthesized group: add to accumulator, continue
    (@mul_munch $ad:ident, {$($acc:tt)*}, ($($inner:tt)*) $($rest:tt)*) => {
        $crate::expr!(@mul_munch $ad, {$($acc)* ($($inner)*)}, $($rest)*)
    };

    // Any other token: add to accumulator, continue
    (@mul_munch $ad:ident, {$($acc:tt)*}, $tok:tt $($rest:tt)*) => {
        $crate::expr!(@mul_munch $ad, {$($acc)* $tok}, $($rest)*)
    };

    // ============================================================
    // MULTIPLICATIVE FOLD: Left-associative folding for * and /
    // ============================================================

    // Helpers to apply the pending operator
    (@apply_mul $ad:ident, $lhs:expr, *, $rhs:expr) => { $ad.mul($lhs, $rhs) };
    (@apply_mul $ad:ident, $lhs:expr, /, $rhs:expr) => { $ad.div($lhs, $rhs) };

    // @end: evaluate accumulated rhs, apply pending op, done
    (@mul_fold $ad:ident, $lhs:expr, $op:tt, {$($rhs:tt)+}, @end) => {{
        let __rhs = $crate::expr!(@unary $ad, $($rhs)+);
        $crate::expr!(@apply_mul $ad, $lhs, $op, __rhs)
    }};

    // Found *: evaluate accumulated rhs, apply pending op, continue with *
    (@mul_fold $ad:ident, $lhs:expr, $op:tt, {$($rhs:tt)+}, * $($rest:tt)+) => {{
        let __rhs = $crate::expr!(@unary $ad, $($rhs)+);
        let __result = $crate::expr!(@apply_mul $ad, $lhs, $op, __rhs);
        $crate::expr!(@mul_fold $ad, __result, *, {}, $($rest)+)
    }};

    // Found /: evaluate accumulated rhs, apply pending op, continue with /
    (@mul_fold $ad:ident, $lhs:expr, $op:tt, {$($rhs:tt)+}, / $($rest:tt)+) => {{
        let __rhs = $crate::expr!(@unary $ad, $($rhs)+);
        let __result = $crate::expr!(@apply_mul $ad, $lhs, $op, __rhs);
        $crate::expr!(@mul_fold $ad, __result, /, {}, $($rest)+)
    }};

    // Parenthesized group: accumulate
    (@mul_fold $ad:ident, $lhs:expr, $op:tt, {$($acc:tt)*}, ($($inner:tt)*) $($rest:tt)*) => {
        $crate::expr!(@mul_fold $ad, $lhs, $op, {$($acc)* ($($inner)*)}, $($rest)*)
    };

    // Any other token: accumulate
    (@mul_fold $ad:ident, $lhs:expr, $op:tt, {$($acc:tt)*}, $tok:tt $($rest:tt)*) => {
        $crate::expr!(@mul_fold $ad, $lhs, $op, {$($acc)* $tok}, $($rest)*)
    };

    // ============================================================
    // UNARY LEVEL: Handle - prefix
    // ============================================================

    (@unary $ad:ident, - $($rest:tt)+) => {{
        let inner = $crate::expr!(@unary $ad, $($rest)+);
        $ad.neg(inner)
    }};

    (@unary $ad:ident, $($rest:tt)+) => {
        $crate::expr!(@atom $ad, $($rest)+)
    };

    // ============================================================
    // ATOM LEVEL: Variables, literals, functions, parentheses
    // ============================================================

    // Parenthesized expression
    (@atom $ad:ident, ($($inner:tt)+)) => {
        $crate::expr!(@add_munch $ad, {}, $($inner)+ @end)
    };

    // Function calls — all 16 unary ops + square + pow variants

    (@atom $ad:ident, sin ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.sin(arg)
    }};

    (@atom $ad:ident, cos ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.cos(arg)
    }};

    (@atom $ad:ident, tan ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.tan(arg)
    }};

    (@atom $ad:ident, exp ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.exp(arg)
    }};

    (@atom $ad:ident, ln ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.ln(arg)
    }};

    (@atom $ad:ident, sqrt ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.sqrt(arg)
    }};

    (@atom $ad:ident, sinh ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.sinh(arg)
    }};

    (@atom $ad:ident, cosh ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.cosh(arg)
    }};

    (@atom $ad:ident, tanh ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.tanh(arg)
    }};

    (@atom $ad:ident, asin ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.asin(arg)
    }};

    (@atom $ad:ident, acos ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.acos(arg)
    }};

    (@atom $ad:ident, atan ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.atan(arg)
    }};

    (@atom $ad:ident, asinh ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.asinh(arg)
    }};

    (@atom $ad:ident, acosh ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.acosh(arg)
    }};

    (@atom $ad:ident, atanh ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.atanh(arg)
    }};

    (@atom $ad:ident, square ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.square(arg)
    }};

    // Binary function calls: both args are Var expressions
    (@atom $ad:ident, pow ($($args:tt)+)) => {
        $crate::expr!(@comma_munch $ad, pow, {}, $($args)+)
    };
    (@atom $ad:ident, pow_log ($($args:tt)+)) => {
        $crate::expr!(@comma_munch $ad, pow_log, {}, $($args)+)
    };
    (@atom $ad:ident, div_log ($($args:tt)+)) => {
        $crate::expr!(@comma_munch $ad, div_log, {}, $($args)+)
    };

    // Binary function calls: first arg is Var, second arg is a scalar (i32 or f64)
    (@atom $ad:ident, powi ($($args:tt)+)) => {
        $crate::expr!(@scalar_munch $ad, powi, {}, $($args)+)
    };
    (@atom $ad:ident, powf ($($args:tt)+)) => {
        $crate::expr!(@scalar_munch $ad, powf, {}, $($args)+)
    };
    (@atom $ad:ident, powi_log ($($args:tt)+)) => {
        $crate::expr!(@scalar_munch $ad, powi_log, {}, $($args)+)
    };
    (@atom $ad:ident, powf_log ($($args:tt)+)) => {
        $crate::expr!(@scalar_munch $ad, powf_log, {}, $($args)+)
    };

    // Unsupported function name — helpful error message
    (@atom $ad:ident, $unknown:ident ($($arg:tt)*)) => {
        compile_error!(concat!(
            "unsupported function `", stringify!($unknown), "` in expr! macro. ",
            "Supported: sin, cos, tan, exp, ln, sqrt, sinh, cosh, tanh, ",
            "asin, acos, atan, asinh, acosh, atanh, square, ",
            "pow, powi, powf, pow_log, powi_log, powf_log, div_log"
        ))
    };

    // Literals
    (@atom $ad:ident, $lit:literal) => {
        $ad.constant($lit as f64)
    };

    // Variables
    (@atom $ad:ident, $var:ident) => {
        $var
    };

    // ============================================================
    // COMMA-MUNCH: Unified two-argument function parsing
    // Finds comma separator in func(arg1, arg2)
    // ============================================================

    // Found comma: left of comma is first arg, right is second arg
    (@comma_munch $ad:ident, $func:ident, {$($lhs:tt)+}, , $($rhs:tt)+) => {{
        let __a = $crate::expr!(@add_munch $ad, {}, $($lhs)+ @end);
        let __b = $crate::expr!(@add_munch $ad, {}, $($rhs)+ @end);
        $ad.$func(__a, __b)
    }};

    // Parenthesized group in function args: accumulate as unit
    (@comma_munch $ad:ident, $func:ident, {$($acc:tt)*}, ($($inner:tt)*) $($rest:tt)*) => {
        $crate::expr!(@comma_munch $ad, $func, {$($acc)* ($($inner)*)}, $($rest)*)
    };

    // Any other token in function args: accumulate and continue
    (@comma_munch $ad:ident, $func:ident, {$($acc:tt)*}, $tok:tt $($rest:tt)*) => {
        $crate::expr!(@comma_munch $ad, $func, {$($acc)* $tok}, $($rest)*)
    };

    // ============================================================
    // SCALAR-MUNCH: Two-argument functions where first arg is Var,
    // second arg is a scalar literal (i32 for powi, f64 for powf)
    // ============================================================

    // Found comma: left is Var expression, right is scalar literal
    (@scalar_munch $ad:ident, $func:ident, {$($lhs:tt)+}, , $($rhs:tt)+) => {{
        let __a = $crate::expr!(@add_munch $ad, {}, $($lhs)+ @end);
        $ad.$func(__a, $($rhs)+)
    }};

    // Parenthesized group: accumulate as unit
    (@scalar_munch $ad:ident, $func:ident, {$($acc:tt)*}, ($($inner:tt)*) $($rest:tt)*) => {
        $crate::expr!(@scalar_munch $ad, $func, {$($acc)* ($($inner)*)}, $($rest)*)
    };

    // Any other token: accumulate and continue
    (@scalar_munch $ad:ident, $func:ident, {$($acc:tt)*}, $tok:tt $($rest:tt)*) => {
        $crate::expr!(@scalar_munch $ad, $func, {$($acc)* $tok}, $($rest)*)
    };
}

#[cfg(test)]
mod tests {
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_expr_simple_add() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let f = expr!(ad, x + y);
        assert_eq!(ad.eval(f).unwrap(), 5.0);
    }

    #[test]
    fn test_expr_simple_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let f = expr!(ad, x - y);
        assert_eq!(ad.eval(f).unwrap(), 2.0);
    }

    #[test]
    fn test_expr_simple_mul() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        let y = ad.var(5.0).unwrap();

        let f = expr!(ad, x * y);
        assert_eq!(ad.eval(f).unwrap(), 20.0);
    }

    #[test]
    fn test_expr_simple_div() {
        let mut ad = AutoDiff::new();
        let x = ad.var(10.0).unwrap();
        let y = ad.var(2.0).unwrap();

        let f = expr!(ad, x / y);
        assert_eq!(ad.eval(f).unwrap(), 5.0);
    }

    #[test]
    fn test_expr_with_literal() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

        let f = expr!(ad, x * x + 3.0);
        assert_eq!(ad.eval(f).unwrap(), 7.0);
    }

    #[test]
    fn test_expr_literal_first() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();

        let f = expr!(ad, 2.0 * x);
        assert_eq!(ad.eval(f).unwrap(), 6.0);
    }

    #[test]
    fn test_expr_sin() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let f = expr!(ad, sin(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let f = expr!(ad, cos(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sin_plus_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let f = expr!(ad, sin(x) + cos(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_exp() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let f = expr!(ad, exp(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();

        let f = expr!(ad, ln(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();

        let f = expr!(ad, sqrt(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sinh_cosh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        let sh = expr!(ad, sinh(x));
        let ch = expr!(ad, cosh(x));
        assert_relative_eq!(ad.eval(sh).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(ch).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_pow() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let f = expr!(ad, pow(x, y));
        assert_relative_eq!(ad.eval(f).unwrap(), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_nested_parens() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        // (x + y) * (x - y) = x² - y² = 4 - 9 = -5
        let f = expr!(ad, (x + y) * (x - y));
        assert_eq!(ad.eval(f).unwrap(), -5.0);
    }

    #[test]
    fn test_expr_unary_neg() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();

        let f = expr!(ad, -x);
        assert_eq!(ad.eval(f).unwrap(), -5.0);
    }

    #[test]
    fn test_expr_rosenbrock() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let y = ad.var(1.0).unwrap();

        // Rosenbrock at (1,1) should be 0
        // f = (1-x)² + 100(y-x²)²
        let f = expr!(
            ad,
            (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
        );
        assert_relative_eq!(ad.eval(f).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_integer_literal() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();

        // Integer literals should work too (cast to f64)
        let f = expr!(ad, x + 3);
        assert_eq!(ad.eval(f).unwrap(), 5.0);
    }

    #[test]
    fn test_expr_multiple_same_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();

        // f = x * x * x = x³ = 27
        let f = expr!(ad, x * x * x);
        assert_eq!(ad.eval(f).unwrap(), 27.0);
    }

    #[test]
    fn test_expr_function_composition() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0).unwrap();

        // f = sin(cos(x)) = sin(1) at x=0
        let f = expr!(ad, sin(cos(x)));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0_f64.sin(), epsilon = 1e-10);
    }

    // --- Associativity tests ---

    #[test]
    fn test_expr_subtraction_left_associative() {
        let mut ad = AutoDiff::new();
        // 10 - 3 - 2 should be (10 - 3) - 2 = 5, not 10 - (3 - 2) = 9
        let f = expr!(ad, 10.0 - 3.0 - 2.0);
        assert_eq!(ad.eval(f).unwrap(), 5.0);
    }

    #[test]
    fn test_expr_division_left_associative() {
        let mut ad = AutoDiff::new();
        // 24 / 4 / 2 should be (24 / 4) / 2 = 3, not 24 / (4 / 2) = 12
        let f = expr!(ad, 24.0 / 4.0 / 2.0);
        assert_eq!(ad.eval(f).unwrap(), 3.0);
    }

    #[test]
    fn test_expr_mixed_add_sub_chain() {
        let mut ad = AutoDiff::new();
        let a = ad.var(10.0).unwrap();
        let b = ad.var(3.0).unwrap();
        let c = ad.var(2.0).unwrap();
        let d = ad.var(1.0).unwrap();

        // a - b + c - d = (((10 - 3) + 2) - 1) = 8
        let f = expr!(ad, a - b + c - d);
        assert_eq!(ad.eval(f).unwrap(), 8.0);
    }

    #[test]
    fn test_expr_mixed_mul_div_chain() {
        let mut ad = AutoDiff::new();
        let a = ad.var(24.0).unwrap();
        let b = ad.var(4.0).unwrap();
        let c = ad.var(3.0).unwrap();
        let d = ad.var(2.0).unwrap();

        // a / b * c / d = (((24 / 4) * 3) / 2) = 9
        let f = expr!(ad, a / b * c / d);
        assert_eq!(ad.eval(f).unwrap(), 9.0);
    }

    #[test]
    fn test_expr_sub_then_add() {
        let mut ad = AutoDiff::new();
        // 5 - 3 + 1 should be (5 - 3) + 1 = 3
        let f = expr!(ad, 5.0 - 3.0 + 1.0);
        assert_eq!(ad.eval(f).unwrap(), 3.0);
    }

    #[test]
    fn test_expr_div_then_mul() {
        let mut ad = AutoDiff::new();
        // 6 / 2 * 3 should be (6 / 2) * 3 = 9
        let f = expr!(ad, 6.0 / 2.0 * 3.0);
        assert_eq!(ad.eval(f).unwrap(), 9.0);
    }

    #[test]
    fn test_expr_unary_neg_after_operator() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        let y = ad.var(2.0).unwrap();

        // x + -y = 3 + (-2) = 1
        let f = expr!(ad, x + -y);
        assert_eq!(ad.eval(f).unwrap(), 1.0);
    }

    #[test]
    fn test_expr_double_negation() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let y = ad.var(3.0).unwrap();

        // x - -y = x - (-y) = 5 - (-3) = 8
        let f = expr!(ad, x - -y);
        assert_eq!(ad.eval(f).unwrap(), 8.0);
    }

    // --- Phase 2: Function coverage tests ---

    #[test]
    fn test_expr_tan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = expr!(ad, tan(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 0.5_f64.tan(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_tanh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = expr!(ad, tanh(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0_f64.tanh(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_asin() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = expr!(ad, asin(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 0.5_f64.asin(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_acos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = expr!(ad, acos(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 0.5_f64.acos(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_atan() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = expr!(ad, atan(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0_f64.atan(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_asinh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0).unwrap();
        let f = expr!(ad, asinh(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.0_f64.asinh(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_acosh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let f = expr!(ad, acosh(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 2.0_f64.acosh(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_atanh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.5).unwrap();
        let f = expr!(ad, atanh(x));
        assert_relative_eq!(ad.eval(f).unwrap(), 0.5_f64.atanh(), epsilon = 1e-10);
    }

    #[test]
    fn test_expr_square() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0).unwrap();
        let f = expr!(ad, square(x));
        assert_eq!(ad.eval(f).unwrap(), 25.0);
    }

    #[test]
    fn test_expr_pow_complex_args() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(1.0).unwrap();
        // pow(x + 1, y * 2) = pow(3, 2) = 9
        let f = expr!(ad, pow(x + 1.0, y * 2.0));
        assert_relative_eq!(ad.eval(f).unwrap(), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_pow_with_literal_base() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        // pow(2.0, x) = 2^3 = 8
        let f = expr!(ad, pow(2.0, x));
        assert_relative_eq!(ad.eval(f).unwrap(), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_powi() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        // powi(x, 2) = 3^2 = 9
        let f = expr!(ad, powi(x, 2));
        assert_relative_eq!(ad.eval(f).unwrap(), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_powf() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        // powf(x, 0.5) = sqrt(4) = 2
        let f = expr!(ad, powf(x, 0.5));
        assert_relative_eq!(ad.eval(f).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_pow_log() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();

        let f = expr!(ad, pow_log(x, y));
        assert_relative_eq!(ad.eval(f).unwrap(), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_pow_log_complex_args() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        // pow_log(x + 1, 2.0) = 3^2 = 9
        let f = expr!(ad, pow_log(x + 1.0, 2.0));
        assert_relative_eq!(ad.eval(f).unwrap(), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_powi_log() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0).unwrap();
        // powi_log(x, 2) = 3^2 = 9
        let f = expr!(ad, powi_log(x, 2));
        assert_relative_eq!(ad.eval(f).unwrap(), 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_powf_log() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0).unwrap();
        // powf_log(x, 0.5) = sqrt(4) = 2
        let f = expr!(ad, powf_log(x, 0.5));
        assert_relative_eq!(ad.eval(f).unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_div_log() {
        let mut ad = AutoDiff::new();
        let x = ad.var(6.0).unwrap();
        let y = ad.var(2.0).unwrap();

        let f = expr!(ad, div_log(x, y));
        assert_eq!(ad.eval(f).unwrap(), 3.0);
    }

    #[test]
    fn test_expr_div_log_complex_args() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        // div_log(x * y, x + y) = 6 / 5 = 1.2
        let f = expr!(ad, div_log(x * y, x + y));
        assert_relative_eq!(ad.eval(f).unwrap(), 1.2, epsilon = 1e-10);
    }
}
