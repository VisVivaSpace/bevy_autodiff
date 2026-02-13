//! Ergonomic macros for building computation graphs.
//!
//! The [`expr!`] macro transforms natural Rust expressions into
//! AutoDiff method calls, making graph construction more readable.
//!
//! # Example
//!
//! ```
//! use bevy_autodiff::{AutoDiff, expr};
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(2.0);
//! let y = ad.var(3.0);
//!
//! // Instead of: ad.add(ad.mul(x, x), ad.mul(x, y))
//! let f = expr!(ad, x * x + x * y);
//! assert_eq!(ad.eval(f), 10.0);  // 4 + 6 = 10
//! ```
//!
//! # Supported Operations
//!
//! - **Binary operators**: `+`, `-`, `*`, `/`
//! - **Unary negation**: `-x`
//! - **Transcendental functions**: `sin`, `cos`, `exp`, `ln`, `sqrt`, `sinh`, `cosh`
//! - **Power function**: `pow(base, exp)`
//! - **Float literals**: automatically wrapped as `ad.constant()`
//! - **Parentheses**: `(expr)` for grouping
//!
//! # Precedence
//!
//! Standard Rust precedence is followed:
//! - `*`, `/` bind tighter than `+`, `-`
//! - Unary `-` binds tighter than binary operators
//! - Function calls and parentheses bind tightest
//!
//! # Limitations
//!
//! Due to Rust macro limitations, some complex expressions may require
//! explicit parentheses for correct parsing.

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
/// let x = ad.var(2.0);
/// let y = ad.var(3.0);
///
/// let sum = expr!(ad, x + y);
/// assert_eq!(ad.eval(sum), 5.0);
///
/// let product = expr!(ad, x * y);
/// assert_eq!(ad.eval(product), 6.0);
/// ```
///
/// ## With Literals
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(2.0);
///
/// let f = expr!(ad, x * x + 3.0);
/// assert_eq!(ad.eval(f), 7.0);  // 4 + 3
/// ```
///
/// ## Transcendental Functions
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(0.0);
///
/// let f = expr!(ad, sin(x) + cos(x));
/// assert!((ad.eval(f) - 1.0).abs() < 1e-10);  // sin(0) + cos(0) = 0 + 1
/// ```
///
/// ## Complex Expressions
/// ```
/// use bevy_autodiff::{AutoDiff, expr};
///
/// let mut ad = AutoDiff::new();
/// let x = ad.var(1.0);
/// let y = ad.var(1.0);
///
/// // Rosenbrock function at (1,1)
/// let f = expr!(ad, (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x));
/// assert!((ad.eval(f) - 0.0).abs() < 1e-10);
/// ```
#[macro_export]
macro_rules! expr {
    // Entry point: start at additive level with empty accumulator
    // Format: @add_munch $ad, {$accumulated}, $rest...
    ($ad:ident, $($tokens:tt)+) => {
        $crate::expr!(@add_munch $ad, {}, $($tokens)+ @end)
    };

    // ============================================================
    // ADDITIVE LEVEL: Munch tokens looking for + or -
    // Format: @add_munch $ad, {$accumulated}, next rest...
    // ============================================================

    // End marker: no + or - found, pass to multiplicative
    (@add_munch $ad:ident, {$($acc:tt)*}, @end) => {
        $crate::expr!(@mul_munch $ad, {}, $($acc)* @end)
    };

    // Found + at additive level
    (@add_munch $ad:ident, {$($left:tt)*}, + $($rest:tt)+) => {{
        let lhs = $crate::expr!(@mul_munch $ad, {}, $($left)* @end);
        let rhs = $crate::expr!(@add_munch $ad, {}, $($rest)+);
        $ad.add(lhs, rhs)
    }};

    // Found - at additive level with NON-EMPTY left side (binary minus)
    (@add_munch $ad:ident, {$($left:tt)+}, - $($rest:tt)+) => {{
        let lhs = $crate::expr!(@mul_munch $ad, {}, $($left)+ @end);
        let rhs = $crate::expr!(@add_munch $ad, {}, $($rest)+);
        $ad.sub(lhs, rhs)
    }};

    // Found - at additive level with EMPTY left side (unary minus) - pass to mul level
    // Note: $rest already contains @end from the entry point
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
    // MULTIPLICATIVE LEVEL: Munch tokens looking for * or /
    // ============================================================

    // End marker: no * or / found, pass to unary (with at least one token)
    (@mul_munch $ad:ident, {$($acc:tt)+}, @end) => {
        $crate::expr!(@unary $ad, $($acc)+)
    };


    // Found * at multiplicative level
    (@mul_munch $ad:ident, {$($left:tt)*}, * $($rest:tt)+) => {{
        let lhs = $crate::expr!(@unary $ad, $($left)*);
        let rhs = $crate::expr!(@mul_munch $ad, {}, $($rest)+);
        $ad.mul(lhs, rhs)
    }};

    // Found / at multiplicative level
    (@mul_munch $ad:ident, {$($left:tt)*}, / $($rest:tt)+) => {{
        let lhs = $crate::expr!(@unary $ad, $($left)*);
        let rhs = $crate::expr!(@mul_munch $ad, {}, $($rest)+);
        $ad.div(lhs, rhs)
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

    // Function calls
    (@atom $ad:ident, sin ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.sin(arg)
    }};

    (@atom $ad:ident, cos ($($arg:tt)+)) => {{
        let arg = $crate::expr!(@add_munch $ad, {}, $($arg)+ @end);
        $ad.cos(arg)
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

    (@atom $ad:ident, pow ($base:tt, $exp:tt)) => {{
        let base = $crate::expr!(@atom $ad, $base);
        let exp_val = $crate::expr!(@atom $ad, $exp);
        $ad.pow(base, exp_val)
    }};

    // Literals
    (@atom $ad:ident, $lit:literal) => {
        $ad.constant($lit as f64)
    };

    // Variables
    (@atom $ad:ident, $var:ident) => {
        $var
    };
}

#[cfg(test)]
mod tests {
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_expr_simple_add() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        let f = expr!(ad, x + y);
        assert_eq!(ad.eval(f), 5.0);
    }

    #[test]
    fn test_expr_simple_sub() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let y = ad.var(3.0);

        let f = expr!(ad, x - y);
        assert_eq!(ad.eval(f), 2.0);
    }

    #[test]
    fn test_expr_simple_mul() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);
        let y = ad.var(5.0);

        let f = expr!(ad, x * y);
        assert_eq!(ad.eval(f), 20.0);
    }

    #[test]
    fn test_expr_simple_div() {
        let mut ad = AutoDiff::new();
        let x = ad.var(10.0);
        let y = ad.var(2.0);

        let f = expr!(ad, x / y);
        assert_eq!(ad.eval(f), 5.0);
    }

    #[test]
    fn test_expr_with_literal() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        let f = expr!(ad, x * x + 3.0);
        assert_eq!(ad.eval(f), 7.0);
    }

    #[test]
    fn test_expr_literal_first() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);

        let f = expr!(ad, 2.0 * x);
        assert_eq!(ad.eval(f), 6.0);
    }

    #[test]
    fn test_expr_sin() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);

        let f = expr!(ad, sin(x));
        assert_relative_eq!(ad.eval(f), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);

        let f = expr!(ad, cos(x));
        assert_relative_eq!(ad.eval(f), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sin_plus_cos() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);

        let f = expr!(ad, sin(x) + cos(x));
        assert_relative_eq!(ad.eval(f), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_exp() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);

        let f = expr!(ad, exp(x));
        assert_relative_eq!(ad.eval(f), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_ln() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);

        let f = expr!(ad, ln(x));
        assert_relative_eq!(ad.eval(f), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sqrt() {
        let mut ad = AutoDiff::new();
        let x = ad.var(4.0);

        let f = expr!(ad, sqrt(x));
        assert_relative_eq!(ad.eval(f), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_sinh_cosh() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);

        let sh = expr!(ad, sinh(x));
        let ch = expr!(ad, cosh(x));
        assert_relative_eq!(ad.eval(sh), 0.0, epsilon = 1e-10);
        assert_relative_eq!(ad.eval(ch), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_pow() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        let f = expr!(ad, pow(x, y));
        assert_relative_eq!(ad.eval(f), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_nested_parens() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // (x + y) * (x - y) = x² - y² = 4 - 9 = -5
        let f = expr!(ad, (x + y) * (x - y));
        assert_eq!(ad.eval(f), -5.0);
    }

    #[test]
    fn test_expr_unary_neg() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);

        let f = expr!(ad, -x);
        assert_eq!(ad.eval(f), -5.0);
    }

    #[test]
    fn test_expr_rosenbrock() {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let y = ad.var(1.0);

        // Rosenbrock at (1,1) should be 0
        // f = (1-x)² + 100(y-x²)²
        let f = expr!(
            ad,
            (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
        );
        assert_relative_eq!(ad.eval(f), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expr_integer_literal() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        // Integer literals should work too (cast to f64)
        let f = expr!(ad, x + 3);
        assert_eq!(ad.eval(f), 5.0);
    }

    #[test]
    fn test_expr_multiple_same_var() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);

        // f = x * x * x = x³ = 27
        let f = expr!(ad, x * x * x);
        assert_eq!(ad.eval(f), 27.0);
    }

    #[test]
    fn test_expr_function_composition() {
        let mut ad = AutoDiff::new();
        let x = ad.var(0.0);

        // f = sin(cos(x)) = sin(1) at x=0
        let f = expr!(ad, sin(cos(x)));
        assert_relative_eq!(ad.eval(f), 1.0_f64.sin(), epsilon = 1e-10);
    }
}
