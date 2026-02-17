//! Tests for the #[autodiff] proc-macro.
//!
//! With the DiffNum-based macro, functions become generic:
//!   `fn f<T: DiffNum>(x: T) -> T`
//!
//! For AD graph construction, call inside `with_context`:
//!   `let result = with_context(&mut ad, || f(x_var));`

use crate::ops::with_context;
use crate::{AutoDiff, DiffNum};

/// Simple quadratic function
#[crate::autodiff]
fn quadratic(x: Var) -> Var {
    x * x + 2.0 * x + 1.0
}

/// Two-variable function
#[crate::autodiff]
fn two_vars(x: Var, y: Var) -> Var {
    x * y + x + y
}

/// Function with transcendental operations
#[crate::autodiff]
fn transcendental(x: Var) -> Var {
    sin(x) + cos(x)
}

/// Function with unary negation
#[crate::autodiff]
fn with_negation(x: Var) -> Var {
    -x * x
}

/// Rosenbrock function (classic optimization test)
#[crate::autodiff]
fn rosenbrock(x: Var, y: Var) -> Var {
    let a = 1.0;
    let b = 100.0;
    (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
}

#[test]
fn test_autodiff_quadratic() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // f(x) = x^2 + 2x + 1 = (x+1)^2
    // f(2) = 9
    let f = with_context(&mut ad, || quadratic(x));
    assert!((ad.eval(f).unwrap() - 9.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_quadratic_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // f(x) = x^2 + 2x + 1
    // f'(x) = 2x + 2
    // f'(2) = 6
    let f = with_context(&mut ad, || quadratic(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_quadratic_direct_f64() {
    // New: same function works with f64 directly (no AD context needed)
    assert_eq!(quadratic(2.0_f64), 9.0);
    assert_eq!(quadratic(0.0_f64), 1.0);
    assert_eq!(quadratic(-1.0_f64), 0.0);
}

#[test]
fn test_autodiff_quadratic_direct_f32() {
    // New: same function works with f32 directly
    assert_eq!(quadratic(2.0_f32), 9.0_f32);
    assert_eq!(quadratic(0.0_f32), 1.0_f32);
}

#[test]
fn test_autodiff_two_vars() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let y = ad.var(3.0).unwrap();

    // f(x, y) = xy + x + y
    // f(2, 3) = 6 + 2 + 3 = 11
    let f = with_context(&mut ad, || two_vars(x, y));
    assert!((ad.eval(f).unwrap() - 11.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_transcendental() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // f(x) = sin(x) + cos(x)
    // f(0) = 0 + 1 = 1
    let f = with_context(&mut ad, || transcendental(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_transcendental_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // f(x) = sin(x) + cos(x)
    // f'(x) = cos(x) - sin(x)
    // f'(0) = 1 - 0 = 1
    let f = with_context(&mut ad, || transcendental(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_negation() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();

    // f(x) = -x^2
    // f(3) = -9
    let f = with_context(&mut ad, || with_negation(x));
    assert!((ad.eval(f).unwrap() - (-9.0)).abs() < 1e-10);
}

#[test]
fn test_autodiff_rosenbrock_minimum() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();
    let y = ad.var(1.0).unwrap();

    // Rosenbrock at (1, 1) should be 0 (global minimum)
    let f = with_context(&mut ad, || rosenbrock(x, y));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_rosenbrock_gradient() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();
    let y = ad.var(1.0).unwrap();

    // At the minimum, gradient should be zero
    let f = with_context(&mut ad, || rosenbrock(x, y));
    let grad = ad.gradient(f).unwrap();

    assert!((grad[0]).abs() < 1e-10);
    assert!((grad[1]).abs() < 1e-10);
}

#[test]
fn test_autodiff_rosenbrock_direct_f64() {
    // Direct evaluation at the minimum
    assert_eq!(rosenbrock(1.0_f64, 1.0_f64), 0.0);
    // Direct evaluation at (0, 0)
    assert_eq!(rosenbrock(0.0_f64, 0.0_f64), 1.0);
}

// ============================================================================
// Division tests
// ============================================================================

/// Function with division
#[crate::autodiff]
fn with_division(x: Var, y: Var) -> Var {
    x / y
}

/// Reciprocal function
#[crate::autodiff]
fn reciprocal(x: Var) -> Var {
    1.0 / x
}

#[test]
fn test_autodiff_division() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(6.0).unwrap();
    let y = ad.var(2.0).unwrap();

    let f = with_context(&mut ad, || with_division(x, y));
    assert!((ad.eval(f).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_reciprocal() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(4.0).unwrap();

    // f(x) = 1/x, f(4) = 0.25
    let f = with_context(&mut ad, || reciprocal(x));
    assert!((ad.eval(f).unwrap() - 0.25).abs() < 1e-10);
}

#[test]
fn test_autodiff_reciprocal_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // f(x) = 1/x, f'(x) = -1/x^2, f'(2) = -0.25
    let f = with_context(&mut ad, || reciprocal(x));
    assert!((ad.derivative(f, x, 1).unwrap() - (-0.25)).abs() < 1e-10);
}

// ============================================================================
// Exponential and logarithm tests
// ============================================================================

/// Exponential function
#[crate::autodiff]
fn exponential(x: Var) -> Var {
    exp(x)
}

/// Natural logarithm
#[crate::autodiff]
fn logarithm(x: Var) -> Var {
    ln(x)
}

/// Exp-log composition (should give x back)
#[crate::autodiff]
fn exp_ln_identity(x: Var) -> Var {
    exp(ln(x))
}

#[test]
fn test_autodiff_exp() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // exp(0) = 1
    let f = with_context(&mut ad, || exponential(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_exp_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();

    // d/dx exp(x) = exp(x), so exp'(1) = e
    let f = with_context(&mut ad, || exponential(x));
    assert!((ad.derivative(f, x, 1).unwrap() - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_autodiff_ln() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(std::f64::consts::E).unwrap();

    // ln(e) = 1
    let f = with_context(&mut ad, || logarithm(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_ln_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // d/dx ln(x) = 1/x, so ln'(2) = 0.5
    let f = with_context(&mut ad, || logarithm(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 0.5).abs() < 1e-10);
}

#[test]
fn test_autodiff_exp_ln_identity() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(5.0).unwrap();

    // exp(ln(x)) = x
    let f = with_context(&mut ad, || exp_ln_identity(x));
    assert!((ad.eval(f).unwrap() - 5.0).abs() < 1e-10);
}

// ============================================================================
// Square root tests
// ============================================================================

/// Square root function
#[crate::autodiff]
fn square_root(x: Var) -> Var {
    sqrt(x)
}

#[test]
fn test_autodiff_sqrt() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(9.0).unwrap();

    let f = with_context(&mut ad, || square_root(x));
    assert!((ad.eval(f).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_sqrt_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(4.0).unwrap();

    // d/dx sqrt(x) = 1/(2*sqrt(x)), so sqrt'(4) = 1/4 = 0.25
    let f = with_context(&mut ad, || square_root(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 0.25).abs() < 1e-10);
}

// ============================================================================
// Hyperbolic function tests
// ============================================================================

/// Hyperbolic sine
#[crate::autodiff]
fn hyperbolic_sin(x: Var) -> Var {
    sinh(x)
}

/// Hyperbolic cosine
#[crate::autodiff]
fn hyperbolic_cos(x: Var) -> Var {
    cosh(x)
}

/// Hyperbolic identity: cosh^2 - sinh^2 = 1
#[crate::autodiff]
fn hyperbolic_identity(x: Var) -> Var {
    cosh(x) * cosh(x) - sinh(x) * sinh(x)
}

#[test]
fn test_autodiff_sinh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // sinh(0) = 0
    let f = with_context(&mut ad, || hyperbolic_sin(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_cosh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // cosh(0) = 1
    let f = with_context(&mut ad, || hyperbolic_cos(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_hyperbolic_identity() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.5).unwrap();

    // cosh^2(x) - sinh^2(x) = 1 for all x
    let f = with_context(&mut ad, || hyperbolic_identity(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_sinh_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx sinh(x) = cosh(x), so sinh'(0) = cosh(0) = 1
    let f = with_context(&mut ad, || hyperbolic_sin(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Power function tests
// ============================================================================

/// Power function
#[crate::autodiff]
fn power_func(x: Var, y: Var) -> Var {
    pow(x, y)
}

/// Square using pow
#[crate::autodiff]
fn square_via_pow(x: Var) -> Var {
    pow(x, 2.0)
}

#[test]
fn test_autodiff_pow() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let y = ad.var(3.0).unwrap();

    // 2^3 = 8
    let f = with_context(&mut ad, || power_func(x, y));
    assert!((ad.eval(f).unwrap() - 8.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_pow_square() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(5.0).unwrap();

    // 5^2 = 25
    let f = with_context(&mut ad, || square_via_pow(x));
    assert!((ad.eval(f).unwrap() - 25.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_pow_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();

    // d/dx x^2 = 2x, so (x^2)'(3) = 6
    let f = with_context(&mut ad, || square_via_pow(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 6.0).abs() < 1e-10);
}

// ============================================================================
// Nested function composition tests
// ============================================================================

/// Nested: sin(cos(x))
#[crate::autodiff]
fn sin_of_cos(x: Var) -> Var {
    sin(cos(x))
}

/// Nested: exp(sin(x))
#[crate::autodiff]
fn exp_of_sin(x: Var) -> Var {
    exp(sin(x))
}

/// Triple nested: sqrt(exp(cos(x)))
#[crate::autodiff]
fn triple_nested(x: Var) -> Var {
    sqrt(exp(cos(x)))
}

#[test]
fn test_autodiff_sin_of_cos() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // sin(cos(0)) = sin(1)
    let f = with_context(&mut ad, || sin_of_cos(x));
    assert!((ad.eval(f).unwrap() - 1.0_f64.sin()).abs() < 1e-10);
}

#[test]
fn test_autodiff_sin_of_cos_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx sin(cos(x)) = cos(cos(x)) * (-sin(x))
    // At x=0: cos(cos(0)) * (-sin(0)) = cos(1) * 0 = 0
    let f = with_context(&mut ad, || sin_of_cos(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_exp_of_sin() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // exp(sin(0)) = exp(0) = 1
    let f = with_context(&mut ad, || exp_of_sin(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_exp_of_sin_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx exp(sin(x)) = exp(sin(x)) * cos(x)
    // At x=0: exp(0) * cos(0) = 1 * 1 = 1
    let f = with_context(&mut ad, || exp_of_sin(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_triple_nested() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // sqrt(exp(cos(0))) = sqrt(exp(1)) = sqrt(e)
    let f = with_context(&mut ad, || triple_nested(x));
    let expected = std::f64::consts::E.sqrt();
    assert!((ad.eval(f).unwrap() - expected).abs() < 1e-10);
}

// ============================================================================
// Higher-order derivative tests
// ============================================================================

/// Cubic function for higher derivatives
#[crate::autodiff]
fn cubic(x: Var) -> Var {
    x * x * x
}

/// Quartic function
#[crate::autodiff]
fn quartic(x: Var) -> Var {
    x * x * x * x
}

#[test]
fn test_autodiff_cubic_derivatives() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // f(x) = x^3
    // f(2) = 8
    // f'(x) = 3x^2, f'(2) = 12
    // f''(x) = 6x, f''(2) = 12
    // f'''(x) = 6, f'''(2) = 6
    // f''''(x) = 0
    let f = with_context(&mut ad, || cubic(x));

    assert!((ad.eval(f).unwrap() - 8.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 1).unwrap() - 12.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 2).unwrap() - 12.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 3).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_quartic_derivatives() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();

    // f(x) = x^4
    // f(1) = 1
    // f'(x) = 4x^3, f'(1) = 4
    // f''(x) = 12x^2, f''(1) = 12
    // f'''(x) = 24x, f'''(1) = 24
    // f''''(x) = 24, f''''(1) = 24
    let f = with_context(&mut ad, || quartic(x));

    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 1).unwrap() - 4.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 2).unwrap() - 12.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 3).unwrap() - 24.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 4).unwrap() - 24.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_exp_higher_derivatives() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // All derivatives of exp(x) at x=0 are 1
    let f = with_context(&mut ad, || exponential(x));

    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 2).unwrap() - 1.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 3).unwrap() - 1.0).abs() < 1e-10);
    assert!((ad.derivative(f, x, 4).unwrap() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Integer literal tests
// ============================================================================

/// Function using integer literals
#[crate::autodiff]
fn with_integers(x: Var) -> Var {
    3 * x * x + 2 * x + 1
}

#[test]
fn test_autodiff_integer_literals() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // 3*4 + 2*2 + 1 = 12 + 4 + 1 = 17
    let f = with_context(&mut ad, || with_integers(x));
    assert!((ad.eval(f).unwrap() - 17.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_integer_literals_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // f(x) = 3x^2 + 2x + 1
    // f'(x) = 6x + 2, f'(2) = 14
    let f = with_context(&mut ad, || with_integers(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 14.0).abs() < 1e-10);
}

// ============================================================================
// Edge case tests
// ============================================================================

/// Constant-only function
#[crate::autodiff]
fn constant_only(_x: Var) -> Var {
    42.0
}

/// Deep nesting with parentheses
#[crate::autodiff]
fn deep_nesting(x: Var, y: Var) -> Var {
    ((x + y) * (x - y)) / ((x * x) + (y * y))
}

#[test]
fn test_autodiff_constant_only() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(999.0).unwrap();

    // Should always return 42 regardless of x
    let f = with_context(&mut ad, || constant_only(x));
    assert!((ad.eval(f).unwrap() - 42.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_constant_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(5.0).unwrap();

    // Derivative of constant is 0
    let f = with_context(&mut ad, || constant_only(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_deep_nesting() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();
    let y = ad.var(4.0).unwrap();

    // ((3+4)*(3-4)) / ((9)+(16)) = (7*(-1)) / 25 = -7/25 = -0.28
    let f = with_context(&mut ad, || deep_nesting(x, y));
    assert!((ad.eval(f).unwrap() - (-0.28)).abs() < 1e-10);
}

// ============================================================================
// Tangent function tests
// ============================================================================

/// Tangent function
#[crate::autodiff]
fn tangent(x: Var) -> Var {
    tan(x)
}

/// Hyperbolic tangent
#[crate::autodiff]
fn hyperbolic_tan(x: Var) -> Var {
    tanh(x)
}

/// tan(x) = sin(x)/cos(x) identity test
#[crate::autodiff]
fn tan_identity(x: Var) -> Var {
    sin(x) / cos(x)
}

#[test]
fn test_autodiff_tan() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // tan(0) = 0
    let f = with_context(&mut ad, || tangent(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_tan_at_pi_4() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(std::f64::consts::FRAC_PI_4).unwrap();

    // tan(π/4) = 1
    let f = with_context(&mut ad, || tangent(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_tan_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx tan(x) = sec²(x) = 1/cos²(x)
    // At x=0: 1/cos²(0) = 1/1 = 1
    let f = with_context(&mut ad, || tangent(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_tan_identity() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.5).unwrap();

    // tan(x) should equal sin(x)/cos(x)
    let f1 = with_context(&mut ad, || tangent(x));
    let f2 = with_context(&mut ad, || tan_identity(x));
    assert!((ad.eval(f1).unwrap() - ad.eval(f2).unwrap()).abs() < 1e-10);
}

#[test]
fn test_autodiff_tanh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // tanh(0) = 0
    let f = with_context(&mut ad, || hyperbolic_tan(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_tanh_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx tanh(x) = sech²(x) = 1/cosh²(x)
    // At x=0: 1/cosh²(0) = 1/1 = 1
    let f = with_context(&mut ad, || hyperbolic_tan(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Inverse trigonometric function tests
// ============================================================================

/// Arc sine
#[crate::autodiff]
fn arc_sin(x: Var) -> Var {
    asin(x)
}

/// Arc cosine
#[crate::autodiff]
fn arc_cos(x: Var) -> Var {
    acos(x)
}

/// Arc tangent
#[crate::autodiff]
fn arc_tan(x: Var) -> Var {
    atan(x)
}

#[test]
fn test_autodiff_asin() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // asin(0) = 0
    let f = with_context(&mut ad, || arc_sin(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_asin_at_half() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.5).unwrap();

    // asin(0.5) = π/6
    let f = with_context(&mut ad, || arc_sin(x));
    assert!((ad.eval(f).unwrap() - std::f64::consts::FRAC_PI_6).abs() < 1e-10);
}

#[test]
fn test_autodiff_asin_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx asin(x) = 1/sqrt(1-x²)
    // At x=0: 1/sqrt(1) = 1
    let f = with_context(&mut ad, || arc_sin(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_acos() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // acos(0) = π/2
    let f = with_context(&mut ad, || arc_cos(x));
    assert!((ad.eval(f).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn test_autodiff_acos_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx acos(x) = -1/sqrt(1-x²)
    // At x=0: -1/sqrt(1) = -1
    let f = with_context(&mut ad, || arc_cos(x));
    assert!((ad.derivative(f, x, 1).unwrap() - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_autodiff_asin_acos_sum() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.5).unwrap();

    // asin(x) + acos(x) = π/2 for all x in [-1, 1]
    let f1 = with_context(&mut ad, || arc_sin(x));
    let f2 = with_context(&mut ad, || arc_cos(x));
    assert!(
        (ad.eval(f1).unwrap() + ad.eval(f2).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10
    );
}

#[test]
fn test_autodiff_atan() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // atan(0) = 0
    let f = with_context(&mut ad, || arc_tan(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_atan_at_one() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();

    // atan(1) = π/4
    let f = with_context(&mut ad, || arc_tan(x));
    assert!((ad.eval(f).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn test_autodiff_atan_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx atan(x) = 1/(1+x²)
    // At x=0: 1/(1+0) = 1
    let f = with_context(&mut ad, || arc_tan(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_atan_derivative_at_one() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();

    // d/dx atan(x) = 1/(1+x²)
    // At x=1: 1/(1+1) = 0.5
    let f = with_context(&mut ad, || arc_tan(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 0.5).abs() < 1e-10);
}

// ============================================================================
// Inverse hyperbolic function tests
// ============================================================================

/// Arc hyperbolic sine
#[crate::autodiff]
fn arc_sinh(x: Var) -> Var {
    asinh(x)
}

/// Arc hyperbolic cosine
#[crate::autodiff]
fn arc_cosh(x: Var) -> Var {
    acosh(x)
}

/// Arc hyperbolic tangent
#[crate::autodiff]
fn arc_tanh(x: Var) -> Var {
    atanh(x)
}

#[test]
fn test_autodiff_asinh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // asinh(0) = 0
    let f = with_context(&mut ad, || arc_sinh(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_asinh_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx asinh(x) = 1/sqrt(x²+1)
    // At x=0: 1/sqrt(1) = 1
    let f = with_context(&mut ad, || arc_sinh(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_acosh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();

    // acosh(1) = 0
    let f = with_context(&mut ad, || arc_cosh(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_acosh_at_two() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // acosh(2) = ln(2 + sqrt(3))
    let f = with_context(&mut ad, || arc_cosh(x));
    assert!((ad.eval(f).unwrap() - 2.0_f64.acosh()).abs() < 1e-10);
}

#[test]
fn test_autodiff_acosh_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // d/dx acosh(x) = 1/sqrt(x²-1)
    // At x=2: 1/sqrt(4-1) = 1/sqrt(3)
    let f = with_context(&mut ad, || arc_cosh(x));
    assert!((ad.derivative(f, x, 1).unwrap() - (1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
}

#[test]
fn test_autodiff_atanh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // atanh(0) = 0
    let f = with_context(&mut ad, || arc_tanh(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_atanh_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // d/dx atanh(x) = 1/(1-x²)
    // At x=0: 1/(1-0) = 1
    let f = with_context(&mut ad, || arc_tanh(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_atanh_at_half() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.5).unwrap();

    // atanh(0.5) = 0.5 * ln(3) ≈ 0.5493
    let f = with_context(&mut ad, || arc_tanh(x));
    assert!((ad.eval(f).unwrap() - 0.5_f64.atanh()).abs() < 1e-10);
}

// ============================================================================
// Composition tests with new functions
// ============================================================================

/// Composition: sin(atan(x))
#[crate::autodiff]
fn sin_of_atan(x: Var) -> Var {
    sin(atan(x))
}

/// Composition: exp(tanh(x))
#[crate::autodiff]
fn exp_of_tanh(x: Var) -> Var {
    exp(tanh(x))
}

#[test]
fn test_autodiff_sin_of_atan() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();

    // sin(atan(1)) = sin(π/4) = √2/2
    let f = with_context(&mut ad, || sin_of_atan(x));
    assert!((ad.eval(f).unwrap() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
}

#[test]
fn test_autodiff_exp_of_tanh() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // exp(tanh(0)) = exp(0) = 1
    let f = with_context(&mut ad, || exp_of_tanh(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Phase 3: square, powi, powf, and method syntax tests
// ============================================================================

/// square() function call
#[crate::autodiff]
fn square_func(x: Var) -> Var {
    square(x)
}

/// powi() function call
#[crate::autodiff]
fn powi_func(x: Var) -> Var {
    powi(x, 3)
}

/// powf() function call — second arg is raw f64
#[crate::autodiff]
fn powf_func(x: Var) -> Var {
    powf(x, 0.5)
}

/// Method syntax: x.sin()
#[crate::autodiff]
fn method_sin(x: Var) -> Var {
    x.sin()
}

/// Method syntax chain: x.sin().cos()
#[crate::autodiff]
fn method_chain(x: Var) -> Var {
    x.sin().cos()
}

/// Method syntax: x.powi(3)
#[crate::autodiff]
fn method_powi(x: Var) -> Var {
    x.powi(3)
}

#[test]
fn test_autodiff_square() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(5.0).unwrap();

    let f = with_context(&mut ad, || square_func(x));
    assert!((ad.eval(f).unwrap() - 25.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_powi() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // x^3 = 8
    let f = with_context(&mut ad, || powi_func(x));
    assert!((ad.eval(f).unwrap() - 8.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_powi_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();

    // d/dx x^3 = 3x^2 = 27
    let f = with_context(&mut ad, || powi_func(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 27.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_powf() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(4.0).unwrap();

    // 4^0.5 = 2
    let f = with_context(&mut ad, || powf_func(x));
    assert!((ad.eval(f).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_method_syntax_sin() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // x.sin() at x=0 = 0
    let f = with_context(&mut ad, || method_sin(x));
    assert!((ad.eval(f).unwrap() - 0.0).abs() < 1e-10);

    // d/dx sin(x) at x=0 = cos(0) = 1
    assert!((ad.derivative(f, x, 1).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_method_syntax_chain() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();

    // cos(sin(0)) = cos(0) = 1
    let f = with_context(&mut ad, || method_chain(x));
    assert!((ad.eval(f).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_method_powi() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // x.powi(3) = 8
    let f = with_context(&mut ad, || method_powi(x));
    assert!((ad.eval(f).unwrap() - 8.0).abs() < 1e-10);

    // d/dx x^3 = 3x^2 = 12
    assert!((ad.derivative(f, x, 1).unwrap() - 12.0).abs() < 1e-10);
}

// ============================================================================
// Phase 4: Logarithmic derivative operations + stable_derivatives attribute
// ============================================================================

/// pow_log() function call
#[crate::autodiff]
fn pow_log_func(x: Var, y: Var) -> Var {
    pow_log(x, y)
}

/// div_log() function call
#[crate::autodiff]
fn div_log_func(x: Var, y: Var) -> Var {
    div_log(x, y)
}

/// powi_log via function call
#[crate::autodiff]
fn powi_log_func(x: Var) -> Var {
    powi_log(x, 3)
}

/// stable_derivatives attribute: pow → pow_log, / → div_log
#[crate::autodiff(stable_derivatives)]
fn stable_power_div(x: Var, y: Var) -> Var {
    pow(x, y) + x / y
}

/// stable_derivatives attribute: powi/powf method syntax routes to _log variants
#[crate::autodiff(stable_derivatives)]
fn stable_method_powi_powf(x: Var) -> Var {
    x.powi(3) + x.powf(0.5)
}

#[test]
fn test_autodiff_pow_log() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let y = ad.var(3.0).unwrap();

    // 2^3 = 8 (Tier 2: closed-form f64, few ULPs expected)
    let f = with_context(&mut ad, || pow_log_func(x, y));
    assert!((ad.eval(f).unwrap() - 8.0).abs() < 1e-13);
}

#[test]
fn test_autodiff_div_log() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(6.0).unwrap();
    let y = ad.var(2.0).unwrap();

    let f = with_context(&mut ad, || div_log_func(x, y));
    assert!((ad.eval(f).unwrap() - 3.0).abs() < 1e-13);
}

#[test]
fn test_autodiff_powi_log_derivative() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // d/dx x^3 = 3x^2 = 12 (Tier 2: closed-form f64)
    let f = with_context(&mut ad, || powi_log_func(x));
    assert!((ad.derivative(f, x, 1).unwrap() - 12.0).abs() < 1e-13);
}

#[test]
fn test_autodiff_stable_derivatives() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let y = ad.var(3.0).unwrap();

    // pow(2,3) + 2/3 = 8 + 0.6667 = 8.6667 (Tier 2: closed-form f64)
    let f = with_context(&mut ad, || stable_power_div(x, y));
    let expected = 8.0 + 2.0 / 3.0;
    assert!((ad.eval(f).unwrap() - expected).abs() < 1e-13);
}

#[test]
fn test_autodiff_stable_derivatives_powi_powf() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    // x.powi(3) + x.powf(0.5) = 8 + sqrt(2) ≈ 9.4142 (Tier 2: closed-form f64)
    let f = with_context(&mut ad, || stable_method_powi_powf(x));
    let expected = 8.0 + 2.0_f64.sqrt();
    assert!((ad.eval(f).unwrap() - expected).abs() < 1e-13);

    // Derivative: 3x^2 + 0.5*x^(-0.5) = 12 + 0.5/sqrt(2) ≈ 12.3536
    let deriv = ad.derivative(f, x, 1).unwrap();
    let expected_deriv = 12.0 + 0.5 * 2.0_f64.powf(-0.5);
    assert!((deriv - expected_deriv).abs() < 1e-13);
}

// ============================================================================
// Part B: Non-float parameter tests (i32 params should NOT be transformed)
// ============================================================================

/// Function with i32 parameter — only float params become T, i32 stays i32
#[crate::autodiff]
fn power_with_int(x: f64, n: i32) -> f64 {
    x.powi(n)
}

/// Function with i32 param using free-function syntax
#[crate::autodiff]
fn power_with_int_free(x: f64, n: i32) -> f64 {
    powi(x, n)
}

#[test]
fn test_autodiff_i32_param_direct_f64() {
    // i32 param works with direct f64 evaluation
    assert!((power_with_int(2.0_f64, 3) - 8.0).abs() < 1e-10);
    assert!((power_with_int(3.0_f64, 2) - 9.0).abs() < 1e-10);
    assert!((power_with_int(5.0_f64, 0) - 1.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_i32_param_direct_f32() {
    // i32 param works with direct f32 evaluation
    assert!((power_with_int(2.0_f32, 3) - 8.0_f32).abs() < 1e-5);
    assert!((power_with_int(3.0_f32, 2) - 9.0_f32).abs() < 1e-5);
}

#[test]
fn test_autodiff_i32_param_var() {
    // i32 param works with Var graph construction
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();

    let f = with_context(&mut ad, || power_with_int(x, 3));
    assert!((ad.eval(f).unwrap() - 8.0).abs() < 1e-10);

    // d/dx x^3 = 3x^2 = 12
    assert!((ad.derivative(f, x, 1).unwrap() - 12.0).abs() < 1e-10);
}

#[test]
fn test_autodiff_i32_param_free_function() {
    // Free-function powi(x, n) with i32 param
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();

    let f = with_context(&mut ad, || power_with_int_free(x, 4));
    assert!((ad.eval(f).unwrap() - 81.0).abs() < 1e-10);

    // d/dx x^4 = 4x^3 = 108
    assert!((ad.derivative(f, x, 1).unwrap() - 108.0).abs() < 1e-10);
}

// ============================================================================
// f64/f32 consistency tests
// ============================================================================

#[test]
fn test_autodiff_f64_f32_consistency_transcendental() {
    // Same function should give consistent results in f64 and f32
    let f64_val = transcendental(0.5_f64);
    let f32_val = transcendental(0.5_f32);
    assert!((f64_val - f32_val as f64).abs() < 1e-6);
}

#[test]
fn test_autodiff_f64_f32_consistency_rosenbrock() {
    let f64_val = rosenbrock(2.0_f64, 3.0_f64);
    let f32_val = rosenbrock(2.0_f32, 3.0_f32);
    assert!((f64_val - f32_val as f64).abs() < 1e-3);
}

// ============================================================================
// stable_derivatives div_log via division operator
// ============================================================================

/// stable_derivatives routes `/` to div_log
#[crate::autodiff(stable_derivatives)]
fn stable_reciprocal(x: Var) -> Var {
    1.0 / x
}

#[test]
fn test_autodiff_stable_div_log_via_operator() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(4.0).unwrap();

    // 1/4 = 0.25
    let f = with_context(&mut ad, || stable_reciprocal(x));
    assert!((ad.eval(f).unwrap() - 0.25).abs() < 1e-13);

    // d/dx (1/x) = -1/x^2 = -1/16
    let deriv = ad.derivative(f, x, 1).unwrap();
    assert!((deriv - (-1.0 / 16.0)).abs() < 1e-13);

    // Second derivative: d²/dx² (1/x) = 2/x^3 = 2/64 = 0.03125
    let d2 = ad.derivative(f, x, 2).unwrap();
    assert!((d2 - (2.0 / 64.0)).abs() < 1e-10);
}
