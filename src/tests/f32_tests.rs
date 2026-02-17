//! Tests verifying `AutoDiff<f32>` works correctly.
//!
//! These mirror a subset of the f64 tests to confirm that the generic
//! float machinery produces correct results at single precision.

use crate::AutoDiff;

// ============================================================================
// Basic evaluation
// ============================================================================

#[test]
fn f32_basic_arithmetic() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(2.0_f32).unwrap();
    let y = ad.var(3.0_f32).unwrap();

    let sum = ad.add(x, y);
    assert_eq!(ad.eval(sum).unwrap(), 5.0_f32);

    let prod = ad.mul(x, y);
    assert_eq!(ad.eval(prod).unwrap(), 6.0_f32);

    let diff = ad.sub(x, y);
    assert_eq!(ad.eval(diff).unwrap(), -1.0_f32);

    let quot = ad.div(x, y);
    assert!((ad.eval(quot).unwrap() - 2.0_f32 / 3.0_f32).abs() < 1e-6);
}

#[test]
fn f32_unary_ops() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(0.0_f32).unwrap();

    let sin_x = ad.sin(x);
    assert_eq!(ad.eval(sin_x).unwrap(), 0.0_f32);
    let cos_x = ad.cos(x);
    assert_eq!(ad.eval(cos_x).unwrap(), 1.0_f32);
    let exp_x = ad.exp(x);
    assert_eq!(ad.eval(exp_x).unwrap(), 1.0_f32);
    let tan_x = ad.tan(x);
    assert_eq!(ad.eval(tan_x).unwrap(), 0.0_f32);
    let sinh_x = ad.sinh(x);
    assert_eq!(ad.eval(sinh_x).unwrap(), 0.0_f32);
    let cosh_x = ad.cosh(x);
    assert_eq!(ad.eval(cosh_x).unwrap(), 1.0_f32);
    let tanh_x = ad.tanh(x);
    assert_eq!(ad.eval(tanh_x).unwrap(), 0.0_f32);
}

#[test]
fn f32_ln_sqrt() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(4.0_f32).unwrap();

    let sqrt_x = ad.sqrt(x);
    assert_eq!(ad.eval(sqrt_x).unwrap(), 2.0_f32);
    let ln_x = ad.ln(x);
    assert!((ad.eval(ln_x).unwrap() - 4.0_f32.ln()).abs() < 1e-6);
}

#[test]
fn f32_square_and_neg() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(3.0_f32).unwrap();

    let sq = ad.square(x);
    assert_eq!(ad.eval(sq).unwrap(), 9.0_f32);
    let neg = ad.neg(x);
    assert_eq!(ad.eval(neg).unwrap(), -3.0_f32);
}

#[test]
fn f32_constants() {
    let mut ad = AutoDiff::<f32>::new();
    let c = ad.constant(42.0_f32);
    assert_eq!(ad.eval(c).unwrap(), 42.0_f32);
}

// ============================================================================
// Differentiation
// ============================================================================

#[test]
fn f32_first_derivative_polynomial() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(2.0_f32).unwrap();

    // f(x) = x^2, f'(x) = 2x, f'(2) = 4
    let f = ad.square(x);
    let df = ad.differentiate(f, x).unwrap();
    assert!((ad.eval(df).unwrap() - 4.0_f32).abs() < 1e-5);
}

#[test]
fn f32_first_derivative_sin() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(0.0_f32).unwrap();

    // d/dx sin(x) = cos(x), cos(0) = 1
    let f = ad.sin(x);
    let df = ad.differentiate(f, x).unwrap();
    assert!((ad.eval(df).unwrap() - 1.0_f32).abs() < 1e-5);
}

#[test]
fn f32_second_derivative() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(1.0_f32).unwrap();

    // f(x) = x^3 (via mul chain), f''(x) = 6x, f''(1) = 6
    let x2 = ad.mul(x, x);
    let x3 = ad.mul(x2, x);
    let d1 = ad.differentiate(x3, x).unwrap();
    let d2 = ad.differentiate(d1, x).unwrap();
    assert!((ad.eval(d2).unwrap() - 6.0_f32).abs() < 1e-4);
}

#[test]
fn f32_derivative_of_constant_is_zero() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(5.0_f32).unwrap();
    let c = ad.constant(7.0_f32);
    let dc = ad.differentiate(c, x).unwrap();
    assert_eq!(ad.eval(dc).unwrap(), 0.0_f32);
}

#[test]
fn f32_mixed_partial_symmetry() {
    // d²f/dxdy = d²f/dydx for f = x*y
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(2.0_f32).unwrap();
    let y = ad.var(3.0_f32).unwrap();

    let f = ad.mul(x, y);
    let dfdx = ad.differentiate(f, x).unwrap();
    let d2fdxdy = ad.differentiate(dfdx, y).unwrap();

    let dfdy = ad.differentiate(f, y).unwrap();
    let d2fdydx = ad.differentiate(dfdy, x).unwrap();

    assert!((ad.eval(d2fdxdy).unwrap() - ad.eval(d2fdydx).unwrap()).abs() < 1e-5);
    // Both should be 1.0
    assert!((ad.eval(d2fdxdy).unwrap() - 1.0_f32).abs() < 1e-5);
}

// ============================================================================
// CompiledGraph
// ============================================================================

#[test]
fn f32_compiled_graph_eval() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(0.0_f32).unwrap();
    let y = ad.var(0.0_f32).unwrap();

    // f = x^2 + y^2
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let f = ad.add(x2, y2);

    let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
    cg.eval(&[3.0_f32, 4.0_f32]).unwrap();
    assert_eq!(cg.value(), 25.0_f32);
}

#[test]
fn f32_compiled_graph_gradient() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(0.0_f32).unwrap();
    let y = ad.var(0.0_f32).unwrap();

    // f = x^2 + y^2, grad = [2x, 2y]
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let f = ad.add(x2, y2);

    let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
    cg.eval(&[3.0_f32, 4.0_f32]).unwrap();
    let grad = cg.gradient();
    assert!((grad[0] - 6.0_f32).abs() < 1e-5); // 2*3
    assert!((grad[1] - 8.0_f32).abs() < 1e-5); // 2*4
}

#[test]
fn f32_compiled_graph_re_eval() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(0.0_f32).unwrap();

    let f = ad.square(x);
    let mut cg = ad.compile_primal(f, &[x]).unwrap();

    // First eval
    cg.eval(&[3.0_f32]).unwrap();
    assert_eq!(cg.value(), 9.0_f32);

    // Re-eval with different input
    cg.eval(&[5.0_f32]).unwrap();
    assert_eq!(cg.value(), 25.0_f32);
}

#[test]
fn f32_compiled_graph_with_derivatives() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(0.0_f32).unwrap();

    // f = x^2, compile with first-order derivatives
    let x2 = ad.square(x);
    let mut cg = ad.compile_order(x2, &[x], 1).unwrap();
    cg.eval(&[3.0_f32]).unwrap();

    assert_eq!(cg.value(), 9.0_f32);
    // df/dx = 2x = 6
    assert!((cg.partial(&[1]).unwrap() - 6.0_f32).abs() < 1e-5);
}

// ============================================================================
// Pythagorean identity at f32 precision
// ============================================================================

#[test]
fn f32_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1
    for x_val in [0.0_f32, 0.5, 1.0, 2.0, -1.0] {
        let mut ad = AutoDiff::<f32>::new();
        let x = ad.var(x_val).unwrap();

        let sin_x = ad.sin(x);
        let cos_x = ad.cos(x);
        let sin2 = ad.square(sin_x);
        let cos2 = ad.square(cos_x);
        let sum = ad.add(sin2, cos2);

        assert!((ad.eval(sum).unwrap() - 1.0_f32).abs() < 1e-5);
    }
}

// ============================================================================
// exp/ln inverse at f32 precision
// ============================================================================

#[test]
fn f32_exp_ln_inverse() {
    for x_val in [0.1_f32, 0.5, 1.0, 2.0, 10.0] {
        let mut ad = AutoDiff::<f32>::new();
        let x = ad.var(x_val).unwrap();
        let ln_x = ad.ln(x);
        let result = ad.exp(ln_x);
        assert!((ad.eval(result).unwrap() - x_val).abs() < 1e-5);
    }
}

// ============================================================================
// f32 gradient via context.gradient()
// ============================================================================

#[test]
fn f32_context_gradient() {
    let mut ad = AutoDiff::<f32>::new();
    let x = ad.var(1.0_f32).unwrap();
    let y = ad.var(2.0_f32).unwrap();

    // f = x*y + x, grad = [y+1, x] = [3, 1]
    let xy = ad.mul(x, y);
    let f = ad.add(xy, x);

    let grad = ad.gradient(f).unwrap();
    assert!((grad[0] - 3.0_f32).abs() < 1e-5);
    assert!((grad[1] - 1.0_f32).abs() < 1e-5);
}
