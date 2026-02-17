//! Tests verifying `DiffNum` trait integration with autodiff.
//!
//! These tests confirm that functions generic over `T: DiffNum` produce
//! correct results for both direct float evaluation and AD graph construction,
//! including differentiation through generic functions.

use crate::ops::with_context;
use crate::{AutoDiff, DiffNum};

// ============================================================================
// Generic functions used across tests
// ============================================================================

fn quadratic<T: DiffNum>(x: T) -> T {
    x * x + T::from_f64(2.0) * x + T::from_f64(1.0)
}

fn two_body<T: DiffNum>(x: T, y: T) -> T {
    x * y + x.sin() + y.exp()
}

fn rosenbrock<T: DiffNum>(x: T, y: T) -> T {
    let a = T::from_f64(1.0);
    let b = T::from_f64(100.0);
    (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
}

fn transcendental_chain<T: DiffNum>(x: T) -> T {
    x.sin().exp() + x.cos().ln()
}

fn power_combo<T: DiffNum>(x: T) -> T {
    x.square() + x.sqrt() + x.powi(3)
}

// ============================================================================
// Direct f64 evaluation
// ============================================================================

#[test]
fn diffnum_quadratic_f64() {
    assert_eq!(quadratic(3.0_f64), 16.0);
    assert_eq!(quadratic(0.0_f64), 1.0);
    assert_eq!(quadratic(-1.0_f64), 0.0);
}

#[test]
fn diffnum_rosenbrock_f64() {
    assert_eq!(rosenbrock(1.0_f64, 1.0), 0.0);
    assert_eq!(rosenbrock(0.0_f64, 0.0), 1.0);
}

#[test]
fn diffnum_two_body_f64() {
    let expected = 2.0 * 3.0 + 2.0_f64.sin() + 3.0_f64.exp();
    assert!((two_body(2.0_f64, 3.0) - expected).abs() < 1e-10);
}

#[test]
fn diffnum_transcendental_chain_f64() {
    let x = 0.5_f64;
    let expected = x.sin().exp() + x.cos().ln();
    assert!((transcendental_chain(x) - expected).abs() < 1e-10);
}

#[test]
fn diffnum_power_combo_f64() {
    let x = 4.0_f64;
    let expected = x * x + x.sqrt() + x.powi(3);
    assert!((power_combo(x) - expected).abs() < 1e-10);
}

// ============================================================================
// Direct f32 evaluation
// ============================================================================

#[test]
fn diffnum_quadratic_f32() {
    assert_eq!(quadratic(3.0_f32), 16.0_f32);
    assert_eq!(quadratic(0.0_f32), 1.0_f32);
}

#[test]
fn diffnum_rosenbrock_f32() {
    assert_eq!(rosenbrock(1.0_f32, 1.0_f32), 0.0_f32);
}

#[test]
fn diffnum_power_combo_f32() {
    let x = 4.0_f32;
    let expected = x * x + x.sqrt() + x.powi(3);
    assert!((power_combo(x) - expected).abs() < 1e-4);
}

// ============================================================================
// Var evaluation (graph construction)
// ============================================================================

#[test]
fn diffnum_quadratic_var() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();
    let f = with_context(&mut ad, || quadratic(x));
    assert_eq!(ad.eval(f).unwrap(), 16.0);
}

#[test]
fn diffnum_rosenbrock_var() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();
    let y = ad.var(1.0).unwrap();
    let f = with_context(&mut ad, || rosenbrock(x, y));
    assert_eq!(ad.eval(f).unwrap(), 0.0);
}

#[test]
fn diffnum_two_body_var() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let y = ad.var(3.0).unwrap();
    let f = with_context(&mut ad, || two_body(x, y));
    let expected = 2.0 * 3.0 + 2.0_f64.sin() + 3.0_f64.exp();
    assert!((ad.eval(f).unwrap() - expected).abs() < 1e-10);
}

#[test]
fn diffnum_transcendental_chain_var() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.5).unwrap();
    let f = with_context(&mut ad, || transcendental_chain(x));
    let expected = 0.5_f64.sin().exp() + 0.5_f64.cos().ln();
    assert!((ad.eval(f).unwrap() - expected).abs() < 1e-10);
}

// ============================================================================
// Differentiation through generic functions
// ============================================================================

#[test]
fn diffnum_quadratic_derivative() {
    // f(x) = x^2 + 2x + 1, f'(x) = 2x + 2
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();
    let f = with_context(&mut ad, || quadratic(x));
    let df = ad.differentiate(f, x).unwrap();
    // f'(3) = 8
    assert!((ad.eval(df).unwrap() - 8.0).abs() < 1e-10);
}

#[test]
fn diffnum_quadratic_second_derivative() {
    // f(x) = x^2 + 2x + 1, f''(x) = 2
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(3.0).unwrap();
    let f = with_context(&mut ad, || quadratic(x));
    let df = ad.differentiate(f, x).unwrap();
    let d2f = ad.differentiate(df, x).unwrap();
    assert!((ad.eval(d2f).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn diffnum_rosenbrock_gradient_at_minimum() {
    // At (1, 1), gradient should be [0, 0]
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();
    let y = ad.var(1.0).unwrap();
    let f = with_context(&mut ad, || rosenbrock(x, y));

    let dfdx = ad.differentiate(f, x).unwrap();
    let dfdy = ad.differentiate(f, y).unwrap();
    assert!((ad.eval(dfdx).unwrap()).abs() < 1e-10);
    assert!((ad.eval(dfdy).unwrap()).abs() < 1e-10);
}

#[test]
fn diffnum_two_body_partial_derivatives() {
    // f(x,y) = x*y + sin(x) + exp(y)
    // df/dx = y + cos(x)
    // df/dy = x + exp(y)
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(1.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let f = with_context(&mut ad, || two_body(x, y));

    let dfdx = ad.differentiate(f, x).unwrap();
    let dfdy = ad.differentiate(f, y).unwrap();

    // df/dx at (1, 0) = 0 + cos(1)
    assert!((ad.eval(dfdx).unwrap() - 1.0_f64.cos()).abs() < 1e-10);
    // df/dy at (1, 0) = 1 + exp(0) = 2
    assert!((ad.eval(dfdy).unwrap() - 2.0).abs() < 1e-10);
}

// ============================================================================
// CompiledGraph with DiffNum functions
// ============================================================================

#[test]
fn diffnum_compiled_quadratic() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();
    let f = with_context(&mut ad, || quadratic(x));

    let mut cg = ad.compile_primal(f, &[x]).unwrap();
    cg.eval(&[3.0]).unwrap();
    assert_eq!(cg.value(), 16.0);

    cg.eval(&[0.0]).unwrap();
    assert_eq!(cg.value(), 1.0);
}

#[test]
fn diffnum_compiled_gradient() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let f = with_context(&mut ad, || rosenbrock(x, y));

    let mut cg = ad.compile_primal(f, &[x, y]).unwrap();
    cg.eval(&[1.0, 1.0]).unwrap();
    assert_eq!(cg.value(), 0.0);

    let grad = cg.gradient();
    assert!((grad[0]).abs() < 1e-10);
    assert!((grad[1]).abs() < 1e-10);
}

#[test]
fn diffnum_compiled_with_order() {
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(0.0).unwrap();
    let f = with_context(&mut ad, || quadratic(x));

    let mut cg = ad.compile_order(f, &[x], 2).unwrap();
    cg.eval(&[3.0]).unwrap();

    assert_eq!(cg.value(), 16.0);
    assert!((cg.partial(&[1]).unwrap() - 8.0).abs() < 1e-10); // f'(3) = 2*3+2 = 8
    assert!((cg.partial(&[2]).unwrap() - 2.0).abs() < 1e-10); // f''(3) = 2
}

// ============================================================================
// DiffNum log variants with Var
// ============================================================================

#[test]
fn diffnum_pow_log_var() {
    fn f<T: DiffNum>(x: T) -> T {
        x.pow_log(T::from_f64(3.0))
    }

    // Direct: 2^3 = 8
    assert_eq!(f(2.0_f64), 8.0);

    // Via AD
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let result = with_context(&mut ad, || f(x));
    assert!((ad.eval(result).unwrap() - 8.0).abs() < 1e-10);
}

#[test]
fn diffnum_div_log_var() {
    fn f<T: DiffNum>(x: T, y: T) -> T {
        x.div_log(y)
    }

    // Direct: 6/2 = 3
    assert_eq!(f(6.0_f64, 2.0_f64), 3.0);

    // Via AD
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(6.0).unwrap();
    let y = ad.var(2.0).unwrap();
    let result = with_context(&mut ad, || f(x, y));
    assert!((ad.eval(result).unwrap() - 3.0).abs() < 1e-10);
}

#[test]
fn diffnum_powi_log_var() {
    fn f<T: DiffNum>(x: T) -> T {
        x.powi_log(3)
    }

    // Direct: 2^3 = 8
    assert_eq!(f(2.0_f64), 8.0);

    // Via AD
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(2.0).unwrap();
    let result = with_context(&mut ad, || f(x));
    assert!((ad.eval(result).unwrap() - 8.0).abs() < 1e-10);
}

// ============================================================================
// Consistency: same function gives same result for f64 and Var
// ============================================================================

#[test]
fn diffnum_f64_var_consistency() {
    // Verify that evaluating a DiffNum function with f64 and with Var+eval
    // gives the same result for various inputs.
    let test_vals = [0.5_f64, 1.0, 2.0, 3.0];

    for &x_val in &test_vals {
        let f64_result = quadratic(x_val);

        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(x_val).unwrap();
        let f = with_context(&mut ad, || quadratic(x));
        let var_result = ad.eval(f).unwrap();

        assert!(
            (f64_result - var_result).abs() < 1e-10,
            "Mismatch at x={x_val}: f64={f64_result}, var={var_result}"
        );
    }
}

#[test]
fn diffnum_f64_var_consistency_multivar() {
    let pairs = [(1.0, 2.0), (0.5, 0.5), (2.0, 3.0)];

    for &(x_val, y_val) in &pairs {
        let f64_result = two_body(x_val, y_val);

        let mut ad = AutoDiff::<f64>::new();
        let x = ad.var(x_val).unwrap();
        let y = ad.var(y_val).unwrap();
        let f = with_context(&mut ad, || two_body(x, y));
        let var_result = ad.eval(f).unwrap();

        assert!(
            (f64_result - var_result).abs() < 1e-10,
            "Mismatch at ({x_val}, {y_val}): f64={f64_result}, var={var_result}"
        );
    }
}
