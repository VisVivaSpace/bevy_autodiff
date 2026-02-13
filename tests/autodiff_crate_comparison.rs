//! Comparison tests against the `autodiff` crate (elrnv/autodiff)
//!
//! These tests validate bevy_autodiff's first-order derivatives against
//! an independent forward-mode AD implementation.

use approx::assert_relative_eq;
use autodiff::Float;
use autodiff::F1 as F;
use bevy_autodiff::AutoDiff;

/// Helper: compare bevy_autodiff's derivative against autodiff crate's derivative
/// for a univariate function.
fn compare_unary<BevyFn, AutodiffFn>(x_val: f64, bevy_fn: BevyFn, autodiff_fn: AutodiffFn)
where
    BevyFn: Fn(&mut AutoDiff, bevy_autodiff::Var) -> bevy_autodiff::Var,
    AutodiffFn: Fn(F) -> F,
{
    let mut ad = AutoDiff::new();
    let x = ad.var(x_val);
    let f = bevy_fn(&mut ad, x);
    let df = ad.differentiate(f, x);
    let bevy_deriv = ad.eval(df);

    let x_ad = F::var(x_val);
    let result = autodiff_fn(x_ad);
    let autodiff_deriv = result.deriv();

    assert_relative_eq!(bevy_deriv, autodiff_deriv, epsilon = 1e-10,);
}

// ============================================================================
// Elementary functions
// ============================================================================

#[test]
fn compare_sin() {
    for &x in &[0.0, 0.5, 1.0, -1.0, 2.0] {
        compare_unary(x, |ad, x| ad.sin(x), |x| x.sin());
    }
}

#[test]
fn compare_cos() {
    for &x in &[0.0, 0.5, 1.0, -1.0, 2.0] {
        compare_unary(x, |ad, x| ad.cos(x), |x| x.cos());
    }
}

#[test]
fn compare_exp() {
    for &x in &[0.0, 0.5, 1.0, -1.0, 2.0] {
        compare_unary(x, |ad, x| ad.exp(x), |x| x.exp());
    }
}

#[test]
fn compare_ln() {
    for &x in &[0.1, 0.5, 1.0, 2.0, 10.0] {
        compare_unary(x, |ad, x| ad.ln(x), |x| x.ln());
    }
}

#[test]
fn compare_sqrt() {
    for &x in &[0.1, 0.5, 1.0, 4.0, 9.0] {
        compare_unary(x, |ad, x| ad.sqrt(x), |x| x.sqrt());
    }
}

#[test]
fn compare_tan() {
    for &x in &[0.0, 0.5, 1.0, -0.5] {
        compare_unary(x, |ad, x| ad.tan(x), |x| x.tan());
    }
}

#[test]
fn compare_sinh() {
    for &x in &[0.0, 0.5, 1.0, -1.0] {
        compare_unary(x, |ad, x| ad.sinh(x), |x| x.sinh());
    }
}

#[test]
fn compare_cosh() {
    for &x in &[0.0, 0.5, 1.0, -1.0] {
        compare_unary(x, |ad, x| ad.cosh(x), |x| x.cosh());
    }
}

#[test]
fn compare_tanh() {
    for &x in &[0.0, 0.5, 1.0, -1.0] {
        compare_unary(x, |ad, x| ad.tanh(x), |x| x.tanh());
    }
}

#[test]
fn compare_asin() {
    for &x in &[0.0, 0.5, -0.5, 0.9] {
        compare_unary(x, |ad, x| ad.asin(x), |x| x.asin());
    }
}

#[test]
fn compare_acos() {
    for &x in &[0.0, 0.5, -0.5, 0.9] {
        compare_unary(x, |ad, x| ad.acos(x), |x| x.acos());
    }
}

#[test]
fn compare_atan() {
    for &x in &[0.0, 0.5, 1.0, -1.0, 2.0] {
        compare_unary(x, |ad, x| ad.atan(x), |x| x.atan());
    }
}

// ============================================================================
// Compositions
// ============================================================================

#[test]
fn compare_exp_sin() {
    // d/dx(exp(sin(x)))
    for &x_val in &[0.0, 0.5, 1.0] {
        compare_unary(
            x_val,
            |ad, x| {
                let s = ad.sin(x);
                ad.exp(s)
            },
            |x| x.sin().exp(),
        );
    }
}

#[test]
fn compare_sin_square() {
    // d/dx(sin(x²))
    for &x_val in &[0.5, 1.0, 2.0] {
        compare_unary(
            x_val,
            |ad, x| {
                let x2 = ad.square(x);
                ad.sin(x2)
            },
            |x| (x * x).sin(),
        );
    }
}

#[test]
fn compare_ln_cos() {
    // d/dx(ln(cos(x))) for small x where cos(x) > 0
    for &x_val in &[0.1, 0.3, 0.5] {
        compare_unary(
            x_val,
            |ad, x| {
                let c = ad.cos(x);
                ad.ln(c)
            },
            |x| x.cos().ln(),
        );
    }
}

// ============================================================================
// Arithmetic combinations
// ============================================================================

#[test]
fn compare_product_rule() {
    // d/dx(x * sin(x))
    for &x_val in &[0.5, 1.0, 2.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let sin_x = ad.sin(x);
        let f = ad.mul(x, sin_x);
        let df = ad.differentiate(f, x);
        let bevy_deriv = ad.eval(df);

        let x_ad = F::var(x_val);
        let result = x_ad * x_ad.sin();
        let autodiff_deriv = result.deriv();

        assert_relative_eq!(bevy_deriv, autodiff_deriv, epsilon = 1e-10);
    }
}

#[test]
fn compare_quotient_rule() {
    // d/dx(sin(x) / x)
    for &x_val in &[0.5, 1.0, 2.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let sin_x = ad.sin(x);
        let f = ad.div(sin_x, x);
        let df = ad.differentiate(f, x);
        let bevy_deriv = ad.eval(df);

        let x_ad = F::var(x_val);
        let result = x_ad.sin() / x_ad;
        let autodiff_deriv = result.deriv();

        assert_relative_eq!(bevy_deriv, autodiff_deriv, epsilon = 1e-10);
    }
}

#[test]
fn compare_polynomial() {
    // d/dx(x³ + 2x² + 3x + 4)
    for &x_val in &[0.0, 1.0, 2.0, -1.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let c2 = ad.constant(2.0);
        let c3 = ad.constant(3.0);
        let c4 = ad.constant(4.0);

        let x2 = ad.square(x);
        let x3 = ad.mul(x2, x);
        let term2 = ad.mul(c2, x2);
        let term1 = ad.mul(c3, x);
        let s0 = ad.add(term1, c4);
        let s1 = ad.add(term2, s0);
        let f = ad.add(x3, s1);
        let df = ad.differentiate(f, x);
        let bevy_deriv = ad.eval(df);

        let x_ad = F::var(x_val);
        let result =
            x_ad * x_ad * x_ad + F::cst(2.0) * x_ad * x_ad + F::cst(3.0) * x_ad + F::cst(4.0);
        let autodiff_deriv = result.deriv();

        assert_relative_eq!(bevy_deriv, autodiff_deriv, epsilon = 1e-10);
    }
}

// ============================================================================
// Power function
// ============================================================================

#[test]
fn compare_powf() {
    // d/dx(x^2.5) = 2.5 * x^1.5
    for &x_val in &[0.5, 1.0, 2.0, 4.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let f = ad.powf(x, 2.5);
        let df = ad.differentiate(f, x);
        let bevy_deriv = ad.eval(df);

        let x_ad = F::var(x_val);
        let result = x_ad.powf(F::cst(2.5));
        let autodiff_deriv = result.deriv();

        assert_relative_eq!(bevy_deriv, autodiff_deriv, epsilon = 1e-9);
    }
}
