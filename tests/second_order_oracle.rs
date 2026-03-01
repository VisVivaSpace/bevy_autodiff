//! Second-order oracle tests against known closed-form values.
//!
//! These tests directly compare d²f/dx² (and selected mixed partials) against
//! analytic formulas for transcendental and power functions. They fill the gap
//! where only Schwarz symmetry (d²f/dxdy = d²f/dydx) was used to validate
//! second-order correctness — never against independent closed-form values.

use approx::assert_relative_eq;
use bevy_autodiff::AutoDiff;

const EPS: f64 = 1e-10;

// ============================================================================
// Helper: compute d²f/dx² at x_val
// ============================================================================

fn second_deriv<F>(x_val: f64, build: F) -> f64
where
    F: Fn(&mut AutoDiff<f64>, bevy_autodiff::Var) -> bevy_autodiff::Var,
{
    let mut ad = AutoDiff::new();
    let x = ad.var(x_val).unwrap();
    let f = build(&mut ad, x);
    ad.derivative(f, x, 2).unwrap()
}

// ============================================================================
// Helper: compute d²f/dxdy at (x_val, y_val)
// ============================================================================

fn mixed_partial<F>(x_val: f64, y_val: f64, build: F) -> f64
where
    F: Fn(&mut AutoDiff<f64>, bevy_autodiff::Var, bevy_autodiff::Var) -> bevy_autodiff::Var,
{
    let mut ad = AutoDiff::new();
    let x = ad.var(x_val).unwrap();
    let y = ad.var(y_val).unwrap();
    let f = build(&mut ad, x, y);
    ad.partial(f, &[1, 1], &[x, y]).unwrap()
}

// ============================================================================
// Transcendental second derivatives
// ============================================================================

#[test]
fn second_order_sin() {
    // d²[sin(x)]/dx² = -sin(x)
    for &x in &[0.5, 1.0, 2.0, 3.0] {
        let got = second_deriv(x, |ad, x| ad.sin(x));
        assert_relative_eq!(got, -x.sin(), epsilon = EPS);
    }
}

#[test]
fn second_order_cos() {
    // d²[cos(x)]/dx² = -cos(x)
    for &x in &[0.5, 1.0, 2.0, 3.0] {
        let got = second_deriv(x, |ad, x| ad.cos(x));
        assert_relative_eq!(got, -x.cos(), epsilon = EPS);
    }
}

#[test]
fn second_order_exp() {
    // d²[exp(x)]/dx² = exp(x)
    for &x in &[0.5, 1.0, 1.5, 2.0] {
        let got = second_deriv(x, |ad, x| ad.exp(x));
        assert_relative_eq!(got, x.exp(), epsilon = EPS);
    }
}

#[test]
fn second_order_ln() {
    // d²[ln(x)]/dx² = -1/x²
    for &x in &[0.5, 1.0, 2.0, 3.0] {
        let got = second_deriv(x, |ad, x| ad.ln(x));
        assert_relative_eq!(got, -1.0 / (x * x), epsilon = EPS);
    }
}

#[test]
fn second_order_sqrt() {
    // d²[sqrt(x)]/dx² = -1/(4*x^(3/2))
    for &x in &[0.5, 1.0, 2.0, 4.0] {
        let got = second_deriv(x, |ad, x| ad.sqrt(x));
        assert_relative_eq!(got, -1.0 / (4.0 * x.powf(1.5)), epsilon = EPS);
    }
}

#[test]
fn second_order_x_cubed() {
    // d²[x³]/dx² = 6x  (via powi)
    for &x in &[0.5, 1.0, 1.5, 2.0] {
        let got = second_deriv(x, |ad, x| ad.powi(x, 3));
        assert_relative_eq!(got, 6.0 * x, epsilon = EPS);
    }
}

#[test]
fn second_order_sinh() {
    // d²[sinh(x)]/dx² = sinh(x)
    for &x in &[0.5, 1.0, 1.5] {
        let got = second_deriv(x, |ad, x| ad.sinh(x));
        assert_relative_eq!(got, x.sinh(), epsilon = EPS);
    }
}

#[test]
fn second_order_cosh() {
    // d²[cosh(x)]/dx² = cosh(x)
    for &x in &[0.5, 1.0, 1.5] {
        let got = second_deriv(x, |ad, x| ad.cosh(x));
        assert_relative_eq!(got, x.cosh(), epsilon = EPS);
    }
}

#[test]
fn second_order_atan() {
    // d²[atan(x)]/dx² = -2x/(1+x²)²
    for &x in &[0.5, 1.0, 2.0] {
        let got = second_deriv(x, |ad, x| ad.atan(x));
        let expected = -2.0 * x / (1.0 + x * x).powi(2);
        assert_relative_eq!(got, expected, epsilon = EPS);
    }
}

#[test]
fn second_order_tanh() {
    // d²[tanh(x)]/dx² = -2*tanh(x)/cosh²(x)
    for &x in &[0.5, 1.0, 1.5] {
        let got = second_deriv(x, |ad, x| ad.tanh(x));
        let expected = -2.0 * x.tanh() / (x.cosh() * x.cosh());
        assert_relative_eq!(got, expected, epsilon = EPS);
    }
}

// ============================================================================
// Logarithmic derivative second-order: match closed-form and standard variants
// ============================================================================

#[test]
fn second_order_powi_log_matches_powi() {
    // d²[powi_log(x,3)]/dx² = d²[x³]/dx² = 6x
    // Confirms powi_log gives identical second derivatives to powi.
    for &x in &[0.5, 1.0, 1.5, 2.0] {
        let got_log = second_deriv(x, |ad, x| ad.powi_log(x, 3));
        let got_std = second_deriv(x, |ad, x| ad.powi(x, 3));
        assert_relative_eq!(got_log, 6.0 * x, epsilon = EPS);
        assert_relative_eq!(got_log, got_std, epsilon = EPS);
    }
}

#[test]
fn second_order_powf_log_matches_powf() {
    // d²[powf_log(x,2.5)]/dx² = 2.5*1.5*x^0.5
    // Confirms powf_log gives identical second derivatives to powf.
    for &x in &[0.5, 1.0, 2.0, 4.0] {
        let got_log = second_deriv(x, |ad, x| ad.powf_log(x, 2.5));
        let got_std = second_deriv(x, |ad, x| ad.powf(x, 2.5));
        let expected = 2.5 * 1.5 * x.powf(0.5);
        assert_relative_eq!(got_log, expected, epsilon = EPS);
        assert_relative_eq!(got_log, got_std, epsilon = EPS);
    }
}

#[test]
fn second_order_div_log_const_denominator() {
    // d²[div_log(x, c)]/dx² = 0 for constant c
    // (x/c is linear in x, so its second derivative is zero)
    for &(x, c) in &[(1.0_f64, 2.0_f64), (3.0, 1.5), (0.5, 4.0)] {
        let mut ad = AutoDiff::new();
        let xv = ad.var(x).unwrap();
        let cv = ad.constant(c);
        let f = ad.div_log(xv, cv);
        let got = ad.derivative(f, xv, 2).unwrap();
        assert_relative_eq!(got, 0.0, epsilon = EPS);
    }
}

#[test]
fn second_order_pow_log_wrt_x() {
    // d²[pow_log(x,y)]/dx² = y*(y-1)*x^(y-2)  (y treated as fixed)
    // Verified via successive differentiation.
    for &(x, y) in &[(2.0_f64, 3.0_f64), (1.5, 2.0), (3.0, 0.5)] {
        let mut ad = AutoDiff::new();
        let xv = ad.var(x).unwrap();
        let yv = ad.var(y).unwrap();
        let f = ad.pow_log(xv, yv);
        let df = ad.differentiate(f, xv).unwrap();
        let d2f = ad.differentiate(df, xv).unwrap();
        let got = ad.eval(d2f).unwrap();
        let expected = y * (y - 1.0) * x.powf(y - 2.0);
        assert_relative_eq!(got, expected, epsilon = 1e-9);
    }
}

// ============================================================================
// Mixed partial oracle: d²f/dxdy against closed-form values
// ============================================================================

#[test]
fn mixed_partial_independent_sum() {
    // f(x,y) = exp(x) + cos(y) → d²f/dxdy = 0
    // The two terms are independent, so cross-differentiation collapses to zero.
    for &(x, y) in &[(0.5_f64, 1.0_f64), (1.0, 0.5), (2.0, 0.3)] {
        let got = mixed_partial(x, y, |ad, x, y| {
            let ex = ad.exp(x);
            let cy = ad.cos(y);
            ad.add(ex, cy)
        });
        assert_relative_eq!(got, 0.0, epsilon = EPS);
    }
}

#[test]
fn mixed_partial_product_x2_y3() {
    // f(x,y) = x² * y³ → d²f/dxdy = 6xy²
    for &(x, y) in &[(1.0_f64, 1.0_f64), (2.0, 3.0), (0.5, 2.0)] {
        let got = mixed_partial(x, y, |ad, x, y| {
            let x2 = ad.square(x);
            let y2 = ad.square(y);
            let y3 = ad.mul(y2, y);
            ad.mul(x2, y3)
        });
        let expected = 6.0 * x * y * y;
        assert_relative_eq!(got, expected, epsilon = EPS);
    }
}

#[test]
fn mixed_partial_sin_sum() {
    // f(x,y) = sin(x+y) → d²f/dxdy = -sin(x+y)
    // Also equals d²f/dx² — both directions give the same result for f(x+y).
    for &(x, y) in &[(0.5_f64, 0.3_f64), (1.0, 0.5), (0.3, 1.2)] {
        let got = mixed_partial(x, y, |ad, x, y| {
            let s = ad.add(x, y);
            ad.sin(s)
        });
        let expected = -(x + y).sin();
        assert_relative_eq!(got, expected, epsilon = EPS);
    }
}

#[test]
fn mixed_partial_rational() {
    // f(x,y) = x/(x²+y²) → d²f/dxdy = 2y(3x²-y²)/(x²+y²)³
    for &(x, y) in &[(1.0_f64, 1.0_f64), (2.0, 1.0), (1.0, 2.0)] {
        let got = mixed_partial(x, y, |ad, x, y| {
            let x2 = ad.square(x);
            let y2 = ad.square(y);
            let r2 = ad.add(x2, y2);
            ad.div(x, r2)
        });
        let r2 = x * x + y * y;
        let expected = 2.0 * y * (3.0 * x * x - y * y) / (r2 * r2 * r2);
        assert_relative_eq!(got, expected, epsilon = 1e-9);
    }
}
