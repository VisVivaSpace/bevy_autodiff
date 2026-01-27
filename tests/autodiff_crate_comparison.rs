//! Comparison tests against the `autodiff` crate (elrnv/autodiff)
//!
//! This module validates vvad's first-order derivatives against the `autodiff` crate,
//! which provides forward-mode automatic differentiation. Both implementations should
//! produce identical derivatives for the same mathematical functions.
//!
//! # Running the Tests
//!
//! ```bash
//! cargo test --test autodiff_crate_comparison
//! ```

use approx::assert_relative_eq;
use autodiff::diff;
use autodiff::Float;
use autodiff::FT;
use vvad::AutoDiff;

/// Helper type alias for clearer code
type Dual = FT<f64>;

/// Helper to create a constant in autodiff
fn cst(x: f64) -> Dual {
    Dual::cst(x)
}

// =============================================================================
// Basic Trigonometric Functions
// =============================================================================

#[test]
fn test_sin_derivative() {
    let test_points = [0.0, 0.5, 1.0, 2.0, std::f64::consts::PI, -1.0];

    for x in test_points {
        // autodiff crate reference
        let deriv_autodiff = diff(|t: FT<f64>| t.sin(), x);

        // vvad computation
        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.sin(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_cos_derivative() {
    let test_points = [0.0, 0.5, 1.0, 2.0, std::f64::consts::PI, -1.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.cos(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.cos(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_tan_derivative() {
    // Avoid points near π/2 where tan has singularities
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.tan(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.tan(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Inverse Trigonometric Functions
// =============================================================================

#[test]
fn test_asin_derivative() {
    // Domain: |x| < 1
    let test_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.asin(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.asin(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_acos_derivative() {
    // Domain: |x| < 1
    let test_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.acos(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.acos(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_atan_derivative() {
    // atan is defined for all real numbers
    let test_points = [-10.0, -1.0, 0.0, 1.0, 10.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.atan(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.atan(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Hyperbolic Functions
// =============================================================================

#[test]
fn test_sinh_derivative() {
    let test_points = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.sinh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.sinh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_cosh_derivative() {
    let test_points = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.cosh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.cosh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_tanh_derivative() {
    let test_points = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.tanh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.tanh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Inverse Hyperbolic Functions
// =============================================================================

#[test]
fn test_asinh_derivative() {
    // asinh is defined for all real numbers
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.asinh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.asinh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_acosh_derivative() {
    // Domain: x > 1
    let test_points = [1.1, 1.5, 2.0, 3.0, 5.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.acosh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.acosh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_atanh_derivative() {
    // Domain: |x| < 1
    let test_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.atanh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.atanh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Exponential and Logarithmic Functions
// =============================================================================

#[test]
fn test_exp_derivative() {
    let test_points = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.exp(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.exp(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_ln_derivative() {
    // Only positive values for ln
    let test_points = [0.1, 0.5, 1.0, 2.0, std::f64::consts::E, 10.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.ln(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.ln(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_sqrt_derivative() {
    // Only positive values for sqrt
    let test_points = [0.25, 1.0, 4.0, 9.0, 16.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.sqrt(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.sqrt(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Composition Tests
// =============================================================================

#[test]
fn test_sin_of_exp() {
    let test_points = [-1.0, 0.0, 0.5, 1.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.exp().sin(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let exp_x = ad.exp(xv);
        let f = ad.sin(exp_x);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_exp_of_sin() {
    let test_points = [-1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.sin().exp(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let sin_x = ad.sin(xv);
        let f = ad.exp(sin_x);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_sqrt_of_one_plus_x_squared() {
    // f(x) = sqrt(1 + x^2), which is related to asinh
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(
            |t: Dual| (cst(1.0) + t * t).sqrt(),
            x,
        );

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let one = ad.constant(1.0);
        let x_sq = ad.mul(xv, xv);
        let sum = ad.add(one, x_sq);
        let f = ad.sqrt(sum);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_atan_of_exp() {
    let test_points = [-1.0, 0.0, 0.5, 1.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.exp().atan(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let exp_x = ad.exp(xv);
        let f = ad.atan(exp_x);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_tanh_of_sin() {
    let test_points = [-1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.sin().tanh(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let sin_x = ad.sin(xv);
        let f = ad.tanh(sin_x);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Polynomial and Arithmetic Tests
// =============================================================================

#[test]
fn test_polynomial() {
    // f(x) = x^3 - 2x^2 + 3x - 4
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

    for x in test_points {
        let deriv_autodiff = diff(
            |t: Dual| {
                t * t * t - cst(2.0) * t * t + cst(3.0) * t - cst(4.0)
            },
            x,
        );

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let two = ad.constant(2.0);
        let three = ad.constant(3.0);
        let four = ad.constant(4.0);

        let x2 = ad.mul(xv, xv);
        let x3 = ad.mul(x2, xv);
        let two_x2 = ad.mul(two, x2);
        let three_x = ad.mul(three, xv);

        let t1 = ad.sub(x3, two_x2);
        let t2 = ad.add(t1, three_x);
        let f = ad.sub(t2, four);

        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_division() {
    // f(x) = 1 / (1 + x^2), which is atan'(x)
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        let deriv_autodiff = diff(
            |t: Dual| cst(1.0) / (cst(1.0) + t * t),
            x,
        );

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let one = ad.constant(1.0);
        let x_sq = ad.mul(xv, xv);
        let denom = ad.add(one, x_sq);
        let f = ad.div(one, denom);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

#[test]
fn test_quotient() {
    // f(x) = sin(x) / cos(x) = tan(x)
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: FT<f64>| t.sin() / t.cos(), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let sin_x = ad.sin(xv);
        let cos_x = ad.cos(xv);
        let f = ad.div(sin_x, cos_x);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-10,
            max_relative = 1e-10
        );
    }
}

// =============================================================================
// Power Function Tests
// =============================================================================

#[test]
fn test_power_function() {
    // f(x) = x^2.5
    let test_points = [0.5, 1.0, 2.0, 3.0, 4.0];

    for x in test_points {
        let deriv_autodiff = diff(|t: Dual| t.powf(cst(2.5)), x);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let exp = ad.constant(2.5);
        let f = ad.pow(xv, exp);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_autodiff,
            epsilon = 1e-9,
            max_relative = 1e-9
        );
    }
}
