//! Comparison tests against std::autodiff
//!
//! This module provides reference tests using Rust's experimental std::autodiff
//! feature (powered by LLVM/Enzyme) as an oracle to validate vvad's first-order
//! derivatives against an independent AD implementation.
//!
//! # Requirements
//!
//! These tests require:
//! - Nightly Rust with Enzyme support
//! - The `std_autodiff_tests` feature enabled
//!
//! # Running the Tests
//!
//! ```bash
//! # Install the Enzyme toolchain if not already installed
//! rustup install enzyme
//!
//! # Run the tests
//! RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme test --features std_autodiff_tests
//! ```
//!
//! # Purpose
//!
//! These tests serve as oracle validation:
//! - std::autodiff uses LLVM/Enzyme for compile-time code transformation
//! - vvad uses runtime computation graphs with Taylor series propagation
//! - Agreement between both approaches validates correctness
//!
//! Note: std::autodiff only supports first-order derivatives, so these tests
//! only validate first derivatives. vvad's higher-order derivatives are tested
//! separately using mathematical identities.

#![cfg(feature = "std_autodiff_tests")]
#![feature(autodiff)]

use approx::assert_relative_eq;
use std::autodiff::autodiff_forward;
use std::autodiff::autodiff_reverse;
use vvad::AutoDiff;

// =============================================================================
// Reference implementations using std::autodiff
// =============================================================================

// Elementary function derivatives (forward mode)

#[autodiff_forward(sin_deriv_std, Dual, Dual)]
fn sin_fn(x: f64) -> f64 {
    x.sin()
}

#[autodiff_forward(cos_deriv_std, Dual, Dual)]
fn cos_fn(x: f64) -> f64 {
    x.cos()
}

#[autodiff_forward(exp_deriv_std, Dual, Dual)]
fn exp_fn(x: f64) -> f64 {
    x.exp()
}

#[autodiff_forward(ln_deriv_std, Dual, Dual)]
fn ln_fn(x: f64) -> f64 {
    x.ln()
}

#[autodiff_forward(sqrt_deriv_std, Dual, Dual)]
fn sqrt_fn(x: f64) -> f64 {
    x.sqrt()
}

#[autodiff_forward(sinh_deriv_std, Dual, Dual)]
fn sinh_fn(x: f64) -> f64 {
    x.sinh()
}

#[autodiff_forward(cosh_deriv_std, Dual, Dual)]
fn cosh_fn(x: f64) -> f64 {
    x.cosh()
}

#[autodiff_forward(tan_deriv_std, Dual, Dual)]
fn tan_fn(x: f64) -> f64 {
    x.tan()
}

#[autodiff_forward(tanh_deriv_std, Dual, Dual)]
fn tanh_fn(x: f64) -> f64 {
    x.tanh()
}

#[autodiff_forward(asin_deriv_std, Dual, Dual)]
fn asin_fn(x: f64) -> f64 {
    x.asin()
}

#[autodiff_forward(acos_deriv_std, Dual, Dual)]
fn acos_fn(x: f64) -> f64 {
    x.acos()
}

#[autodiff_forward(atan_deriv_std, Dual, Dual)]
fn atan_fn(x: f64) -> f64 {
    x.atan()
}

#[autodiff_forward(asinh_deriv_std, Dual, Dual)]
fn asinh_fn(x: f64) -> f64 {
    x.asinh()
}

#[autodiff_forward(acosh_deriv_std, Dual, Dual)]
fn acosh_fn(x: f64) -> f64 {
    x.acosh()
}

#[autodiff_forward(atanh_deriv_std, Dual, Dual)]
fn atanh_fn(x: f64) -> f64 {
    x.atanh()
}

// Gradient computations (reverse mode)

#[autodiff_reverse(rosenbrock_grad_std, Active, Active, Active)]
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x.powi(2)).powi(2)
}

#[autodiff_reverse(polynomial_grad_std, Active, Active, Active, Active)]
fn polynomial(x: f64, y: f64, z: f64) -> f64 {
    x * x + y * y * y + z * x * y
}

#[autodiff_reverse(sum_of_squares_grad_std, Active, Active, Active)]
fn sum_of_squares(x: f64, y: f64) -> f64 {
    x * x + y * y
}

#[autodiff_reverse(product_grad_std, Active, Active, Active)]
fn product(x: f64, y: f64) -> f64 {
    x * y
}

// =============================================================================
// Elementary Function Tests
// =============================================================================

#[test]
fn test_sin_derivative() {
    let test_points = [0.0, 0.5, 1.0, 2.0, std::f64::consts::PI, -1.0];

    for x in test_points {
        // std::autodiff reference
        let (_, deriv_std) = sin_deriv_std(x, 1.0);

        // vvad computation
        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.sin(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "sin derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_cos_derivative() {
    let test_points = [0.0, 0.5, 1.0, 2.0, std::f64::consts::PI, -1.0];

    for x in test_points {
        let (_, deriv_std) = cos_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.cos(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "cos derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_exp_derivative() {
    let test_points = [-1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let (_, deriv_std) = exp_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.exp(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "exp derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_ln_derivative() {
    // Only positive values for ln
    let test_points = [0.5, 1.0, 2.0, std::f64::consts::E, 10.0];

    for x in test_points {
        let (_, deriv_std) = ln_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.ln(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "ln derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_sqrt_derivative() {
    // Only positive values for sqrt
    let test_points = [0.25, 1.0, 4.0, 9.0, 16.0];

    for x in test_points {
        let (_, deriv_std) = sqrt_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.sqrt(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "sqrt derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_sinh_derivative() {
    let test_points = [-1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let (_, deriv_std) = sinh_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.sinh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "sinh derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_cosh_derivative() {
    let test_points = [-1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let (_, deriv_std) = cosh_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.cosh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "cosh derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_tan_derivative() {
    // Avoid points near π/2 where tan has singularities
    let test_points = [-1.0, -0.5, 0.0, 0.5, 1.0];

    for x in test_points {
        let (_, deriv_std) = tan_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.tan(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "tan derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_tanh_derivative() {
    let test_points = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let (_, deriv_std) = tanh_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.tanh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "tanh derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_asin_derivative() {
    // Domain: |x| < 1
    let test_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    for x in test_points {
        let (_, deriv_std) = asin_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.asin(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "asin derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_acos_derivative() {
    // Domain: |x| < 1
    let test_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    for x in test_points {
        let (_, deriv_std) = acos_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.acos(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "acos derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_atan_derivative() {
    // atan is defined for all real numbers
    let test_points = [-10.0, -1.0, 0.0, 1.0, 10.0];

    for x in test_points {
        let (_, deriv_std) = atan_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.atan(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "atan derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_asinh_derivative() {
    // asinh is defined for all real numbers
    let test_points = [-2.0, -1.0, 0.0, 1.0, 2.0];

    for x in test_points {
        let (_, deriv_std) = asinh_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.asinh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "asinh derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_acosh_derivative() {
    // Domain: x > 1
    let test_points = [1.1, 1.5, 2.0, 3.0, 5.0];

    for x in test_points {
        let (_, deriv_std) = acosh_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.acosh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "acosh derivative mismatch at x={}",
            x
        );
    }
}

#[test]
fn test_atanh_derivative() {
    // Domain: |x| < 1
    let test_points = [-0.9, -0.5, 0.0, 0.5, 0.9];

    for x in test_points {
        let (_, deriv_std) = atanh_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let f = ad.atanh(xv);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "atanh derivative mismatch at x={}",
            x
        );
    }
}

// =============================================================================
// Multi-Variable Gradient Tests
// =============================================================================

#[test]
fn test_rosenbrock_gradient() {
    let test_cases = [(1.0, 1.0), (0.0, 0.0), (1.0, 3.0), (-1.0, 2.0), (2.0, 4.0)];

    for (x_val, y_val) in test_cases {
        // std::autodiff reference
        let (val_std, dx_std, dy_std) = rosenbrock_grad_std(x_val, y_val, 1.0);

        // vvad computation
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let y = ad.var(y_val);

        // f = (1 - x)^2 + 100 * (y - x^2)^2
        let one = ad.constant(1.0);
        let hundred = ad.constant(100.0);
        let one_minus_x = ad.sub(one, x);
        let term1 = ad.square(one_minus_x);
        let x_sq = ad.square(x);
        let y_minus_x_sq = ad.sub(y, x_sq);
        let term2_inner = ad.square(y_minus_x_sq);
        let term2 = ad.mul(hundred, term2_inner);
        let f = ad.add(term1, term2);

        let val_vvad = ad.eval(f);
        let grad_vvad = ad.gradient_reverse(f);

        assert_relative_eq!(
            val_vvad,
            val_std,
            epsilon = 1e-10,
            "Rosenbrock value mismatch at ({}, {})",
            x_val,
            y_val
        );
        assert_relative_eq!(
            grad_vvad[0],
            dx_std,
            epsilon = 1e-10,
            "Rosenbrock dx mismatch at ({}, {})",
            x_val,
            y_val
        );
        assert_relative_eq!(
            grad_vvad[1],
            dy_std,
            epsilon = 1e-10,
            "Rosenbrock dy mismatch at ({}, {})",
            x_val,
            y_val
        );
    }
}

#[test]
fn test_polynomial_gradient() {
    let test_cases = [
        (2.0, 3.0, 4.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (-1.0, 2.0, 3.0),
    ];

    for (x_val, y_val, z_val) in test_cases {
        // std::autodiff reference
        // f(x, y, z) = x^2 + y^3 + z*x*y
        let (val_std, dx_std, dy_std, dz_std) =
            polynomial_grad_std(x_val, y_val, z_val, 1.0);

        // vvad computation
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let y = ad.var(y_val);
        let z = ad.var(z_val);

        // f = x^2 + y^3 + z*x*y
        let x2 = ad.square(x);
        let y3 = ad.mul(ad.mul(y, y), y);
        let xy = ad.mul(x, y);
        let zxy = ad.mul(z, xy);
        let sum1 = ad.add(x2, y3);
        let f = ad.add(sum1, zxy);

        let val_vvad = ad.eval(f);
        let grad_vvad = ad.gradient_reverse(f);

        assert_relative_eq!(
            val_vvad,
            val_std,
            epsilon = 1e-10,
            "Polynomial value mismatch at ({}, {}, {})",
            x_val,
            y_val,
            z_val
        );
        assert_relative_eq!(
            grad_vvad[0],
            dx_std,
            epsilon = 1e-10,
            "Polynomial dx mismatch at ({}, {}, {})",
            x_val,
            y_val,
            z_val
        );
        assert_relative_eq!(
            grad_vvad[1],
            dy_std,
            epsilon = 1e-10,
            "Polynomial dy mismatch at ({}, {}, {})",
            x_val,
            y_val,
            z_val
        );
        assert_relative_eq!(
            grad_vvad[2],
            dz_std,
            epsilon = 1e-10,
            "Polynomial dz mismatch at ({}, {}, {})",
            x_val,
            y_val,
            z_val
        );
    }
}

#[test]
fn test_sum_of_squares_gradient() {
    let test_cases = [(1.0, 2.0), (3.0, 4.0), (0.0, 0.0), (-1.0, 1.0)];

    for (x_val, y_val) in test_cases {
        // std::autodiff reference
        let (val_std, dx_std, dy_std) = sum_of_squares_grad_std(x_val, y_val, 1.0);

        // vvad computation
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let y = ad.var(y_val);

        let x2 = ad.square(x);
        let y2 = ad.square(y);
        let f = ad.add(x2, y2);

        let val_vvad = ad.eval(f);
        let grad_vvad = ad.gradient_reverse(f);

        assert_relative_eq!(val_vvad, val_std, epsilon = 1e-10);
        assert_relative_eq!(grad_vvad[0], dx_std, epsilon = 1e-10);
        assert_relative_eq!(grad_vvad[1], dy_std, epsilon = 1e-10);
    }
}

#[test]
fn test_product_gradient() {
    let test_cases = [(2.0, 3.0), (1.0, 1.0), (0.0, 5.0), (-2.0, 4.0)];

    for (x_val, y_val) in test_cases {
        // std::autodiff reference
        let (val_std, dx_std, dy_std) = product_grad_std(x_val, y_val, 1.0);

        // vvad computation
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let y = ad.var(y_val);
        let f = ad.mul(x, y);

        let val_vvad = ad.eval(f);
        let grad_vvad = ad.gradient_reverse(f);

        assert_relative_eq!(val_vvad, val_std, epsilon = 1e-10);
        assert_relative_eq!(grad_vvad[0], dx_std, epsilon = 1e-10);
        assert_relative_eq!(grad_vvad[1], dy_std, epsilon = 1e-10);
    }
}

// =============================================================================
// Forward vs Reverse Mode Agreement Test
// =============================================================================

#[test]
fn test_forward_reverse_agreement() {
    // Test that vvad's forward and reverse mode produce the same gradients
    let mut ad = AutoDiff::new();
    let x = ad.var(1.0);
    let y = ad.var(2.0);

    // f = x^2 * y + sin(x) * exp(y)
    let x2 = ad.square(x);
    let x2y = ad.mul(x2, y);
    let sin_x = ad.sin(x);
    let exp_y = ad.exp(y);
    let sin_exp = ad.mul(sin_x, exp_y);
    let f = ad.add(x2y, sin_exp);

    let grad_forward = ad.gradient(f);
    let grad_reverse = ad.gradient_reverse(f);

    assert_relative_eq!(
        grad_forward[0],
        grad_reverse[0],
        epsilon = 1e-10,
        "Forward/reverse dx mismatch"
    );
    assert_relative_eq!(
        grad_forward[1],
        grad_reverse[1],
        epsilon = 1e-10,
        "Forward/reverse dy mismatch"
    );
}

// =============================================================================
// Composition Tests
// =============================================================================

#[autodiff_forward(composed_deriv_std, Dual, Dual)]
fn composed_fn(x: f64) -> f64 {
    (x.sin()).exp()
}

#[test]
fn test_composed_function_derivative() {
    let test_points = [0.0, 0.5, 1.0, -0.5];

    for x in test_points {
        let (_, deriv_std) = composed_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let sin_x = ad.sin(xv);
        let f = ad.exp(sin_x);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "exp(sin(x)) derivative mismatch at x={}",
            x
        );
    }
}

#[autodiff_forward(chain_deriv_std, Dual, Dual)]
fn chain_fn(x: f64) -> f64 {
    (x * x).sin()
}

#[test]
fn test_chain_rule() {
    let test_points = [0.0, 0.5, 1.0, 2.0];

    for x in test_points {
        let (_, deriv_std) = chain_deriv_std(x, 1.0);

        let mut ad = AutoDiff::new();
        let xv = ad.var(x);
        let x2 = ad.square(xv);
        let f = ad.sin(x2);
        let deriv_vvad = ad.derivative(f, xv, 1);

        assert_relative_eq!(
            deriv_vvad,
            deriv_std,
            epsilon = 1e-10,
            "sin(x^2) derivative mismatch at x={}",
            x
        );
    }
}
