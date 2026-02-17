//! End-to-end integration tests for bevy_autodiff_fit.
//!
//! These tests exercise the full pipeline: fit → compile → eval → differentiate,
//! and cross-validate the dense, sparse, standalone, compiled, and graph-based paths.

use bevy_autodiff::AutoDiff;
use bevy_autodiff_fit::{
    FitOptions, PiecewiseCompiled, fit_dense, fit_sparse, uniform_breakpoints,
};

/// Test suite of analytic functions with known derivatives.
/// Each function is tested through multiple evaluation paths.

#[test]
fn end_to_end_exp_dense() {
    // f(x) = exp(x) on [0, 2], f'(x) = exp(x), f''(x) = exp(x)
    let n = 100;
    let x_data: Vec<f64> = (0..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.exp()).collect();

    let result = fit_dense(&x_data, &y_data, &[0.0, 2.0], &FitOptions { degree: 20 }).unwrap();

    // Standalone eval
    for &x in &[0.0, 0.5, 1.0, 1.5, 2.0] {
        let val = result.fit.eval(x);
        assert!(
            (val - x.exp()).abs() < 1e-3,
            "f({x}): got {val}, expected {}",
            x.exp()
        );
    }

    // Compiled eval with derivatives
    let mut compiled = PiecewiseCompiled::new(&result.fit, 2).unwrap();
    for &x in &[0.3, 0.7, 1.2, 1.8] {
        compiled.eval(x).unwrap();
        let val = compiled.value();
        let d1 = compiled.partial(&[1]).unwrap();
        let d2 = compiled.partial(&[2]).unwrap();
        let exact = x.exp();

        assert!((val - exact).abs() < 1e-3, "f({x}): {val} vs {exact}");
        assert!((d1 - exact).abs() < 1e-1, "f'({x}): {d1} vs {exact}");
        assert!((d2 - exact).abs() < 1.0, "f''({x}): {d2} vs {exact}");
    }

    // Reliability check
    assert!(result.reliability[0].max_reliable_order >= 2);
}

#[test]
fn end_to_end_polynomial_sparse() {
    // f(x) = 3x³ - 2x + 1 on [-1, 1]
    // f'(x) = 9x² - 2, f''(x) = 18x
    let f = |x: f64| 3.0 * x * x * x - 2.0 * x + 1.0;
    let f_prime = |x: f64| 9.0 * x * x - 2.0;
    let f_double_prime = |x: f64| 18.0 * x;

    // 15 scattered points (more than degree 8)
    let x_data: Vec<f64> = (0..15).map(|i| -1.0 + 2.0 * i as f64 / 14.0).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| f(x)).collect();

    let result = fit_sparse(&x_data, &y_data, &[-1.0, 1.0], &FitOptions { degree: 8 }).unwrap();

    let mut compiled = PiecewiseCompiled::new(&result.fit, 2).unwrap();
    for &x in &[-0.5, 0.0, 0.5] {
        compiled.eval(x).unwrap();
        let d1 = compiled.partial(&[1]).unwrap();
        let d2 = compiled.partial(&[2]).unwrap();

        assert!(
            (d1 - f_prime(x)).abs() < 1e-2,
            "f'({x}): {d1} vs {}",
            f_prime(x)
        );
        assert!(
            (d2 - f_double_prime(x)).abs() < 0.5,
            "f''({x}): {d2} vs {}",
            f_double_prime(x)
        );
    }
}

#[test]
fn dense_vs_sparse_cross_validation() {
    // Both paths should give similar results for sin(x) on [0, π]
    let n = 50;
    let x_data: Vec<f64> = (0..=n)
        .map(|i| std::f64::consts::PI * i as f64 / n as f64)
        .collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
    let bp = [0.0, std::f64::consts::PI];
    let opts = FitOptions { degree: 16 };

    let dense_result = fit_dense(&x_data, &y_data, &bp, &opts).unwrap();
    let sparse_result = fit_sparse(&x_data, &y_data, &bp, &opts).unwrap();

    for &x in &[0.5, 1.0, 2.0, 2.5] {
        let dense_val = dense_result.fit.eval(x);
        let sparse_val = sparse_result.fit.eval(x);
        assert!(
            (dense_val - sparse_val).abs() < 1e-3,
            "at {x}: dense={dense_val}, sparse={sparse_val}"
        );
    }
}

#[test]
fn graph_derivative_matches_compiled() {
    // Verify that graph-based differentiation matches compiled derivatives
    let n = 100;
    let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| (2.0 * x).sin()).collect();
    let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 16 }).unwrap();

    let seg = result.fit.segment(0);
    let test_x = 0.4;

    // Graph-based derivative
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(test_x).unwrap();
    let f = seg.build_graph(&mut ad, x);
    let dfdx = ad.differentiate(f, x).unwrap();
    let graph_deriv = ad.eval(dfdx).unwrap();

    // Compiled derivative
    let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();
    compiled.eval(test_x).unwrap();
    let compiled_deriv = compiled.partial(&[1]).unwrap();

    assert!(
        (graph_deriv - compiled_deriv).abs() < 1e-10,
        "graph={graph_deriv}, compiled={compiled_deriv}"
    );
}

#[test]
fn multi_segment_derivatives() {
    // Fit sin(x) on [0, 2π] with 4 segments, test derivatives in each
    let n = 200;
    let two_pi = 2.0 * std::f64::consts::PI;
    let x_data: Vec<f64> = (0..=n).map(|i| two_pi * i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
    let bp = uniform_breakpoints(0.0, two_pi, 4);
    let result = fit_dense(&x_data, &y_data, &bp, &FitOptions { degree: 16 }).unwrap();

    let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();

    // Test points in each segment
    let test_points = [0.5, 1.5, 3.0, 4.5, 5.5];
    for &x in &test_points {
        compiled.eval(x).unwrap();
        let val = compiled.value();
        let d1 = compiled.partial(&[1]).unwrap();

        assert!(
            (val - x.sin()).abs() < 1e-3,
            "sin({x}): got {val}, expected {}",
            x.sin()
        );
        assert!(
            (d1 - x.cos()).abs() < 1e-1,
            "cos({x}): got {d1}, expected {}",
            x.cos()
        );
    }
}

#[test]
fn compose_fit_with_ad_ops() {
    // Build a graph: g(x) = fit(x)² where fit ≈ sin(x)
    // g'(x) = 2*sin(x)*cos(x) = sin(2x)
    let n = 100;
    let x_data: Vec<f64> = (0..=n)
        .map(|i| std::f64::consts::PI * i as f64 / n as f64)
        .collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
    let result = fit_dense(
        &x_data,
        &y_data,
        &[0.0, std::f64::consts::PI],
        &FitOptions { degree: 20 },
    )
    .unwrap();

    let seg = result.fit.segment(0);
    let test_x = 1.0;

    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(test_x).unwrap();
    let f = seg.build_graph(&mut ad, x);
    let g = ad.mul(f, f); // g = sin(x)²

    let val = ad.eval(g).unwrap();
    let expected_val = test_x.sin().powi(2);
    assert!(
        (val - expected_val).abs() < 1e-3,
        "g({test_x}): {val} vs {expected_val}"
    );

    let dgdx = ad.differentiate(g, x).unwrap();
    let deriv = ad.eval(dgdx).unwrap();
    let expected_deriv = (2.0 * test_x).sin(); // sin(2x)
    assert!(
        (deriv - expected_deriv).abs() < 1e-2,
        "g'({test_x}): {deriv} vs {expected_deriv}"
    );
}

#[test]
fn standalone_derivative_matches_compiled() {
    // Cross-validate PiecewiseFit::eval_derivative vs PiecewiseCompiled::partial
    let n = 100;
    let x_data: Vec<f64> = (0..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.exp()).collect();
    let result = fit_dense(&x_data, &y_data, &[0.0, 2.0], &FitOptions { degree: 16 }).unwrap();

    let mut compiled = PiecewiseCompiled::new(&result.fit, 2).unwrap();

    for &x in &[0.3, 0.7, 1.0, 1.5] {
        let standalone_d1 = result.fit.eval_derivative(x, 1);
        let standalone_d2 = result.fit.eval_derivative(x, 2);

        compiled.eval(x).unwrap();
        let compiled_d1 = compiled.partial(&[1]).unwrap();
        let compiled_d2 = compiled.partial(&[2]).unwrap();

        assert!(
            (standalone_d1 - compiled_d1).abs() < 1e-6,
            "d1 at {x}: standalone={standalone_d1}, compiled={compiled_d1}"
        );
        assert!(
            (standalone_d2 - compiled_d2).abs() < 1e-4,
            "d2 at {x}: standalone={standalone_d2}, compiled={compiled_d2}"
        );
    }
}

#[test]
fn reliability_informs_derivative_quality() {
    // Fit with deliberately low degree — reliability should warn about higher derivatives
    let n = 100;
    let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();

    // Very low degree: only 3 coefficients
    let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 3 }).unwrap();

    // Reliability should report limited derivative reliability
    let rel = &result.reliability[0];
    assert!(
        rel.max_reliable_order <= 3,
        "degree-3 fit should have limited reliability, got {}",
        rel.max_reliable_order
    );
}
