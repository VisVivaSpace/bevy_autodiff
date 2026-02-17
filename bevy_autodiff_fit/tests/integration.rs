//! End-to-end integration tests for bevy_autodiff_fit.
//!
//! These tests exercise the full pipeline: fit → compile → eval → differentiate,
//! and cross-validate the dense, sparse, standalone, compiled, and graph-based paths.
//!
//! # Tolerance justification
//!
//! Tolerances follow the aerospace-numerical-methods tier system:
//!
//! - **Tier 2 (exact arithmetic)**: Sparse QR on exact polynomials, graph vs compiled
//!   comparisons — near machine epsilon (1e-12 to 1e-14).
//!
//! - **Tier 3 (algorithm-dependent)**: Dense fits using linear interpolation resampling.
//!   Error dominated by O(h²) interpolation where h = data spacing.
//!   For n=100 on [0,π]: h ≈ 0.031, max |f''| ≈ 1, so resampling error ≈ h²/8 ≈ 1.2e-4.
//!   First derivatives amplified by degree/domain_width factor.
//!   Test points avoid domain endpoints where boundary effects amplify error.

use bevy_autodiff::AutoDiff;
use bevy_autodiff_fit::{
    FitOptions, PiecewiseCompiled, fit_dense, fit_sparse, uniform_breakpoints,
};

#[test]
fn end_to_end_exp_dense() {
    // f(x) = exp(x) on [0, 2], f'(x) = exp(x), f''(x) = exp(x)
    let n = 100;
    let x_data: Vec<f64> = (0..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.exp()).collect();

    let result = fit_dense(&x_data, &y_data, &[0.0, 2.0], &FitOptions { degree: 20 }).unwrap();

    // Standalone eval — test interior points only (endpoints have boundary effects)
    // Observed max error: 1.9e-4 at interior points; tolerance 5e-4 gives 2.5x margin
    for &x in &[0.2, 0.5, 1.0, 1.5, 1.8] {
        let val = result.fit.eval(x);
        assert!(
            (val - x.exp()).abs() < 5e-4,
            "f({x}): got {val}, expected {}, err = {:.2e}",
            x.exp(),
            (val - x.exp()).abs()
        );
    }

    // Compiled eval with derivatives — interior points
    // Observed: f err < 2e-4, f' err < 2e-3, f'' err < 2e-2 (interior)
    let mut compiled = PiecewiseCompiled::new(&result.fit, 2).unwrap();
    for &x in &[0.3, 0.7, 1.2, 1.8] {
        compiled.eval(x).unwrap();
        let val = compiled.value();
        let d1 = compiled.partial(&[1]).unwrap();
        let d2 = compiled.partial(&[2]).unwrap();
        let exact = x.exp();

        assert!((val - exact).abs() < 5e-4, "f({x}): {val} vs {exact}, err = {:.2e}", (val - exact).abs());
        assert!((d1 - exact).abs() < 5e-3, "f'({x}): {d1} vs {exact}, err = {:.2e}", (d1 - exact).abs());
        assert!((d2 - exact).abs() < 5e-2, "f''({x}): {d2} vs {exact}, err = {:.2e}", (d2 - exact).abs());
    }

    assert!(result.reliability[0].max_reliable_order >= 2);
}

#[test]
fn end_to_end_polynomial_sparse() {
    // f(x) = 3x³ - 2x + 1 on [-1, 1]
    // f'(x) = 9x² - 2, f''(x) = 18x
    //
    // Sparse QR on a polynomial of degree 3 with degree-8 fit:
    // the QR solver recovers exact coefficients (Tier 2, exact arithmetic).
    // Observed errors: ~1e-15 (machine epsilon).
    let f = |x: f64| 3.0 * x * x * x - 2.0 * x + 1.0;
    let f_prime = |x: f64| 9.0 * x * x - 2.0;
    let f_double_prime = |x: f64| 18.0 * x;

    // 15 scattered points (more than degree 8)
    let x_data: Vec<f64> = (0..15).map(|i| -1.0 + 2.0 * i as f64 / 14.0).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| f(x)).collect();

    let result = fit_sparse(&x_data, &y_data, &[-1.0, 1.0], &FitOptions { degree: 8 }).unwrap();

    // Tier 2: near machine epsilon for exact polynomial recovery
    let mut compiled = PiecewiseCompiled::new(&result.fit, 2).unwrap();
    for &x in &[-0.8, -0.5, 0.0, 0.5, 0.8] {
        compiled.eval(x).unwrap();
        let val = compiled.value();
        let d1 = compiled.partial(&[1]).unwrap();
        let d2 = compiled.partial(&[2]).unwrap();

        assert!(
            (val - f(x)).abs() < 1e-12,
            "f({x}): {val} vs {}, err = {:.2e}",
            f(x),
            (val - f(x)).abs()
        );
        assert!(
            (d1 - f_prime(x)).abs() < 1e-10,
            "f'({x}): {d1} vs {}, err = {:.2e}",
            f_prime(x),
            (d1 - f_prime(x)).abs()
        );
        assert!(
            (d2 - f_double_prime(x)).abs() < 1e-8,
            "f''({x}): {d2} vs {}, err = {:.2e}",
            f_double_prime(x),
            (d2 - f_double_prime(x)).abs()
        );
    }
}

#[test]
fn dense_vs_sparse_cross_validation() {
    // Both paths should give similar results for sin(x) on [0, π]
    // Different algorithms (DCT vs QR) with same data — expect ~1e-4 agreement
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
            (dense_val - sparse_val).abs() < 5e-4,
            "at {x}: dense={dense_val}, sparse={sparse_val}, diff={:.2e}",
            (dense_val - sparse_val).abs()
        );
    }
}

#[test]
fn graph_derivative_matches_compiled() {
    // Graph-based differentiation vs compiled: same Chebyshev coefficients,
    // same Clenshaw evaluation — should be bit-identical (Tier 1/2).
    // Observed difference: 0.0
    let n = 100;
    let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| (2.0 * x).sin()).collect();
    let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 16 }).unwrap();

    let seg = result.fit.segment(0);
    let test_x = 0.4;

    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(test_x).unwrap();
    let f = seg.build_graph(&mut ad, x);
    let dfdx = ad.differentiate(f, x).unwrap();
    let graph_deriv = ad.eval(dfdx).unwrap();

    let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();
    compiled.eval(test_x).unwrap();
    let compiled_deriv = compiled.partial(&[1]).unwrap();

    // Tier 2: same computation, different evaluation path
    assert!(
        (graph_deriv - compiled_deriv).abs() < 1e-12,
        "graph={graph_deriv}, compiled={compiled_deriv}, diff={:.2e}",
        (graph_deriv - compiled_deriv).abs()
    );
}

#[test]
fn multi_segment_derivatives() {
    // Fit sin(x) on [0, 2π] with 4 segments, test derivatives in each
    // Observed: f err < 1.1e-4, f' err < 1.2e-3
    let n = 200;
    let two_pi = 2.0 * std::f64::consts::PI;
    let x_data: Vec<f64> = (0..=n).map(|i| two_pi * i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
    let bp = uniform_breakpoints(0.0, two_pi, 4);
    let result = fit_dense(&x_data, &y_data, &bp, &FitOptions { degree: 16 }).unwrap();

    let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();

    // Test points well inside each segment (avoid segment boundaries)
    let test_points = [0.5, 1.5, 3.0, 4.5, 5.5];
    for &x in &test_points {
        compiled.eval(x).unwrap();
        let val = compiled.value();
        let d1 = compiled.partial(&[1]).unwrap();

        assert!(
            (val - x.sin()).abs() < 5e-4,
            "sin({x}): got {val}, expected {}, err = {:.2e}",
            x.sin(),
            (val - x.sin()).abs()
        );
        assert!(
            (d1 - x.cos()).abs() < 5e-3,
            "cos({x}): got {d1}, expected {}, err = {:.2e}",
            x.cos(),
            (d1 - x.cos()).abs()
        );
    }
}

#[test]
fn compose_fit_with_ad_ops() {
    // Build a graph: g(x) = fit(x)² where fit ≈ sin(x)
    // g'(x) = 2*sin(x)*cos(x) = sin(2x)
    // Error: fit error + composition amplification
    // Observed: f err ~1e-4, f' err ~1e-3
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
    // Composition squares the fit error: 2*|sin(x)|*|fit_err| ≈ 2*0.84*5e-5 ≈ 1e-4
    assert!(
        (val - expected_val).abs() < 5e-4,
        "g({test_x}): {val} vs {expected_val}, err = {:.2e}",
        (val - expected_val).abs()
    );

    let dgdx = ad.differentiate(g, x).unwrap();
    let deriv = ad.eval(dgdx).unwrap();
    let expected_deriv = (2.0 * test_x).sin(); // sin(2x)
    // Derivative of composition: 2*fit*fit' errors compound
    assert!(
        (deriv - expected_deriv).abs() < 5e-3,
        "g'({test_x}): {deriv} vs {expected_deriv}, err = {:.2e}",
        (deriv - expected_deriv).abs()
    );
}

#[test]
fn standalone_derivative_matches_compiled() {
    // Cross-validate PiecewiseFit::eval_derivative vs PiecewiseCompiled::partial
    // Both use the same Chebyshev coefficients, different eval paths.
    // eval_derivative: Chebyshev derivative recurrence + Clenshaw
    // compiled::partial: symbolic differentiation of Clenshaw graph
    // These should agree to near machine epsilon for first derivatives,
    // with small accumulated roundoff for second derivatives.
    // Observed: d1 diff ~1e-14, d2 diff ~1e-11
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

        // Tier 2: same coefficients, different evaluation algorithms
        assert!(
            (standalone_d1 - compiled_d1).abs() < 1e-8,
            "d1 at {x}: standalone={standalone_d1}, compiled={compiled_d1}, diff={:.2e}",
            (standalone_d1 - compiled_d1).abs()
        );
        assert!(
            (standalone_d2 - compiled_d2).abs() < 1e-6,
            "d2 at {x}: standalone={standalone_d2}, compiled={compiled_d2}, diff={:.2e}",
            (standalone_d2 - compiled_d2).abs()
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

    let rel = &result.reliability[0];
    assert!(
        rel.max_reliable_order <= 3,
        "degree-3 fit should have limited reliability, got {}",
        rel.max_reliable_order
    );
}
