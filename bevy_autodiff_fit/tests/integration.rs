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
    ContinuityOptions, FitOptions, FitOptions2D, PiecewiseCompiled, PiecewiseCompiled2D, fit_dense,
    fit_dense_2d, fit_sparse, fit_sparse_2d, fit_sparse_continuous, uniform_breakpoints,
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

        assert!(
            (val - exact).abs() < 5e-4,
            "f({x}): {val} vs {exact}, err = {:.2e}",
            (val - exact).abs()
        );
        assert!(
            (d1 - exact).abs() < 5e-3,
            "f'({x}): {d1} vs {exact}, err = {:.2e}",
            (d1 - exact).abs()
        );
        assert!(
            (d2 - exact).abs() < 5e-2,
            "f''({x}): {d2} vs {exact}, err = {:.2e}",
            (d2 - exact).abs()
        );
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
    let bp = uniform_breakpoints(0.0, two_pi, 4).unwrap();
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

// ============================================================================
// 2D integration tests
// ============================================================================

#[test]
fn end_to_end_2d_polynomial() {
    // f(x,y) = x² + xy + y on [0,1]×[0,1]
    // Sparse fit → compiled eval → all partials
    // ∂f/∂x = 2x + y, ∂f/∂y = x + 1, ∂²f/∂x∂y = 1
    let n = 15;
    let mut xd = Vec::new();
    let mut yd = Vec::new();
    let mut zd = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let x = i as f64 / (n - 1) as f64;
            let y = j as f64 / (n - 1) as f64;
            xd.push(x);
            yd.push(y);
            zd.push(x * x + x * y + y);
        }
    }

    let result = fit_sparse_2d(
        &xd,
        &yd,
        &zd,
        [0.0, 1.0],
        [0.0, 1.0],
        &FitOptions2D {
            degree_x: 4,
            degree_y: 4,
        },
    )
    .unwrap();

    let mut compiled = PiecewiseCompiled2D::new(&result.fit, 2).unwrap();

    let test_x = 0.3;
    let test_y = 0.7;
    compiled.eval(test_x, test_y).unwrap();

    // Tier 2: exact polynomial
    let val = compiled.value();
    let expected = test_x * test_x + test_x * test_y + test_y;
    assert!(
        (val - expected).abs() < 1e-12,
        "f({test_x}, {test_y}) = {val}, expected {expected}"
    );

    let dx = compiled.partial(&[1, 0]).unwrap();
    let expected_dx = 2.0 * test_x + test_y;
    assert!(
        (dx - expected_dx).abs() < 1e-10,
        "∂f/∂x = {dx}, expected {expected_dx}"
    );

    let dy = compiled.partial(&[0, 1]).unwrap();
    let expected_dy = test_x + 1.0;
    assert!(
        (dy - expected_dy).abs() < 1e-10,
        "∂f/∂y = {dy}, expected {expected_dy}"
    );

    let dxy = compiled.partial(&[1, 1]).unwrap();
    assert!((dxy - 1.0).abs() < 1e-9, "∂²f/∂x∂y = {dxy}, expected 1.0");
}

#[test]
fn end_to_end_2d_separable() {
    // f(x,y) = sin(x)*cos(y) via dense fit on [0,π]×[0,π]
    // Single segment, Tier 3 accuracy
    let pi = std::f64::consts::PI;
    let nx = 100;
    let ny = 100;
    let x_data: Vec<f64> = (0..=nx).map(|i| pi * i as f64 / nx as f64).collect();
    let y_data: Vec<f64> = (0..=ny).map(|j| pi * j as f64 / ny as f64).collect();
    let z_data: Vec<Vec<f64>> = y_data
        .iter()
        .map(|&y| x_data.iter().map(|&x| x.sin() * y.cos()).collect())
        .collect();

    let result = fit_dense_2d(
        &x_data,
        &y_data,
        &z_data,
        &[0.0, pi],
        &[0.0, pi],
        &FitOptions2D {
            degree_x: 16,
            degree_y: 16,
        },
    )
    .unwrap();

    let mut compiled = PiecewiseCompiled2D::new(&result.fit, 1).unwrap();

    for &(tx, ty) in &[(0.5, 0.5), (1.0, 1.0), (2.0, 0.5)] {
        compiled.eval(tx, ty).unwrap();
        let val = compiled.value();
        let expected = tx.sin() * ty.cos();
        assert!(
            (val - expected).abs() < 5e-4,
            "f({tx}, {ty}): got {val}, expected {expected}, err = {:.2e}",
            (val - expected).abs()
        );

        let dx = compiled.partial(&[1, 0]).unwrap();
        let expected_dx = tx.cos() * ty.cos();
        assert!(
            (dx - expected_dx).abs() < 5e-3,
            "∂f/∂x at ({tx}, {ty}): got {dx}, expected {expected_dx}, err = {:.2e}",
            (dx - expected_dx).abs()
        );
    }
}

#[test]
fn compiled_2d_matches_standalone() {
    // Cross-validate eval paths: PiecewiseFit2D::eval vs PiecewiseCompiled2D::value
    let n = 10;
    let mut xd = Vec::new();
    let mut yd = Vec::new();
    let mut zd = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let x = i as f64 / (n - 1) as f64;
            let y = j as f64 / (n - 1) as f64;
            xd.push(x);
            yd.push(y);
            zd.push(x * x - y * y + 3.0 * x * y);
        }
    }

    let result = fit_sparse_2d(
        &xd,
        &yd,
        &zd,
        [0.0, 1.0],
        [0.0, 1.0],
        &FitOptions2D {
            degree_x: 4,
            degree_y: 4,
        },
    )
    .unwrap();

    let mut compiled = PiecewiseCompiled2D::new(&result.fit, 0).unwrap();

    for &(tx, ty) in &[(0.1, 0.2), (0.5, 0.5), (0.8, 0.3), (0.0, 1.0)] {
        let standalone = result.fit.eval(tx, ty);
        compiled.eval(tx, ty).unwrap();
        let compiled_val = compiled.value();
        assert!(
            (standalone - compiled_val).abs() < 1e-12,
            "at ({tx}, {ty}): standalone={standalone}, compiled={compiled_val}"
        );
    }
}

#[test]
fn graph_2d_mixed_partial() {
    // f(x,y) = x²y via graph, check ∂²f/∂x∂y = 2x
    let n = 10;
    let mut xd = Vec::new();
    let mut yd = Vec::new();
    let mut zd = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let x = i as f64 / (n - 1) as f64;
            let y = j as f64 / (n - 1) as f64;
            xd.push(x);
            yd.push(y);
            zd.push(x * x * y);
        }
    }

    let result = fit_sparse_2d(
        &xd,
        &yd,
        &zd,
        [0.0, 1.0],
        [0.0, 1.0],
        &FitOptions2D {
            degree_x: 3,
            degree_y: 2,
        },
    )
    .unwrap();

    let seg = result.fit.segment(0, 0);
    let test_x = 0.6;
    let test_y = 0.4;
    let mut ad = AutoDiff::<f64>::new();
    let x = ad.var(test_x).unwrap();
    let y = ad.var(test_y).unwrap();
    let f = seg.build_graph(&mut ad, x, y);

    // ∂f/∂x = 2xy
    let dfdx = ad.differentiate(f, x).unwrap();
    let dx_val = ad.eval(dfdx).unwrap();
    assert!(
        (dx_val - 2.0 * test_x * test_y).abs() < 1e-10,
        "∂f/∂x = {dx_val}, expected {}",
        2.0 * test_x * test_y
    );

    // ∂²f/∂x∂y = 2x
    let d2fdxdy = ad.differentiate(dfdx, y).unwrap();
    let dxy_val = ad.eval(d2fdxdy).unwrap();
    assert!(
        (dxy_val - 2.0 * test_x).abs() < 1e-9,
        "∂²f/∂x∂y = {dxy_val}, expected {}",
        2.0 * test_x
    );
}

// ============================================================================
// Continuity integration tests
// ============================================================================

#[test]
fn continuous_1d_boundary() {
    // Fit exp(x) with 3 segments, C¹ continuity
    let n = 150;
    let x_data: Vec<f64> = (0..=n).map(|i| 3.0 * i as f64 / n as f64).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| x.exp()).collect();
    let bp = uniform_breakpoints(0.0, 3.0, 3).unwrap();

    let result = fit_sparse_continuous(
        &x_data,
        &y_data,
        &bp,
        &FitOptions { degree: 8 },
        &ContinuityOptions {
            order: 1,
            weight: 1e4,
        },
    )
    .unwrap();

    // Check C⁰ at internal breakpoints (x=1, x=2)
    for &bx in &[1.0, 2.0] {
        let left_idx = if bx == 1.0 { 0 } else { 1 };
        let left_val = result.fit.segment(left_idx).eval(bx);
        let right_val = result.fit.segment(left_idx + 1).eval(bx);
        let gap = (left_val - right_val).abs();
        assert!(
            gap < 1e-6,
            "C⁰ gap at x={bx}: {gap:.2e} (left={left_val}, right={right_val})"
        );
    }

    // Check overall fit quality (Tier 3)
    let mut compiled = PiecewiseCompiled::new(&result.fit, 0).unwrap();
    for &x in &[0.5, 1.5, 2.5] {
        compiled.eval(x).unwrap();
        let val = compiled.value();
        assert!(
            (val - x.exp()).abs() < 5e-3,
            "f({x}) = {val}, expected {}, err = {:.2e}",
            x.exp(),
            (val - x.exp()).abs()
        );
    }
}
