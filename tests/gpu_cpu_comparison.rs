//! GPU vs CPU oracle comparison tests.
//!
//! Compiles graphs on the CPU, evaluates on both CPU (f64) and GPU (f32),
//! and verifies results match within f32 tolerance.
//!
//! Run with: cargo test --features wgpu --test gpu_cpu_comparison

#![cfg(feature = "wgpu")]

use bevy_autodiff::gpu::GpuContext;
use bevy_autodiff::AutoDiff;

fn gpu() -> Option<GpuContext> {
    GpuContext::new().ok()
}

const TOL: f32 = 1e-4;

/// Helper: build a 1-input graph, evaluate on GPU and CPU, compare.
fn compare_unary(
    ctx: &GpuContext,
    build: impl Fn(&mut AutoDiff, bevy_autodiff::Var) -> bevy_autodiff::Var,
    x_vals: &[f32],
) {
    let mut ad = AutoDiff::new();
    let x = ad.var(0.0).unwrap();
    let f = build(&mut ad, x);
    let mut graph = ad.compile_primal(f, &[x]).unwrap();
    let gpu_graph = ctx.prepare(&graph).unwrap();

    let results = gpu_graph.eval_batch(ctx, &[x_vals]).unwrap();
    let gpu_vals = results.values();

    for (i, &xv) in x_vals.iter().enumerate() {
        graph.eval(&[xv as f64]);
        let cpu_val = graph.value() as f32;
        assert!(
            (gpu_vals[i] - cpu_val).abs() < TOL,
            "mismatch at x={xv}: gpu={}, cpu={cpu_val}",
            gpu_vals[i]
        );
    }
}

// =========================================================================
// Unary operations
// =========================================================================

#[test]
fn gpu_cpu_neg() {
    let Some(ctx) = gpu() else { return };
    compare_unary(&ctx, |ad, x| ad.neg(x), &[-2.0, -1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn gpu_cpu_sin() {
    let Some(ctx) = gpu() else { return };
    compare_unary(&ctx, |ad, x| ad.sin(x), &[0.0, 0.5, 1.0, 2.0, 3.0]);
}

#[test]
fn gpu_cpu_cos() {
    let Some(ctx) = gpu() else { return };
    compare_unary(&ctx, |ad, x| ad.cos(x), &[0.0, 0.5, 1.0, 2.0, 3.0]);
}

#[test]
fn gpu_cpu_tan() {
    let Some(ctx) = gpu() else { return };
    compare_unary(&ctx, |ad, x| ad.tan(x), &[0.0, 0.3, 0.7, 1.0]);
}

#[test]
fn gpu_cpu_exp() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.exp(x),
        &[-1.0, 0.0, 0.5, 1.0, 2.0],
    );
}

#[test]
fn gpu_cpu_ln() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.ln(x),
        &[0.1, 0.5, 1.0, 2.0, 10.0],
    );
}

#[test]
fn gpu_cpu_sqrt() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.sqrt(x),
        &[0.01, 0.25, 1.0, 4.0, 9.0],
    );
}

#[test]
fn gpu_cpu_sinh() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.sinh(x),
        &[-1.0, 0.0, 0.5, 1.0, 2.0],
    );
}

#[test]
fn gpu_cpu_cosh() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.cosh(x),
        &[-1.0, 0.0, 0.5, 1.0, 2.0],
    );
}

#[test]
fn gpu_cpu_tanh() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.tanh(x),
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
    );
}

#[test]
fn gpu_cpu_asin() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.asin(x),
        &[-0.9, -0.5, 0.0, 0.5, 0.9],
    );
}

#[test]
fn gpu_cpu_acos() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.acos(x),
        &[-0.9, -0.5, 0.0, 0.5, 0.9],
    );
}

#[test]
fn gpu_cpu_atan() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.atan(x),
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
    );
}

#[test]
fn gpu_cpu_asinh() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.asinh(x),
        &[-2.0, -1.0, 0.0, 1.0, 2.0],
    );
}

#[test]
fn gpu_cpu_acosh() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.acosh(x),
        &[1.0, 1.5, 2.0, 3.0, 5.0],
    );
}

#[test]
fn gpu_cpu_atanh() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| ad.atanh(x),
        &[-0.9, -0.5, 0.0, 0.5, 0.9],
    );
}

// =========================================================================
// Binary operations
// =========================================================================

/// Helper for 2-input binary ops.
fn compare_binary(
    ctx: &GpuContext,
    build: impl Fn(&mut AutoDiff, bevy_autodiff::Var, bevy_autodiff::Var) -> bevy_autodiff::Var,
    x_vals: &[f32],
    y_vals: &[f32],
) {
    let mut ad = AutoDiff::new();
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let f = build(&mut ad, x, y);
    let mut graph = ad.compile_primal(f, &[x, y]).unwrap();
    let gpu_graph = ctx.prepare(&graph).unwrap();

    let results = gpu_graph.eval_batch(ctx, &[x_vals, y_vals]).unwrap();
    let gpu_vals = results.values();

    for (i, (&xv, &yv)) in x_vals.iter().zip(y_vals).enumerate() {
        graph.eval(&[xv as f64, yv as f64]);
        let cpu_val = graph.value() as f32;
        assert!(
            (gpu_vals[i] - cpu_val).abs() < TOL,
            "mismatch at x={xv},y={yv}: gpu={}, cpu={cpu_val}",
            gpu_vals[i]
        );
    }
}

#[test]
fn gpu_cpu_add() {
    let Some(ctx) = gpu() else { return };
    compare_binary(
        &ctx,
        |ad, x, y| ad.add(x, y),
        &[1.0, 2.0, -3.0, 0.0],
        &[4.0, -1.0, 3.0, 0.0],
    );
}

#[test]
fn gpu_cpu_sub() {
    let Some(ctx) = gpu() else { return };
    compare_binary(
        &ctx,
        |ad, x, y| ad.sub(x, y),
        &[1.0, 5.0, -3.0, 0.0],
        &[4.0, 2.0, 3.0, 7.0],
    );
}

#[test]
fn gpu_cpu_mul() {
    let Some(ctx) = gpu() else { return };
    compare_binary(
        &ctx,
        |ad, x, y| ad.mul(x, y),
        &[1.0, 2.0, -3.0, 0.0],
        &[4.0, -1.0, 3.0, 7.0],
    );
}

#[test]
fn gpu_cpu_div() {
    let Some(ctx) = gpu() else { return };
    compare_binary(
        &ctx,
        |ad, x, y| ad.div(x, y),
        &[1.0, 6.0, -3.0, 0.0],
        &[2.0, 3.0, 1.5, 7.0],
    );
}

#[test]
fn gpu_cpu_pow() {
    let Some(ctx) = gpu() else { return };
    compare_binary(
        &ctx,
        |ad, x, y| ad.pow(x, y),
        &[2.0, 3.0, 4.0, 1.0],
        &[3.0, 2.0, 0.5, 10.0],
    );
}

// =========================================================================
// Compositions
// =========================================================================

#[test]
fn gpu_cpu_sin_exp() {
    let Some(ctx) = gpu() else { return };
    compare_unary(
        &ctx,
        |ad, x| {
            let e = ad.exp(x);
            ad.sin(e)
        },
        &[0.0, 0.1, 0.5, 1.0],
    );
}

#[test]
fn gpu_cpu_complex_expression() {
    // f(x, y) = sin(x*y) + exp(x)
    let Some(ctx) = gpu() else { return };
    let mut ad = AutoDiff::new();
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let xy = ad.mul(x, y);
    let sin_xy = ad.sin(xy);
    let exp_x = ad.exp(x);
    let f = ad.add(sin_xy, exp_x);
    let mut graph = ad.compile_primal(f, &[x, y]).unwrap();
    let gpu_graph = ctx.prepare(&graph).unwrap();

    let x_vals = vec![0.1, 0.5, 1.0, 2.0];
    let y_vals = vec![0.2, 0.6, 1.5, 0.5];
    let results = gpu_graph.eval_batch(&ctx, &[&x_vals, &y_vals]).unwrap();

    for (i, (&xv, &yv)) in x_vals.iter().zip(&y_vals).enumerate() {
        graph.eval(&[xv as f64, yv as f64]);
        let cpu = graph.value() as f32;
        assert!(
            (results.values()[i] - cpu).abs() < TOL,
            "complex expr at x={xv},y={yv}: gpu={}, cpu={cpu}",
            results.values()[i]
        );
    }
}

// =========================================================================
// Partials via forward-mode symbolic derivatives
// =========================================================================

#[test]
fn gpu_cpu_partials_two_inputs() {
    // f(x, y) = x*y, df/dx = y, df/dy = x
    let Some(ctx) = gpu() else { return };
    let mut ad = AutoDiff::new();
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let f = ad.mul(x, y);
    let graph = ad.compile_order(f, &[x, y], 1).unwrap();
    let gpu_graph = ctx.prepare(&graph).unwrap();

    let x_vals = vec![1.0, 2.0, 3.0];
    let y_vals = vec![4.0, 5.0, 6.0];
    let results = gpu_graph.eval_batch(&ctx, &[&x_vals, &y_vals]).unwrap();

    // df/dx = y
    let dfdx = results.partials(&[1, 0]).expect("df/dx should exist");
    for (i, &yv) in y_vals.iter().enumerate() {
        assert!(
            (dfdx[i] - yv).abs() < TOL,
            "df/dx at y={yv}: got {}",
            dfdx[i]
        );
    }

    // df/dy = x
    let dfdy = results.partials(&[0, 1]).expect("df/dy should exist");
    for (i, &xv) in x_vals.iter().enumerate() {
        assert!(
            (dfdy[i] - xv).abs() < TOL,
            "df/dy at x={xv}: got {}",
            dfdy[i]
        );
    }
}

// =========================================================================
// Batch sizes
// =========================================================================

#[test]
fn gpu_batch_size_1() {
    let Some(ctx) = gpu() else { return };
    compare_unary(&ctx, |ad, x| ad.sin(x), &[1.0]);
}

#[test]
fn gpu_batch_size_1000() {
    let Some(ctx) = gpu() else { return };
    let x_vals: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
    compare_unary(&ctx, |ad, x| ad.sin(x), &x_vals);
}

#[test]
fn gpu_batch_size_100k() {
    let Some(ctx) = gpu() else { return };
    let x_vals: Vec<f32> = (0..100_000).map(|i| i as f32 * 0.0001).collect();
    compare_unary(&ctx, |ad, x| ad.sin(x), &x_vals);
}
