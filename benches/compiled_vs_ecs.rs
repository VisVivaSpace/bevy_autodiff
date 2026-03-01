//! Benchmarks comparing CompiledGraph vs ECS-based evaluation.
//!
//! Run with: cargo bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use bevy_autodiff::{AutoDiff, Var};

/// Build Rosenbrock function: (1-x)² + 100*(y-x²)²
fn build_rosenbrock(
    ad: &mut AutoDiff<f64>,
) -> (bevy_autodiff::Var, bevy_autodiff::Var, bevy_autodiff::Var) {
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let one = ad.constant(1.0);
    let hundred = ad.constant(100.0);

    let one_minus_x = ad.sub(one, x);
    let term1 = ad.square(one_minus_x);
    let x_sq = ad.square(x);
    let y_minus_x_sq = ad.sub(y, x_sq);
    let term2_inner = ad.square(y_minus_x_sq);
    let term2 = ad.mul(hundred, term2_inner);
    let f = ad.add(term1, term2);

    (x, y, f)
}

fn bench_ecs_eval(c: &mut Criterion) {
    c.bench_function("ecs_rosenbrock_eval", |b| {
        b.iter(|| {
            let mut ad = AutoDiff::new();
            let (x, y, f) = build_rosenbrock(&mut ad);
            // We have to rebuild the graph each time for ECS since there's
            // no re-evaluation with new inputs without rebuilding
            let _ = ad.eval(f).unwrap();
            let _ = (x, y); // suppress unused warnings
        });
    });
}

fn bench_ecs_gradient(c: &mut Criterion) {
    c.bench_function("ecs_rosenbrock_gradient", |b| {
        b.iter(|| {
            let mut ad = AutoDiff::new();
            let (_x, _y, f) = build_rosenbrock(&mut ad);
            let _ = ad.gradient(f).unwrap();
        });
    });
}

fn bench_compiled_eval(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (x, y, f) = build_rosenbrock(&mut ad);
    let mut cg = ad.compile_order(f, &[x, y], 1).unwrap();

    c.bench_function("compiled_rosenbrock_eval", |b| {
        b.iter(|| {
            cg.eval(&[0.5, 0.8]).unwrap();
            let _ = cg.value();
        });
    });
}

fn bench_compiled_gradient(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (x, y, f) = build_rosenbrock(&mut ad);
    let mut cg = ad.compile_order(f, &[x, y], 1).unwrap();

    c.bench_function("compiled_rosenbrock_gradient", |b| {
        b.iter(|| {
            cg.eval(&[0.5, 0.8]).unwrap();
            let _ = cg.partial(&[1, 0]).unwrap();
            let _ = cg.partial(&[0, 1]).unwrap();
        });
    });
}

fn bench_compiled_hessian(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (x, y, f) = build_rosenbrock(&mut ad);
    let mut cg = ad.compile_order(f, &[x, y], 2).unwrap();

    c.bench_function("compiled_rosenbrock_hessian", |b| {
        b.iter(|| {
            cg.eval(&[0.5, 0.8]).unwrap();
            let _ = cg.value();
            let _ = cg.partial(&[1, 0]).unwrap();
            let _ = cg.partial(&[0, 1]).unwrap();
            let _ = cg.partial(&[2, 0]).unwrap();
            let _ = cg.partial(&[1, 1]).unwrap();
            let _ = cg.partial(&[0, 2]).unwrap();
        });
    });
}

fn bench_compiled_multipoint(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (x, y, f) = build_rosenbrock(&mut ad);
    let mut cg = ad.compile_order(f, &[x, y], 1).unwrap();

    let points = vec![
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [0.5, 0.25],
        [-1.0, 1.0],
        [2.0, 4.0],
        [0.1, 0.01],
        [-0.5, 0.25],
        [1.5, 2.25],
        [3.0, 9.0],
    ];

    c.bench_function("compiled_rosenbrock_10_points", |b| {
        b.iter(|| {
            for pt in &points {
                cg.eval(pt).unwrap();
                let _ = cg.value();
                let _ = cg.partial(&[1, 0]).unwrap();
                let _ = cg.partial(&[0, 1]).unwrap();
            }
        });
    });
}

fn bench_reverse_gradient(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (x, y, f) = build_rosenbrock(&mut ad);
    let mut cg = ad.compile_primal(f, &[x, y]).unwrap();

    c.bench_function("reverse_rosenbrock_gradient", |b| {
        b.iter(|| {
            cg.eval(&[0.5, 0.8]).unwrap();
            let _ = cg.gradient();
        });
    });
}

/// Build two-body gravitational acceleration: a = -mu/r³ * [x, y, z]
/// This is a deeper graph with sqrt, pow, div — representative of real
/// scientific computing workloads. 6 inputs (x,y,z,vx,vy,vz), 3 outputs.
fn build_two_body(ad: &mut AutoDiff<f64>) -> ([Var; 6], [Var; 3]) {
    let x = ad.var(6578.0).unwrap(); // km (LEO radius)
    let y = ad.var(0.0).unwrap();
    let z = ad.var(0.0).unwrap();
    let vx = ad.var(0.0).unwrap();
    let vy = ad.var(7.784).unwrap(); // km/s (circular velocity)
    let vz = ad.var(0.0).unwrap();
    let mu = ad.constant(398600.4418); // km³/s² (Earth)

    // r² = x² + y² + z²
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let z2 = ad.square(z);
    let r2_xy = ad.add(x2, y2);
    let r2 = ad.add(r2_xy, z2);

    // r³ = r² * sqrt(r²)
    let r = ad.sqrt(r2);
    let r3 = ad.mul(r2, r);

    // -mu/r³
    let neg_mu = ad.neg(mu);
    let coeff = ad.div(neg_mu, r3);

    // acceleration components
    let ax = ad.mul(coeff, x);
    let ay = ad.mul(coeff, y);
    let az = ad.mul(coeff, z);

    ([x, y, z, vx, vy, vz], [ax, ay, az])
}

fn bench_twobody_reverse_gradient(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (inputs, outputs) = build_two_body(&mut ad);

    // Compile each acceleration component with reverse-mode
    let mut cg_ax = ad.compile_primal(outputs[0], &inputs).unwrap();
    let mut cg_ay = ad.compile_primal(outputs[1], &inputs).unwrap();
    let mut cg_az = ad.compile_primal(outputs[2], &inputs).unwrap();

    let state = [6578.0, 0.0, 0.0, 0.0, 7.784, 0.0];

    c.bench_function("reverse_twobody_jacobian_row", |b| {
        b.iter(|| {
            // One Jacobian row (gradient of one acceleration component)
            cg_ax.eval(&state).unwrap();
            let _ = cg_ax.gradient();
        });
    });

    c.bench_function("reverse_twobody_jacobian_full", |b| {
        b.iter(|| {
            // Full 3x6 Jacobian (all three acceleration gradients)
            cg_ax.eval(&state).unwrap();
            let _ = cg_ax.gradient();
            cg_ay.eval(&state).unwrap();
            let _ = cg_ay.gradient();
            cg_az.eval(&state).unwrap();
            let _ = cg_az.gradient();
        });
    });
}

fn bench_twobody_forward_hessian(c: &mut Criterion) {
    let mut ad = AutoDiff::new();
    let (inputs, outputs) = build_two_body(&mut ad);
    // Only position inputs for the Hessian (3 inputs, not 6)
    let pos_inputs = [inputs[0], inputs[1], inputs[2]];
    let mut cg = ad.compile_order(outputs[0], &pos_inputs, 2).unwrap();

    let state = [6578.0, 0.0, 0.0];

    c.bench_function("forward_twobody_hessian", |b| {
        b.iter(|| {
            cg.eval(&state).unwrap();
            let _ = cg.value();
            // All first and second partials
            let _ = cg.partial(&[1, 0, 0]);
            let _ = cg.partial(&[0, 1, 0]);
            let _ = cg.partial(&[0, 0, 1]);
            let _ = cg.partial(&[2, 0, 0]);
            let _ = cg.partial(&[1, 1, 0]);
            let _ = cg.partial(&[1, 0, 1]);
            let _ = cg.partial(&[0, 2, 0]);
            let _ = cg.partial(&[0, 1, 1]);
            let _ = cg.partial(&[0, 0, 2]);
        });
    });
}

// ============================================================================
// Scaling: compile_order cost vs derivative order
// ============================================================================

/// Measures how compilation cost grows with derivative order on Rosenbrock.
/// Healthy growth is roughly exponential in order (graph size doubles each
/// level). Super-exponential growth would hint at a constant-folding bug.
fn bench_compile_order_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_order_scaling");
    for order in [1usize, 2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("rosenbrock_order", order),
            &order,
            |b, &order| {
                b.iter(|| {
                    let mut ad = AutoDiff::new();
                    let (x, y, f) = build_rosenbrock(&mut ad);
                    let _ = ad.compile_order(f, &[x, y], order).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Scaling: compile time vs expression depth
// ============================================================================

/// Measures how compile time scales with depth of a sin-chain f = sin(sin(…sin(x)…)).
/// Linear growth = O(n) topological sort. Quadratic = bug in topo sort.
fn bench_graph_depth_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_depth_scaling");
    for depth in [5usize, 10, 20, 40] {
        group.bench_with_input(
            BenchmarkId::new("sin_chain_compile", depth),
            &depth,
            |b, &depth| {
                b.iter(|| {
                    let mut ad = AutoDiff::new();
                    let x = ad.var(1.0).unwrap();
                    let mut v = x;
                    for _ in 0..depth {
                        v = ad.sin(v);
                    }
                    let _ = ad.compile_order(v, &[x], 1).unwrap();
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Scaling: reverse vs forward gradient as input count grows
// ============================================================================

/// Compares reverse-mode and forward-mode gradient cost as the number of
/// inputs grows (f = Σ sin(xᵢ)).
///
/// Reverse-mode: O(nodes) regardless of input count — gradient() one call.
/// Forward-mode: O(n·nodes) — compile_order computes n symbolic partials.
///
/// Only the eval + extraction phase is timed (compilation amortized out).
fn bench_gradient_vs_inputs(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_input_scaling");
    for n in [1usize, 2, 4, 8, 16] {
        // --- reverse mode ---
        group.bench_with_input(
            BenchmarkId::new("reverse", n),
            &n,
            |b, &n| {
                let mut ad = AutoDiff::new();
                let inputs: Vec<Var> = (0..n).map(|_| ad.var(1.0).unwrap()).collect();
                let sins: Vec<Var> = inputs.iter().map(|&xi| ad.sin(xi)).collect();
                let f = sins[1..].iter().fold(sins[0], |acc, &s| ad.add(acc, s));
                let mut cg = ad.compile_primal(f, &inputs).unwrap();
                let vals: Vec<f64> = vec![1.0; n];
                b.iter(|| {
                    cg.eval(&vals).unwrap();
                    let _ = cg.gradient();
                });
            },
        );
        // --- forward mode ---
        group.bench_with_input(
            BenchmarkId::new("forward", n),
            &n,
            |b, &n| {
                let mut ad = AutoDiff::new();
                let inputs: Vec<Var> = (0..n).map(|_| ad.var(1.0).unwrap()).collect();
                let sins: Vec<Var> = inputs.iter().map(|&xi| ad.sin(xi)).collect();
                let f = sins[1..].iter().fold(sins[0], |acc, &s| ad.add(acc, s));
                let mut cg = ad.compile_order(f, &inputs, 1).unwrap();
                let vals: Vec<f64> = vec![1.0; n];
                b.iter(|| {
                    cg.eval(&vals).unwrap();
                    for i in 0..n {
                        let mut idx = vec![0usize; n];
                        idx[i] = 1;
                        let _ = cg.partial(&idx).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Constant folding: d^k[constant]/dx^k should be O(1) in k
// ============================================================================

/// Differentiates a constant node to arbitrary order.
/// Constant folding should collapse each derivative to zero immediately,
/// so cost should be O(1) regardless of order.
fn bench_constant_folding(c: &mut Criterion) {
    let mut group = c.benchmark_group("constant_folding");
    for order in [1usize, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("d_const_dx", order),
            &order,
            |b, &order| {
                b.iter(|| {
                    let mut ad = AutoDiff::new();
                    let x = ad.var(1.0).unwrap();
                    let c = ad.constant(42.0);
                    let _ = ad.derivative(c, x, order).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ecs_eval,
    bench_ecs_gradient,
    bench_compiled_eval,
    bench_compiled_gradient,
    bench_compiled_hessian,
    bench_compiled_multipoint,
    bench_reverse_gradient,
    bench_twobody_reverse_gradient,
    bench_twobody_forward_hessian,
    bench_compile_order_scaling,
    bench_graph_depth_scaling,
    bench_gradient_vs_inputs,
    bench_constant_folding,
);
criterion_main!(benches);
