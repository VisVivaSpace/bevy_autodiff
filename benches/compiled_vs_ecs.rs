//! Benchmarks comparing CompiledGraph evaluation against ECS-based derivative evaluation.
//!
//! For each expression, we measure:
//! - ECS: rebuilding AutoDiff + computing derivatives from scratch each time
//! - Compiled: compile once, then eval() at new inputs repeatedly

use bevy_autodiff::{AutoDiff, MultiIndex};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// =============================================================================
// Expression builders (build graph once, return the pieces we need)
// =============================================================================

/// sin(exp(x)) — single variable, order 2
fn build_sin_exp() -> (AutoDiff, bevy_autodiff::Var, bevy_autodiff::Var) {
    let mut ad = AutoDiff::new();
    let x = ad.var(1.0);
    let exp_x = ad.exp(x);
    let f = ad.sin(exp_x);
    (ad, x, f)
}

/// x*y + sin(x) — two variables, order 2
fn build_xy_plus_sin() -> (AutoDiff, bevy_autodiff::Var, bevy_autodiff::Var, bevy_autodiff::Var) {
    let mut ad = AutoDiff::new();
    let x = ad.var(1.0);
    let y = ad.var(1.0);
    let xy = ad.mul(x, y);
    let sin_x = ad.sin(x);
    let f = ad.add(xy, sin_x);
    (ad, x, y, f)
}

/// (1-x)² + 100(y-x²)² — Rosenbrock, two variables, order 2
fn build_rosenbrock() -> (AutoDiff, bevy_autodiff::Var, bevy_autodiff::Var, bevy_autodiff::Var) {
    let mut ad = AutoDiff::new();
    let x = ad.var(1.0);
    let y = ad.var(1.0);
    let one = ad.constant(1.0);
    let hundred = ad.constant(100.0);

    let one_minus_x = ad.sub(one, x);
    let term1 = ad.square(one_minus_x);

    let x2 = ad.square(x);
    let y_minus_x2 = ad.sub(y, x2);
    let term2_inner = ad.square(y_minus_x2);
    let term2 = ad.mul(hundred, term2_inner);

    let f = ad.add(term1, term2);
    (ad, x, y, f)
}

// =============================================================================
// Benchmark: Single variable — sin(exp(x)), order 2
// =============================================================================

fn bench_single_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_var_sin_exp");

    // --- Compiled: compile once, then eval 1000 times ---
    group.bench_function("compiled_eval", |b| {
        let (ad, x, f) = build_sin_exp();
        let mut cg = ad.compile::<3, 1>(f, &[x]);

        b.iter(|| {
            cg.eval(black_box(&[2.3]));
            let _ = black_box(cg.value());
            let _ = black_box(cg.partial(&[1]));
            let _ = black_box(cg.partial(&[2]));
        });
    });

    // --- ECS: rebuild and compute from scratch ---
    group.bench_function("ecs_derivative", |b| {
        b.iter(|| {
            let mut ad = AutoDiff::new();
            let x = ad.var(black_box(2.3));
            let exp_x = ad.exp(x);
            let f = ad.sin(exp_x);

            let _ = black_box(ad.eval(f));
            let _ = black_box(ad.derivative(f, x, 1));
            let _ = black_box(ad.derivative(f, x, 2));
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: Two variables — x*y + sin(x), all second partials
// =============================================================================

fn bench_two_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("two_var_xy_sin");

    // --- Compiled: compile once, eval repeatedly ---
    group.bench_function("compiled_eval", |b| {
        let (ad, x, y, f) = build_xy_plus_sin();
        let mut cg = ad.compile::<3, 2>(f, &[x, y]);

        b.iter(|| {
            cg.eval(black_box(&[2.3, 1.7]));
            let _ = black_box(cg.value());
            let _ = black_box(cg.partial(&[1, 0]));
            let _ = black_box(cg.partial(&[0, 1]));
            let _ = black_box(cg.partial(&[2, 0]));
            let _ = black_box(cg.partial(&[0, 2]));
            let _ = black_box(cg.partial(&[1, 1]));
        });
    });

    // --- ECS: rebuild and compute from scratch ---
    group.bench_function("ecs_partial", |b| {
        b.iter(|| {
            let mut ad = AutoDiff::new();
            let x = ad.var(black_box(2.3));
            let y = ad.var(black_box(1.7));
            let xy = ad.mul(x, y);
            let sin_x = ad.sin(x);
            let f = ad.add(xy, sin_x);

            let _ = black_box(ad.eval(f));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![1, 0])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![0, 1])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![2, 0])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![0, 2])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![1, 1])));
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: Rosenbrock — (1-x)² + 100(y-x²)², all second partials
// =============================================================================

fn bench_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("rosenbrock");

    // --- Compiled: compile once, eval repeatedly ---
    group.bench_function("compiled_eval", |b| {
        let (ad, x, y, f) = build_rosenbrock();
        let mut cg = ad.compile::<3, 2>(f, &[x, y]);

        b.iter(|| {
            cg.eval(black_box(&[0.5, 0.8]));
            let _ = black_box(cg.value());
            let _ = black_box(cg.partial(&[1, 0]));
            let _ = black_box(cg.partial(&[0, 1]));
            let _ = black_box(cg.partial(&[2, 0]));
            let _ = black_box(cg.partial(&[0, 2]));
            let _ = black_box(cg.partial(&[1, 1]));
        });
    });

    // --- ECS: rebuild and compute from scratch ---
    group.bench_function("ecs_partial", |b| {
        b.iter(|| {
            let mut ad = AutoDiff::new();
            let x = ad.var(black_box(0.5));
            let y = ad.var(black_box(0.8));
            let one = ad.constant(1.0);
            let hundred = ad.constant(100.0);

            let one_minus_x = ad.sub(one, x);
            let term1 = ad.square(one_minus_x);
            let x2 = ad.square(x);
            let y_minus_x2 = ad.sub(y, x2);
            let term2_inner = ad.square(y_minus_x2);
            let term2 = ad.mul(hundred, term2_inner);
            let f = ad.add(term1, term2);

            let _ = black_box(ad.eval(f));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![1, 0])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![0, 1])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![2, 0])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![0, 2])));
            let _ = black_box(ad.partial(f, &MultiIndex::new(vec![1, 1])));
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: Compile cost (one-time overhead)
// =============================================================================

fn bench_compile_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile_cost");

    group.bench_function("single_var_sin_exp", |b| {
        let (ad, x, f) = build_sin_exp();
        b.iter(|| {
            let _ = black_box(ad.compile::<3, 1>(f, &[x]));
        });
    });

    group.bench_function("two_var_xy_sin", |b| {
        let (ad, x, y, f) = build_xy_plus_sin();
        b.iter(|| {
            let _ = black_box(ad.compile::<3, 2>(f, &[x, y]));
        });
    });

    group.bench_function("rosenbrock", |b| {
        let (ad, x, y, f) = build_rosenbrock();
        b.iter(|| {
            let _ = black_box(ad.compile::<3, 2>(f, &[x, y]));
        });
    });

    group.finish();
}

// =============================================================================
// Benchmark: Compiled eval throughput (batch of varying size)
// =============================================================================

fn bench_compiled_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_throughput");

    for count in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("rosenbrock_evals", count),
            &count,
            |b, &count| {
                let (ad, x, y, f) = build_rosenbrock();
                let mut cg = ad.compile::<3, 2>(f, &[x, y]);

                b.iter(|| {
                    for i in 0..count {
                        let t = i as f64 * 0.001;
                        cg.eval(black_box(&[0.5 + t, 0.8 + t]));
                        let _ = black_box(cg.partial(&[1, 0]));
                        let _ = black_box(cg.partial(&[0, 1]));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_var,
    bench_two_var,
    bench_rosenbrock,
    bench_compile_cost,
    bench_compiled_throughput,
);
criterion_main!(benches);
