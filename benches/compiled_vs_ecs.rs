//! Benchmarks comparing CompiledGraph vs ECS-based evaluation.
//!
//! Run with: cargo bench

use criterion::{criterion_group, criterion_main, Criterion};

use bevy_autodiff::AutoDiff;

/// Build Rosenbrock function: (1-x)² + 100*(y-x²)²
fn build_rosenbrock(
    ad: &mut AutoDiff,
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

criterion_group!(
    benches,
    bench_ecs_eval,
    bench_ecs_gradient,
    bench_compiled_eval,
    bench_compiled_gradient,
    bench_compiled_hessian,
    bench_compiled_multipoint,
);
criterion_main!(benches);
