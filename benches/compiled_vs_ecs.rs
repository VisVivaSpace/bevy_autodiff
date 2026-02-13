//! Benchmarks temporarily disabled during Taylor->symbolic differentiation migration.
//! Will be re-enabled in Step 5.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder_bench(_c: &mut Criterion) {
    // Benchmarks will be re-enabled after CompiledGraph is rebuilt
}

criterion_group!(benches, placeholder_bench);
criterion_main!(benches);
