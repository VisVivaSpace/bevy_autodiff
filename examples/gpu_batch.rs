//! GPU batch evaluation example.
//!
//! Demonstrates evaluating a compiled computation graph at many input points
//! in parallel on the GPU.
//!
//! Run with: cargo run --example gpu_batch --features wgpu

use bevy_autodiff::gpu::GpuContext;
use bevy_autodiff::AutoDiff;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create GPU context
    let gpu = GpuContext::new()?;
    println!("GPU context created.");

    // 2. Build computation graph on CPU
    let mut ad = AutoDiff::new();
    let x = ad.var(0.0);
    let y = ad.var(0.0);

    // f(x, y) = sin(x * y) + exp(x)
    let xy = ad.mul(x, y);
    let sin_xy = ad.sin(xy);
    let exp_x = ad.exp(x);
    let f = ad.add(sin_xy, exp_x);

    // 3. Compile with first-order partials
    let graph = ad.compile_order(f, &[x, y], 1);
    println!(
        "Graph compiled: {} nodes, {} inputs",
        graph.num_nodes(),
        graph.num_inputs()
    );

    // 4. Prepare for GPU
    let gpu_graph = gpu.prepare(&graph)?;

    // 5. Generate sample data (e.g., Monte Carlo inputs)
    let num_samples = 1_000_000;
    let x_samples: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 / num_samples as f32) * 2.0 - 1.0) // [-1, 1)
        .collect();
    let y_samples: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 / num_samples as f32) * 3.0) // [0, 3)
        .collect();

    // 6. Dispatch to GPU
    let start = std::time::Instant::now();
    let results = gpu_graph.eval_batch(&gpu, &[&x_samples, &y_samples])?;
    let elapsed = start.elapsed();

    println!(
        "Evaluated {} samples in {:.2?} ({:.0} samples/sec)",
        results.num_samples(),
        elapsed,
        results.num_samples() as f64 / elapsed.as_secs_f64()
    );

    // 7. Read results
    let values = results.values();
    println!("f(0, 0) = {:.6}", values[num_samples / 2]); // x=0, y≈1.5

    if let Some(dfdx) = results.partials(&[1, 0]) {
        println!("df/dx(0, 0) = {:.6}", dfdx[num_samples / 2]);
    }
    if let Some(dfdy) = results.partials(&[0, 1]) {
        println!("df/dy(0, 0) = {:.6}", dfdy[num_samples / 2]);
    }

    // 8. Show a few sample outputs
    println!("\nFirst 5 samples:");
    for i in 0..5 {
        print!("  x={:.3}, y={:.3}", x_samples[i], y_samples[i]);
        print!(" => f={:.6}", values[i]);
        if let Some(dfdx) = results.partials(&[1, 0]) {
            print!(", df/dx={:.6}", dfdx[i]);
        }
        println!();
    }

    Ok(())
}
