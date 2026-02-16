//! Parallel evaluation using Bevy's ECS scheduler.
//!
//! Demonstrates the shallow Bevy integration pattern:
//! 1. Build and compile a graph using `AutoDiff` (private World)
//! 2. Clone the compiled graph to many entities as a `Component`
//! 3. Use `par_iter_mut()` to evaluate all graphs in parallel via Bevy's `ComputeTaskPool`
//!
//! Run with: cargo run --example bevy_par_eval

use bevy_autodiff::{AutoDiff, CompiledGraph};
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;

/// Per-entity input values for evaluation.
#[derive(Component)]
struct InputPoint {
    x: f64,
    y: f64,
}

/// System that evaluates all CompiledGraphs in parallel.
fn eval_system(mut query: Query<(&InputPoint, &mut CompiledGraph)>) {
    query.par_iter_mut().for_each(|(point, mut cg)| {
        cg.eval(&[point.x, point.y]).unwrap();
    });
}

fn main() {
    // 1. Build computation graph using AutoDiff's private World
    let mut ad = AutoDiff::new();
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();

    // f(x, y) = x² + x·y + y²
    let x2 = ad.square(x);
    let xy = ad.mul(x, y);
    let y2 = ad.square(y);
    let sum = ad.add(x2, xy);
    let f = ad.add(sum, y2);

    // 2. Compile once (this is the artifact that crosses into the app's ECS)
    let template = ad.compile_primal(f, &[x, y]).unwrap();

    // 3. Create a Bevy World and spawn entities with cloned graphs
    let num_entities = 1000;
    let mut world = World::new();

    for i in 0..num_entities {
        let t = i as f64 / num_entities as f64;
        world.spawn((
            InputPoint {
                x: t * 4.0 - 2.0, // [-2, 2)
                y: t * 3.0 - 1.0, // [-1, 2)
            },
            template.clone(),
        ));
    }

    // 4. Run the parallel evaluation system
    let start = std::time::Instant::now();
    world.run_system_once(eval_system).unwrap();
    let elapsed = start.elapsed();

    println!("Evaluated {num_entities} entities in {elapsed:.2?}");

    // 5. Read back results (gradient() takes &mut self for backward pass)
    let mut query = world.query::<(&InputPoint, &mut CompiledGraph)>();
    let mut count = 0;
    for (point, mut cg) in query.iter_mut(&mut world) {
        if count < 5 {
            let grad = cg.gradient().to_vec();
            let val = cg.value();
            println!(
                "  ({:.2}, {:.2}) => f={:.4}, grad=[{:.4}, {:.4}]",
                point.x, point.y, val, grad[0], grad[1]
            );
        }
        count += 1;
    }
    println!("  ... ({} total)", count);
}
