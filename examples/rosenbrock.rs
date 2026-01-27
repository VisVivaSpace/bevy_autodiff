//! Rosenbrock function optimization example
//!
//! Demonstrates gradient descent on the classic Rosenbrock "banana" function.
//! The Rosenbrock function is f(x,y) = (a-x)^2 + b(y-x^2)^2
//! with typical values a=1, b=100.
//!
//! The global minimum is at (a, a^2) = (1, 1) where f(1,1) = 0.
//!
//! Run with: cargo run --example rosenbrock

use bevy_autodiff::AutoDiff;

fn main() {
    // Rosenbrock parameters
    let a = 1.0;
    let b = 100.0;

    // Starting point
    let mut x_val = -1.0;
    let mut y_val = 1.0;

    // Learning rate
    let lr = 0.001;
    let max_iterations = 10000;

    println!("Optimizing Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2");
    println!("Parameters: a={}, b={}", a, b);
    println!("Starting point: ({}, {})", x_val, y_val);
    println!("Learning rate: {}", lr);
    println!();

    for iter in 0..max_iterations {
        // Create a fresh AutoDiff context each iteration
        let mut ad = AutoDiff::new();

        // Create variables at current position
        let x = ad.var(x_val);
        let y = ad.var(y_val);

        // Build Rosenbrock function: (a-x)^2 + b*(y-x^2)^2
        let a_const = ad.constant(a);
        let b_const = ad.constant(b);

        // (a - x)
        let a_minus_x = ad.sub(a_const, x);
        // (a - x)^2
        let term1 = ad.square(a_minus_x);

        // x^2
        let x_sq = ad.square(x);
        // (y - x^2)
        let y_minus_x_sq = ad.sub(y, x_sq);
        // (y - x^2)^2
        let diff_sq = ad.square(y_minus_x_sq);
        // b * (y - x^2)^2
        let term2 = ad.mul(b_const, diff_sq);

        // f = (a-x)^2 + b*(y-x^2)^2
        let f = ad.add(term1, term2);

        // Evaluate function value
        let f_val = ad.eval(f);

        // Compute gradient
        let grad = ad.gradient(f);

        // Print progress periodically
        if iter % 1000 == 0 || iter == max_iterations - 1 {
            println!(
                "Iter {}: f({:.6}, {:.6}) = {:.10}, |grad| = {:.6}",
                iter,
                x_val,
                y_val,
                f_val,
                (grad[0].powi(2) + grad[1].powi(2)).sqrt()
            );
        }

        // Check convergence
        if f_val < 1e-12 {
            println!("\nConverged at iteration {}!", iter);
            break;
        }

        // Gradient descent update
        x_val -= lr * grad[0];
        y_val -= lr * grad[1];
    }

    println!("\nFinal position: ({:.6}, {:.6})", x_val, y_val);
    println!("Expected minimum: ({}, {})", a, a * a);
    println!(
        "Distance from minimum: {:.6}",
        ((x_val - a).powi(2) + (y_val - a * a).powi(2)).sqrt()
    );
}
