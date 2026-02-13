//! Rosenbrock function example
//!
//! Demonstrates building the Rosenbrock computation graph.
//! Gradient-based optimization will be re-enabled after differentiate() is implemented.
//!
//! Run with: cargo run --example rosenbrock

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    let x = ad.var(1.0);
    let y = ad.var(1.0);

    // Build Rosenbrock function: (1-x)^2 + 100*(y-x^2)^2
    let one = ad.constant(1.0);
    let hundred = ad.constant(100.0);

    let a_minus_x = ad.sub(one, x);
    let term1 = ad.square(a_minus_x);

    let x_sq = ad.square(x);
    let y_minus_x_sq = ad.sub(y, x_sq);
    let diff_sq = ad.square(y_minus_x_sq);
    let term2 = ad.mul(hundred, diff_sq);

    let f = ad.add(term1, term2);

    // At minimum (1,1): f = 0
    let value = ad.eval(f);
    println!("f(1, 1) = {} (should be 0)", value);

    // Gradient-based optimization will be re-enabled after differentiate() is implemented
}
