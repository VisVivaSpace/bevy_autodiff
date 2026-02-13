//! Basic bevy_autodiff usage example
//!
//! Demonstrates variable creation and arithmetic operations.
//!
//! Run with: cargo run --example basic

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variable: x = 2
    let x = ad.var(2.0);

    // Build computation: f(x) = x^2 + 3x + 1
    let x_squared = ad.square(x);
    let three = ad.constant(3.0);
    let three_x = ad.mul(three, x);
    let one = ad.constant(1.0);
    let sum1 = ad.add(x_squared, three_x);
    let f = ad.add(sum1, one);

    // Evaluate f(2) = 4 + 6 + 1 = 11
    let value = ad.eval(f);
    println!("f(2) = {}", value);

    // Derivative computation will be re-enabled after differentiate() is implemented
}
