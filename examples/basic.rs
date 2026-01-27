//! Basic vvad usage example
//!
//! Demonstrates variable creation, arithmetic operations, and derivative computation.
//!
//! Run with: cargo run --example basic

use vvad::AutoDiff;

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

    // First derivative: f'(x) = 2x + 3
    // At x=2: f'(2) = 4 + 3 = 7
    let df_dx = ad.derivative(f, x, 1);
    println!("f'(2) = {}", df_dx);

    // Second derivative: f''(x) = 2
    let d2f_dx2 = ad.derivative(f, x, 2);
    println!("f''(2) = {}", d2f_dx2);
}
