//! Basic bevy_autodiff usage example
//!
//! Demonstrates variable creation, arithmetic operations, and differentiation.
//!
//! Run with: cargo run --example basic

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variable: x = 2
    let x = ad.var(2.0).unwrap();

    // Build computation graph: f(x) = x^2 + 3x + 1
    let x_squared = ad.square(x);
    let three = ad.constant(3.0);
    let three_x = ad.mul(three, x);
    let one = ad.constant(1.0);
    let sum1 = ad.add(x_squared, three_x);
    let f = ad.add(sum1, one);

    // Evaluate f(2) = 4 + 6 + 1 = 11
    let value = ad.eval(f).unwrap();
    println!("f(2) = {}", value);

    // Differentiate: f'(x) = 2x + 3
    let dfdx = ad.differentiate(f, x).unwrap();
    println!("f'(2) = {} (expected 7)", ad.eval(dfdx).unwrap());

    // Second derivative: f''(x) = 2
    let d2fdx2 = ad.differentiate(dfdx, x).unwrap();
    println!("f''(2) = {} (expected 2)", ad.eval(d2fdx2).unwrap());
}
