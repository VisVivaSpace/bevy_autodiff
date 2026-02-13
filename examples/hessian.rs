//! Hessian matrix computation example
//!
//! Demonstrates building a computation graph for second-order derivatives.
//! Hessian computation will be re-enabled after differentiate() is implemented.
//!
//! Run with: cargo run --example hessian

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variables: x=1, y=2
    let x = ad.var(1.0);
    let y = ad.var(2.0);

    // Build computation: f(x, y) = x^2*y + y^3
    let x2 = ad.square(x);
    let x2y = ad.mul(x2, y);
    let y2 = ad.square(y);
    let y3 = ad.mul(y, y2);
    let f = ad.add(x2y, y3);

    // Evaluate f(1, 2) = 1*2 + 8 = 10
    let value = ad.eval(f);
    println!("f(1, 2) = {}", value);

    // Hessian computation will be re-enabled after differentiate() is implemented
}
