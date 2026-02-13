//! Gradient computation example
//!
//! Demonstrates building a multivariate computation graph.
//! Gradient computation will be re-enabled after differentiate() is implemented.
//!
//! Run with: cargo run --example gradient

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variables: x=1, y=2
    let x = ad.var(1.0);
    let y = ad.var(2.0);

    // Build computation: f(x, y) = x^2 + x*y + y^2
    let x2 = ad.square(x);
    let xy = ad.mul(x, y);
    let y2 = ad.square(y);
    let sum1 = ad.add(x2, xy);
    let f = ad.add(sum1, y2);

    // Evaluate f(1, 2) = 1 + 2 + 4 = 7
    let value = ad.eval(f);
    println!("f(1, 2) = {}", value);

    // Gradient computation will be re-enabled after differentiate() is implemented
}
