//! Gradient computation example
//!
//! Demonstrates computing the gradient of a multivariate function.
//!
//! Run with: cargo run --example gradient

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variables: x=1, y=2
    let x = ad.var(1.0).unwrap();
    let y = ad.var(2.0).unwrap();

    // Build computation: f(x, y) = x^2 + x*y + y^2
    let x2 = ad.square(x);
    let xy = ad.mul(x, y);
    let y2 = ad.square(y);
    let sum1 = ad.add(x2, xy);
    let f = ad.add(sum1, y2);

    // Evaluate f(1, 2) = 1 + 2 + 4 = 7
    let value = ad.eval(f).unwrap();
    println!("f(1, 2) = {}", value);

    // Gradient: [df/dx, df/dy] = [2x + y, x + 2y] = [4, 5]
    let grad = ad.gradient(f).unwrap();
    println!("gradient = {:?} (expected [4.0, 5.0])", grad);

    // Individual partial derivatives
    let dfdx = ad.differentiate(f, x).unwrap();
    let dfdy = ad.differentiate(f, y).unwrap();
    println!("df/dx = {}", ad.eval(dfdx).unwrap());
    println!("df/dy = {}", ad.eval(dfdy).unwrap());
}
