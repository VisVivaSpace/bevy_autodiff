//! Hessian matrix computation example
//!
//! Demonstrates computing second-order partial derivatives
//! using successive differentiation.
//!
//! Run with: cargo run --example hessian

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variables: x=1, y=2
    let x = ad.var(1.0).unwrap();
    let y = ad.var(2.0).unwrap();

    // Build computation: f(x, y) = x^2*y + y^3
    let x2 = ad.square(x);
    let x2y = ad.mul(x2, y);
    let y2 = ad.square(y);
    let y3 = ad.mul(y, y2);
    let f = ad.add(x2y, y3);

    // Evaluate f(1, 2) = 1*2 + 8 = 10
    let value = ad.eval(f).unwrap();
    println!("f(1, 2) = {}", value);

    // Gradient
    let dfdx = ad.differentiate(f, x).unwrap();
    let dfdy = ad.differentiate(f, y).unwrap();
    println!("df/dx = {} (expected 4: 2xy = 2*1*2)", ad.eval(dfdx).unwrap());
    println!("df/dy = {} (expected 13: x² + 3y² = 1 + 12)", ad.eval(dfdy).unwrap());

    // Hessian (second derivatives)
    let d2fdx2 = ad.differentiate(dfdx, x).unwrap();
    let d2fdxdy = ad.differentiate(dfdx, y).unwrap();
    let d2fdydx = ad.differentiate(dfdy, x).unwrap();
    let d2fdy2 = ad.differentiate(dfdy, y).unwrap();

    println!("\nHessian matrix at (1, 2):");
    println!("  d²f/dx²  = {} (expected 4: 2y)", ad.eval(d2fdx2).unwrap());
    println!("  d²f/dxdy = {} (expected 2: 2x)", ad.eval(d2fdxdy).unwrap());
    println!("  d²f/dydx = {} (expected 2: 2x)", ad.eval(d2fdydx).unwrap());
    println!("  d²f/dy²  = {} (expected 12: 6y)", ad.eval(d2fdy2).unwrap());

    // Verify mixed partial symmetry: d²f/dxdy = d²f/dydx
    assert_eq!(ad.eval(d2fdxdy).unwrap(), ad.eval(d2fdydx).unwrap());
    println!("\nMixed partial symmetry verified!");
}
