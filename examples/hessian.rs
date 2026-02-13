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

    // Gradient
    let dfdx = ad.differentiate(f, x);
    let dfdy = ad.differentiate(f, y);
    println!("df/dx = {} (expected 4: 2xy = 2*1*2)", ad.eval(dfdx));
    println!("df/dy = {} (expected 13: x² + 3y² = 1 + 12)", ad.eval(dfdy));

    // Hessian (second derivatives)
    let d2fdx2 = ad.differentiate(dfdx, x);
    let d2fdxdy = ad.differentiate(dfdx, y);
    let d2fdydx = ad.differentiate(dfdy, x);
    let d2fdy2 = ad.differentiate(dfdy, y);

    println!("\nHessian matrix at (1, 2):");
    println!("  d²f/dx²  = {} (expected 4: 2y)", ad.eval(d2fdx2));
    println!("  d²f/dxdy = {} (expected 2: 2x)", ad.eval(d2fdxdy));
    println!("  d²f/dydx = {} (expected 2: 2x)", ad.eval(d2fdydx));
    println!("  d²f/dy²  = {} (expected 12: 6y)", ad.eval(d2fdy2));

    // Verify mixed partial symmetry: d²f/dxdy = d²f/dydx
    assert_eq!(ad.eval(d2fdxdy), ad.eval(d2fdydx));
    println!("\nMixed partial symmetry verified!");
}
