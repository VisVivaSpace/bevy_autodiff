//! Hessian matrix computation example
//!
//! Demonstrates computing second-order partial derivatives (Hessian matrix).
//!
//! Run with: cargo run --example hessian

use vvad::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    // Create input variables: x=1, y=2
    let x = ad.var(1.0);
    let y = ad.var(2.0);

    // Build computation: f(x, y) = x^2*y + y^3
    // This has interesting second derivatives
    let x2 = ad.square(x);
    let x2y = ad.mul(x2, y);
    let y2 = ad.square(y);
    let y3 = ad.mul(y, y2);
    let f = ad.add(x2y, y3);

    // Evaluate f(1, 2) = 1*2 + 8 = 10
    let value = ad.eval(f);
    println!("f(1, 2) = {}", value);
    println!();

    // Compute Hessian matrix
    // H = | d2f/dx2   d2f/dxdy |
    //     | d2f/dydx  d2f/dy2  |
    //
    // For f = x^2*y + y^3:
    // df/dx = 2xy        df/dy = x^2 + 3y^2
    // d2f/dx2 = 2y       d2f/dxdy = 2x
    // d2f/dydx = 2x      d2f/dy2 = 6y
    //
    // At (1, 2):
    // H = | 4  2 |
    //     | 2 12 |

    let hessian = ad.hessian(f);
    println!("Hessian matrix:");
    for row in &hessian {
        println!("  {:?}", row);
    }
    println!();

    // Expected values
    let expected = [[4.0, 2.0], [2.0, 12.0]];
    println!("Expected:");
    for row in &expected {
        println!("  {:?}", row);
    }

    // Verify symmetry (Hessian should be symmetric for smooth functions)
    if hessian.len() >= 2 && hessian[0].len() >= 2 && hessian[1].len() >= 2 {
        let diff = (hessian[0][1] - hessian[1][0]).abs();
        if diff < 1e-10 {
            println!("\nHessian is symmetric (as expected)");
        }
    }
}
