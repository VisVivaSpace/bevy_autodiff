//! Gradient computation example
//!
//! Demonstrates forward and reverse mode gradient computation for multivariate functions.
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

    // Compute gradient using forward mode
    // df/dx = 2x + y = 2(1) + 2 = 4
    // df/dy = x + 2y = 1 + 2(2) = 5
    let grad = ad.gradient(f);
    println!("Forward gradient: {:?}", grad);

    // Compute gradient using reverse mode (more efficient for many inputs)
    let grad_rev = ad.gradient_reverse(f);
    println!("Reverse gradient: {:?}", grad_rev);

    // Verify gradients match
    let df_dx_expected = 4.0;
    let df_dy_expected = 5.0;

    println!("\nExpected: df/dx = {}, df/dy = {}", df_dx_expected, df_dy_expected);

    // Check forward mode
    if grad.len() >= 2 {
        let diff_x = (grad[0] - df_dx_expected).abs();
        let diff_y = (grad[1] - df_dy_expected).abs();
        if diff_x < 1e-10 && diff_y < 1e-10 {
            println!("Forward gradient matches expected values!");
        }
    }

    // Check reverse mode
    if grad_rev.len() >= 2 {
        let diff_x = (grad_rev[0] - df_dx_expected).abs();
        let diff_y = (grad_rev[1] - df_dy_expected).abs();
        if diff_x < 1e-10 && diff_y < 1e-10 {
            println!("Reverse gradient matches expected values!");
        }
    }
}
