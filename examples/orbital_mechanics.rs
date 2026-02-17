//! Orbital mechanics example
//!
//! Demonstrates computing gravitational potential and its gradient.
//!
//! Run with: cargo run --example orbital_mechanics

use bevy_autodiff::AutoDiff;

fn main() {
    let mu = 398600.4418; // Earth gravitational parameter (km^3/s^2)

    let x_val: f64 = 6778.0;
    let y_val: f64 = 0.0;
    let z_val: f64 = 0.0;

    let mut ad = AutoDiff::new();

    let x = ad.var(x_val).unwrap();
    let y = ad.var(y_val).unwrap();
    let z = ad.var(z_val).unwrap();

    // r = sqrt(x^2 + y^2 + z^2)
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let z2 = ad.square(z);
    let xy_sum = ad.add(x2, y2);
    let r2 = ad.add(xy_sum, z2);
    let r = ad.sqrt(r2);

    // V = -mu/r
    let mu_const = ad.constant(mu);
    let mu_over_r = ad.div(mu_const, r);
    let neg_one = ad.constant(-1.0);
    let v = ad.mul(neg_one, mu_over_r);

    let v_val = ad.eval(v).unwrap();
    let r_val = (x_val.powi(2) + y_val.powi(2) + z_val.powi(2)).sqrt();
    println!("Potential V = {} km^2/s^2", v_val);
    println!("Expected:   V = {} km^2/s^2", -mu / r_val);

    // Gradient of potential: acceleration = -grad(V)
    let grad = ad.gradient(v).unwrap();
    println!("\nGradient of V:");
    println!("  dV/dx = {} (acceleration x-component)", grad[0]);
    println!("  dV/dy = {}", grad[1]);
    println!("  dV/dz = {}", grad[2]);

    // Validate: autodiff gradient matches the analytical closed-form derivative.
    // V = -mu/r, so dV/dx = mu*x/r^3. This is a closed-form f64 comparison
    // with ~2 rounding operations, so 1e-10 (≈50ε relative) is conservative.
    let expected_dvdx: f64 = mu * x_val / r_val.powi(3);
    println!("\nExpected dV/dx = {}", expected_dvdx);
    assert!(
        (grad[0] - expected_dvdx).abs() < 1e-10 * expected_dvdx.abs(),
        "autodiff dV/dx = {}, analytical = {}, diff = {:.2e}",
        grad[0],
        expected_dvdx,
        (grad[0] - expected_dvdx).abs()
    );
}
