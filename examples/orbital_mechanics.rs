//! Orbital mechanics example
//!
//! Demonstrates computing gravitational potential.
//! Gradient/Hessian computation will be re-enabled after differentiate() is implemented.
//!
//! Run with: cargo run --example orbital_mechanics

use bevy_autodiff::AutoDiff;

fn main() {
    let mu = 398600.4418; // Earth gravitational parameter (km^3/s^2)

    let x_val = 6778.0;
    let y_val = 0.0;
    let z_val = 0.0;

    let mut ad = AutoDiff::new();

    let x = ad.var(x_val);
    let y = ad.var(y_val);
    let z = ad.var(z_val);

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

    let v_val = ad.eval(v);
    let r_val = (x_val.powi(2) + y_val.powi(2) + z_val.powi(2)).sqrt();
    println!("Potential V = {} km^2/s^2", v_val);
    println!("Expected:   V = {} km^2/s^2", -mu / r_val);

    // Gradient/Hessian computation will be re-enabled after differentiate() is implemented
}
