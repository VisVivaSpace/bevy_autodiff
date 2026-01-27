//! Orbital mechanics example
//!
//! Demonstrates computing gravitational potential derivatives for space mission design.
//! Uses the two-body gravitational potential V = -mu/r where r = sqrt(x^2 + y^2 + z^2).
//!
//! Run with: cargo run --example orbital_mechanics

use vvad::AutoDiff;

fn main() {
    // Gravitational parameter (km^3/s^2) - Earth
    let mu = 398600.4418;

    // Position vector (km) - typical LEO orbit radius
    let x_val = 6778.0;
    let y_val = 0.0;
    let z_val = 0.0;

    let mut ad = AutoDiff::new();

    // Create position variables
    let x = ad.var(x_val);
    let y = ad.var(y_val);
    let z = ad.var(z_val);

    // Compute r = sqrt(x^2 + y^2 + z^2)
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let z2 = ad.square(z);
    let xy_sum = ad.add(x2, y2);
    let r2 = ad.add(xy_sum, z2);
    let r = ad.sqrt(r2);

    // Compute gravitational potential V = -mu/r
    let mu_const = ad.constant(mu);
    let mu_over_r = ad.div(mu_const, r);
    let neg_one = ad.constant(-1.0);
    let v = ad.mul(neg_one, mu_over_r);

    // Evaluate potential
    let r_val = (x_val.powi(2) + y_val.powi(2) + z_val.powi(2)).sqrt();
    let v_expected = -mu / r_val;

    println!("Two-Body Gravitational Potential");
    println!("================================");
    println!("mu = {} km^3/s^2 (Earth)", mu);
    println!("Position: ({}, {}, {}) km", x_val, y_val, z_val);
    println!("Radius: {} km", r_val);
    println!();

    let v_val = ad.eval(v);
    println!("Potential V = {} km^2/s^2", v_val);
    println!("Expected:   V = {} km^2/s^2", v_expected);
    println!();

    // Compute gradient (acceleration = -grad(V))
    // For V = -mu/r:
    // dV/dx = mu*x/r^3, dV/dy = mu*y/r^3, dV/dz = mu*z/r^3
    // Acceleration = -grad(V) = -mu*r_vec/r^3
    let grad_v = ad.gradient(v);

    println!("Gradient of potential (dV/dx, dV/dy, dV/dz):");
    println!("  Computed: ({:.6}, {:.6}, {:.6})", grad_v[0], grad_v[1], grad_v[2]);

    // Expected gradient
    let r3 = r_val.powi(3);
    let dv_dx_expected = mu * x_val / r3;
    let dv_dy_expected = mu * y_val / r3;
    let dv_dz_expected = mu * z_val / r3;
    println!(
        "  Expected: ({:.6}, {:.6}, {:.6})",
        dv_dx_expected, dv_dy_expected, dv_dz_expected
    );
    println!();

    // Gravitational acceleration = -grad(V)
    println!("Gravitational acceleration (km/s^2):");
    println!("  a = ({:.6}, {:.6}, {:.6})", -grad_v[0], -grad_v[1], -grad_v[2]);
    let a_mag = (grad_v[0].powi(2) + grad_v[1].powi(2) + grad_v[2].powi(2)).sqrt();
    println!("  |a| = {:.6} km/s^2", a_mag);
    println!("  |a| = {:.3} m/s^2", a_mag * 1000.0);
    println!();

    // For comparison: Earth surface gravity
    let r_earth = 6371.0; // km
    let g_surface = mu / (r_earth * r_earth);
    println!(
        "Earth surface gravity: {:.3} m/s^2 (for reference)",
        g_surface * 1000.0
    );

    // Compute second derivatives (useful for state transition matrix)
    println!();
    println!("Second derivatives (Hessian of V):");
    let hessian = ad.hessian(v);
    for (i, row) in hessian.iter().enumerate() {
        let labels = ["x", "y", "z"];
        print!("  d2V/d{}d_: ", labels[i]);
        for (j, val) in row.iter().enumerate() {
            print!("{}={:.6e}  ", labels[j], val);
        }
        println!();
    }
}
