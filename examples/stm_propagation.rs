//! State Transition Matrix propagation using bevy_autodiff + rkf78
//!
//! Demonstrates how to:
//! 1. Build the gravity gradient Jacobian symbolically with bevy_autodiff
//! 2. Compile the derivatives ONCE, before integration
//! 3. Evaluate them efficiently inside the rkf78 right-hand side
//! 4. Propagate the 6×6 State Transition Matrix alongside a two-body orbit
//!
//! The key pattern: the Jacobian is defined and compiled *outside* the ODE
//! right-hand side function. At each integration step, the pre-compiled
//! graphs are simply evaluated at the current position.
//!
//! Run with: cargo run --example stm_propagation

use std::cell::RefCell;

use bevy_autodiff::{AutoDiff, CompiledGraph};
use rkf78::{OdeSystem, Rkf78, Tolerances};

/// Earth gravitational parameter (km³/s²)
const MU: f64 = 398600.4418;

// =========================================================================
// Step 1: Build the Jacobian outside the RHS
// =========================================================================

/// Two-body orbital dynamics with State Transition Matrix.
///
/// The gravity gradient Jacobian (∂a/∂r) is computed automatically
/// by bevy_autodiff. The compiled graphs are built once and reused
/// at every integration step.
///
/// State layout (42 components):
///   [0..3]   position (x, y, z) in km
///   [3..6]   velocity (vx, vy, vz) in km/s
///   [6..42]  STM Φ (6×6, row-major)
struct TwoBodyStm {
    /// Compiled graph for ax = -μx/r³, with ∂ax/∂(x,y,z)
    cg_ax: RefCell<CompiledGraph<f64>>,
    /// Compiled graph for ay = -μy/r³, with ∂ay/∂(x,y,z)
    cg_ay: RefCell<CompiledGraph<f64>>,
    /// Compiled graph for az = -μz/r³, with ∂az/∂(x,y,z)
    cg_az: RefCell<CompiledGraph<f64>>,
}

/// Builds the two-body force model and compiles its Jacobian.
///
/// This happens ONCE before integration. The returned struct holds
/// pre-compiled derivative graphs that can be evaluated at any
/// position without rebuilding.
fn build_two_body_system() -> TwoBodyStm {
    let mut ad = AutoDiff::new();

    // Inputs: position components
    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();
    let z = ad.var(0.0).unwrap();

    // r² = x² + y² + z²
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let z2 = ad.square(z);
    let sum_xy = ad.add(x2, y2);
    let r2 = ad.add(sum_xy, z2);

    // r = sqrt(r²)
    let r = ad.sqrt(r2);

    // -μ / r³
    let mu_const = ad.constant(MU);
    let neg_mu = ad.neg(mu_const);
    let r3 = ad.mul(r, r2);
    let factor = ad.div(neg_mu, r3);

    // Acceleration: a = (-μ/r³) * r_vec
    let ax = ad.mul(factor, x);
    let ay = ad.mul(factor, y);
    let az = ad.mul(factor, z);

    // Compile each acceleration component with first-order partials
    // w.r.t. [x, y, z]. After this, evaluating the Jacobian at any
    // position is a single forward pass through a flat array of ops.
    let inputs = [x, y, z];
    let partials = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];

    let cg_ax = ad.compile(ax, &inputs, &partials).unwrap();
    let cg_ay = ad.compile(ay, &inputs, &partials).unwrap();
    let cg_az = ad.compile(az, &inputs, &partials).unwrap();

    TwoBodyStm {
        cg_ax: RefCell::new(cg_ax),
        cg_ay: RefCell::new(cg_ay),
        cg_az: RefCell::new(cg_az),
    }
}

// =========================================================================
// Step 2: Implement the ODE system for rkf78
// =========================================================================

/// Evaluates a compiled acceleration graph at `pos`, returning the
/// acceleration value and its 3 partial derivatives (one row of G).
fn eval_accel_row(cg: &RefCell<CompiledGraph<f64>>, pos: &[f64; 3]) -> (f64, [f64; 3]) {
    let mut cg = cg.borrow_mut();
    cg.eval(pos).unwrap();
    let val = cg.value();
    let d0 = cg.partial(&[1, 0, 0]).unwrap();
    let d1 = cg.partial(&[0, 1, 0]).unwrap();
    let d2 = cg.partial(&[0, 0, 1]).unwrap();
    (val, [d0, d1, d2])
}

impl OdeSystem<42> for TwoBodyStm {
    fn rhs(&self, _t: f64, y: &[f64; 42], dydt: &mut [f64; 42]) {
        let pos = [y[0], y[1], y[2]];

        // Evaluate compiled graphs at current position.
        // Each call returns the acceleration AND its 3 partial derivatives.
        let (ax, g_row0) = eval_accel_row(&self.cg_ax, &pos);
        let (ay, g_row1) = eval_accel_row(&self.cg_ay, &pos);
        let (az, g_row2) = eval_accel_row(&self.cg_az, &pos);

        // Gravity gradient: G = ∂a/∂r (3×3)
        let g = [g_row0, g_row1, g_row2];

        // --- Orbital state dynamics ---
        // dr/dt = v
        dydt[0] = y[3];
        dydt[1] = y[4];
        dydt[2] = y[5];
        // dv/dt = a(r)
        dydt[3] = ax;
        dydt[4] = ay;
        dydt[5] = az;

        // --- STM dynamics: dΦ/dt = A · Φ ---
        //
        //     [ 0₃  I₃ ]
        // A = [         ]
        //     [ G   0₃ ]
        //
        // Top 3 rows:    (A·Φ)[i][j]   = Φ[i+3][j]           (identity block)
        // Bottom 3 rows: (A·Φ)[i+3][j] = Σ_k G[i][k]·Φ[k][j] (gravity gradient)
        for j in 0..6 {
            for i in 0..3 {
                // Top: dΦ[i][j]/dt = Φ[i+3][j]
                dydt[6 + i * 6 + j] = y[6 + (i + 3) * 6 + j];

                // Bottom: dΦ[i+3][j]/dt = Σ_k G[i][k] · Φ[k][j]
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += g[i][k] * y[6 + k * 6 + j];
                }
                dydt[6 + (i + 3) * 6 + j] = sum;
            }
        }
    }
}

// =========================================================================
// Helpers
// =========================================================================

/// Extracts the 6×6 STM from the 42-element state vector.
fn extract_stm(y: &[f64; 42]) -> [[f64; 6]; 6] {
    let mut phi = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            phi[i][j] = y[6 + i * 6 + j];
        }
    }
    phi
}

/// Determinant of a 6×6 matrix via Gaussian elimination with partial pivoting.
fn det_6x6(m: &[[f64; 6]; 6]) -> f64 {
    let mut a = *m;
    let mut det = 1.0;
    for col in 0..6 {
        let mut max_row = col;
        for row in (col + 1)..6 {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        if max_row != col {
            a.swap(col, max_row);
            det = -det;
        }
        if a[col][col].abs() < 1e-30 {
            return 0.0;
        }
        det *= a[col][col];
        #[allow(clippy::needless_range_loop)]
        for row in (col + 1)..6 {
            let factor = a[row][col] / a[col][col];
            for k in (col + 1)..6 {
                a[row][k] -= factor * a[col][k];
            }
        }
    }
    det
}

/// Analytical gravity gradient for two-body (for verification).
///
/// G_ij = (μ/r⁵)(3 rᵢ rⱼ − r² δᵢⱼ)
fn analytical_gravity_gradient(pos: [f64; 3]) -> [[f64; 3]; 3] {
    let r2 = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
    let r = r2.sqrt();
    let r5 = r2 * r2 * r;
    let mu_r5 = MU / r5;

    let mut g = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let kronecker = if i == j { r2 } else { 0.0 };
            g[i][j] = mu_r5 * (3.0 * pos[i] * pos[j] - kronecker);
        }
    }
    g
}

// =========================================================================
// Main
// =========================================================================

fn main() {
    // --- Build the force model and compile its Jacobian (once) ---
    let sys = build_two_body_system();

    // --- Verify autodiff Jacobian against analytical formula ---
    let r0 = 6778.0; // km (LEO, ~400 km altitude)
    let pos = [r0, 0.0, 0.0];

    let (_, ad_row0) = eval_accel_row(&sys.cg_ax, &pos);
    let (_, ad_row1) = eval_accel_row(&sys.cg_ay, &pos);
    let g_analytical = analytical_gravity_gradient(pos);

    println!("=== Jacobian verification at r = [{}, 0, 0] km ===", r0);
    println!(
        "  dax/dx: autodiff = {:+.10e}, analytical = {:+.10e}",
        ad_row0[0], g_analytical[0][0]
    );
    println!(
        "  dax/dy: autodiff = {:+.10e}, analytical = {:+.10e}",
        ad_row0[1], g_analytical[0][1]
    );
    println!(
        "  day/dx: autodiff = {:+.10e}, analytical = {:+.10e}",
        ad_row1[0], g_analytical[1][0]
    );

    let max_err = (0..3)
        .flat_map(|i| {
            let ad_row = match i {
                0 => ad_row0,
                1 => ad_row1,
                _ => eval_accel_row(&sys.cg_az, &pos).1,
            };
            (0..3).map(move |j| (ad_row[j] - g_analytical[i][j]).abs())
        })
        .fold(0.0_f64, f64::max);
    println!("  Max error across 3x3: {:.2e}", max_err);

    // --- Initial conditions: circular LEO orbit in x-y plane ---
    let v0 = (MU / r0).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / MU).sqrt();

    let mut y0 = [0.0f64; 42];
    y0[0] = r0; //  x = r0
    y0[4] = v0; // vy = circular velocity
    // STM initialized to 6×6 identity
    for i in 0..6 {
        y0[6 + i * 6 + i] = 1.0;
    }

    println!("\n=== Circular LEO orbit ===");
    println!("  Altitude: {:.0} km", r0 - 6378.0);
    println!("  Radius:   {:.1} km", r0);
    println!("  Velocity: {:.6} km/s", v0);
    println!("  Period:   {:.1} s ({:.1} min)", period, period / 60.0);

    // --- Integrate one orbital period ---
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut solver = Rkf78::new(tol);

    let (_tf, yf) = solver
        .integrate(&sys, 0.0, &y0, period, 10.0)
        .expect("Integration failed");

    // --- Results ---
    let pos_err = ((yf[0] - r0).powi(2) + yf[1].powi(2) + yf[2].powi(2)).sqrt();

    println!("\n=== After one orbital period ===");
    println!(
        "  Final position: [{:.6}, {:.6}, {:.6}] km",
        yf[0], yf[1], yf[2]
    );
    println!(
        "  Final velocity: [{:.6}, {:.6}, {:.6}] km/s",
        yf[3], yf[4], yf[5]
    );
    println!("  Position return error: {:.2e} km", pos_err);

    // --- State Transition Matrix ---
    let phi = extract_stm(&yf);
    let det = det_6x6(&phi);

    println!("\n=== State Transition Matrix (Φ) ===");
    for row in &phi {
        println!(
            "  [{:12.6}, {:12.6}, {:12.6}, {:12.6}, {:12.6}, {:12.6}]",
            row[0], row[1], row[2], row[3], row[4], row[5]
        );
    }
    println!(
        "\n  det(Φ) = {:.12} (Liouville: should be 1.0 for conservative system)",
        det
    );

    // --- Perturbation test: STM prediction vs actual propagation ---
    let delta_x = 0.001; // 1 meter perturbation in x

    // STM prediction: δy_final ≈ Φ · δy_0  (only δy_0[0] = delta_x is nonzero)
    let mut predicted = [0.0f64; 6];
    for i in 0..6 {
        predicted[i] = phi[i][0] * delta_x;
    }

    // Actual: re-propagate with perturbed initial conditions
    let mut y0_pert = y0;
    y0_pert[0] += delta_x;
    let mut solver2 = Rkf78::new(Tolerances::new(1e-12, 1e-12));
    let (_, yf_pert) = solver2
        .integrate(&sys, 0.0, &y0_pert, period, 10.0)
        .expect("Perturbed integration failed");

    let mut actual = [0.0f64; 6];
    for i in 0..6 {
        actual[i] = yf_pert[i] - yf[i];
    }

    println!(
        "\n=== Perturbation test (δx = {} km = {} m) ===",
        delta_x,
        delta_x * 1000.0
    );
    println!(
        "  {:>10}  {:>14}  {:>14}  {:>10}",
        "Component", "STM predicted", "Actual", "Error"
    );
    let labels = ["δx", "δy", "δz", "δvx", "δvy", "δvz"];
    for i in 0..6 {
        let err = (predicted[i] - actual[i]).abs();
        println!(
            "  {:>10}  {:14.6e}  {:14.6e}  {:10.2e}",
            labels[i], predicted[i], actual[i], err
        );
    }
}
