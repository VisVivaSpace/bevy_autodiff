//! Rosenbrock function example
//!
//! Demonstrates gradient computation for optimization.
//!
//! Run with: cargo run --example rosenbrock

use bevy_autodiff::AutoDiff;

fn main() {
    let mut ad = AutoDiff::new();

    let x = ad.var(0.0).unwrap();
    let y = ad.var(0.0).unwrap();

    // Build Rosenbrock function: (1-x)^2 + 100*(y-x^2)^2
    let one = ad.constant(1.0);
    let hundred = ad.constant(100.0);

    let a_minus_x = ad.sub(one, x);
    let term1 = ad.square(a_minus_x);

    let x_sq = ad.square(x);
    let y_minus_x_sq = ad.sub(y, x_sq);
    let diff_sq = ad.square(y_minus_x_sq);
    let term2 = ad.mul(hundred, diff_sq);

    let f = ad.add(term1, term2);

    // At (0, 0): f = 1, gradient = [-2, 0]
    println!("f(0, 0) = {} (should be 1)", ad.eval(f).unwrap());

    let grad = ad.gradient(f).unwrap();
    println!("gradient at (0, 0) = {:?} (should be [-2.0, 0.0])", grad);

    // At minimum (1,1): gradient = [0, 0]
    // gradient() uses CompiledGraph internally, so it evaluates correctly
    // at the current input values — no need to rebuild the graph.
    ad.set_input(x, 1.0).unwrap();
    ad.set_input(y, 1.0).unwrap();

    let grad2 = ad.gradient(f).unwrap();
    println!("\ngradient at (1, 1) = {:?} (should be [0.0, 0.0])", grad2);

    // Use compiled graph for fast repeated evaluation
    println!("\n--- CompiledGraph for Rosenbrock ---");
    let mut ad3 = AutoDiff::new();
    let xc = ad3.var(0.0).unwrap();
    let yc = ad3.var(0.0).unwrap();
    let onec = ad3.constant(1.0);
    let hc = ad3.constant(100.0);

    let amx = ad3.sub(onec, xc);
    let t1c = ad3.square(amx);
    let xsq = ad3.square(xc);
    let ymxs = ad3.sub(yc, xsq);
    let dsc = ad3.square(ymxs);
    let t2c = ad3.mul(hc, dsc);
    let fc = ad3.add(t1c, t2c);

    let mut cg = ad3.compile_order(fc, &[xc, yc], 1).unwrap();

    for &(xv, yv) in &[(0.0, 0.0), (1.0, 1.0), (0.5, 0.25)] {
        cg.eval(&[xv, yv]).unwrap();
        println!(
            "  f({}, {}) = {:.6}, grad = [{:.6}, {:.6}]",
            xv,
            yv,
            cg.value(),
            cg.partial(&[1, 0]).unwrap(),
            cg.partial(&[0, 1]).unwrap(),
        );
    }
}
