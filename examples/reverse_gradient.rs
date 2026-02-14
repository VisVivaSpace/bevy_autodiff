//! Reverse-mode gradient computation example
//!
//! Demonstrates computing the gradient via a single backward pass over the
//! compiled graph, instead of building separate symbolic derivative subgraphs
//! for each input. This is O(1) in the number of inputs.
//!
//! Run with: cargo run --example reverse_gradient

use bevy_autodiff::AutoDiff;

fn main() {
    // =========================================================================
    // 1. Basic reverse-mode gradient
    // =========================================================================

    let mut ad = AutoDiff::new();

    let x = ad.var(1.0);
    let y = ad.var(2.0);

    // f(x, y) = x^2 + x*y + y^2
    let x2 = ad.square(x);
    let xy = ad.mul(x, y);
    let y2 = ad.square(y);
    let sum1 = ad.add(x2, xy);
    let f = ad.add(sum1, y2);

    // compile_primal: only the function value, no symbolic derivative graphs.
    // Reverse-mode gradient() handles the derivatives at eval time.
    let mut cg = ad.compile_primal(f, &[x, y]);

    cg.eval(&[1.0, 2.0]);
    let val = cg.value();
    // gradient() does a single backward pass — cost is independent of #inputs.
    let grad = cg.gradient().to_vec();
    println!("f(1, 2) = {} (expected 7)", val);
    println!("gradient = {:?} (expected [4.0, 5.0])", grad);
    // df/dx = 2x + y = 2 + 2 = 4  ✓
    // df/dy = x + 2y = 1 + 4 = 5  ✓

    // Re-evaluate at a different point without recompiling.
    cg.eval(&[3.0, -1.0]);
    let val = cg.value();
    let grad = cg.gradient().to_vec();
    println!("\nf(3, -1) = {} (expected 7)", val); // 9 + (-3) + 1 = 7
    println!("gradient = {:?} (expected [5.0, 1.0])", grad);
    // df/dx = 2x + y = 6 + (-1) = 5  ✓
    // df/dy = x + 2y = 3 + (-2) = 1  ✓

    // =========================================================================
    // 2. Rosenbrock function — a classic optimization test
    // =========================================================================

    println!("\n--- Rosenbrock gradient descent ---");

    let mut ad = AutoDiff::new();
    let x = ad.var(0.0);
    let y = ad.var(0.0);

    // f(x,y) = (1 - x)^2 + 100*(y - x^2)^2
    let one = ad.constant(1.0);
    let hundred = ad.constant(100.0);
    let one_minus_x = ad.sub(one, x);
    let term1 = ad.square(one_minus_x);
    let x_sq = ad.square(x);
    let y_minus_x_sq = ad.sub(y, x_sq);
    let term2_inner = ad.square(y_minus_x_sq);
    let term2 = ad.mul(hundred, term2_inner);
    let f = ad.add(term1, term2);

    // Compile once, evaluate many times.
    let mut cg = ad.compile_primal(f, &[x, y]);

    // Simple gradient descent
    let mut pos = [0.0_f64, 0.0_f64];
    let lr = 0.001;

    for step in 0..5001 {
        cg.eval(&pos);
        let val = cg.value();
        let grad = cg.gradient().to_vec();

        if step % 1000 == 0 {
            println!(
                "  step {:>5}: f = {:.6}, pos = [{:.4}, {:.4}], |grad| = {:.6}",
                step,
                val,
                pos[0],
                pos[1],
                (grad[0] * grad[0] + grad[1] * grad[1]).sqrt()
            );
        }

        pos[0] -= lr * grad[0];
        pos[1] -= lr * grad[1];
    }

    println!(
        "\n  Final: pos = [{:.6}, {:.6}] (minimum at [1, 1])",
        pos[0], pos[1]
    );

    // =========================================================================
    // 3. eval_gradient convenience method
    // =========================================================================

    println!("\n--- eval_gradient convenience ---");

    let mut ad = AutoDiff::new();
    let x = ad.var(0.0);
    let y = ad.var(0.0);
    let z = ad.var(0.0);

    // f(x, y, z) = x*y + y*z + z*x   (3 inputs, one backward pass)
    let xy = ad.mul(x, y);
    let yz = ad.mul(y, z);
    let zx = ad.mul(z, x);
    let s1 = ad.add(xy, yz);
    let f = ad.add(s1, zx);

    let mut cg = ad.compile_primal(f, &[x, y, z]);

    // eval_gradient does eval + gradient in one call
    let grad = cg.eval_gradient(&[2.0, 3.0, 5.0]).to_vec();
    let val = cg.value();
    println!("f(2, 3, 5) = {} (expected 31)", val); // 6 + 15 + 10 = 31
    println!("gradient = {:?}", grad);
    println!("expected = [8.0, 7.0, 5.0]");
    // df/dx = y + z = 8,  df/dy = x + z = 7,  df/dz = y + x = 5
}
