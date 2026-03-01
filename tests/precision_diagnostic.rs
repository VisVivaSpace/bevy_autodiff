//! Regression and diagnostic tests for the `smart_sub` fix in `context.rs`.
//!
//! Run with: `cargo test --test precision_diagnostic -- --nocapture`
//!
//! **Bug (fixed):** `differentiate()` called `self.sub(b, one)` instead of
//! `self.smart_sub(b, one)` when applying the power rule.  `sub` created a
//! `Sub(constant(n), constant(1))` node that evaluates correctly at runtime
//! but is **not** marked `IsConstant`, so constant-folding shortcuts never
//! fired on it.  Over-differentiating a polynomial eventually built a
//! `Pow(x, -1)` sub-expression; at x = 0 that evaluates to NaN.
//! `smart_sub` folds `constant(n) − constant(1)` to a proper `constant(n-1)`,
//! keeping all subsequent constant checks correct and avoiding the NaN.

use bevy_autodiff::AutoDiff;

// ============================================================================
// 0a: Precision at multiple x values for powi(x, 5), orders 1–5
// ============================================================================

#[test]
fn diagnose_powi5_precision() {
    // f = x^5; exact derivatives: f'=5x^4, f''=20x^3, f'''=60x^2, f''''=120x, f'''''=120
    println!("\n=== diagnose_powi5_precision ===");
    for &x in &[2.0_f64, 1.5, 0.7, 0.1] {
        println!("\n--- powi(x, 5) at x = {} ---", x);
        let exact = [
            5.0 * x.powi(4),
            20.0 * x.powi(3),
            60.0 * x.powi(2),
            120.0 * x,
            120.0,
        ];
        for order in 1..=5_usize {
            let mut ad = AutoDiff::new();
            let xv = ad.var(x).unwrap();
            let f = ad.powi(xv, 5);
            let got = ad.derivative(f, xv, order).unwrap();
            let exp = exact[order - 1];
            println!(
                "  order {}: got={:.15e}  expected={:.15e}  abs_err={:.3e}",
                order,
                got,
                exp,
                (got - exp).abs()
            );
        }

        // Also compare powi vs powi_log side-by-side
        println!("  --- powi vs powi_log comparison ---");
        for order in 1..=5_usize {
            let exp = exact[order - 1];

            let mut ad_std = AutoDiff::new();
            let xv = ad_std.var(x).unwrap();
            let f = ad_std.powi(xv, 5);
            let got_std = ad_std.derivative(f, xv, order).unwrap();

            let mut ad_log = AutoDiff::new();
            let xv = ad_log.var(x).unwrap();
            let f = ad_log.powi_log(xv, 5);
            let got_log = ad_log.derivative(f, xv, order).unwrap();

            println!(
                "  order {}: powi_err={:.3e}  powi_log_err={:.3e}",
                order,
                (got_std - exp).abs(),
                (got_log - exp).abs()
            );
        }
    }
}

// ============================================================================
// 0b: Over-differentiation NaN check for powi(x, 3)
// ============================================================================

#[test]
fn diagnose_over_differentiation() {
    // d^4[x^3]/dx^4 and d^5[x^3]/dx^5 should both be 0 exactly.
    // Before the smart_sub fix, x = 0.0 produced NaN because the un-folded
    // Sub(3, 1) node was not recognised as a constant, so the chain eventually
    // reached Pow(x, -1) = 1/x and evaluated it at x = 0.
    // After the fix all values — including x = 0 — must be exactly 0.
    println!("\n=== diagnose_over_differentiation ===");
    println!("(d^4 and d^5 of x^3 should both be 0)\n");
    for &x in &[2.0_f64, 0.5, 0.0] {
        let d4 = {
            let mut ad = AutoDiff::new();
            let xv = ad.var(x).unwrap();
            let f = ad.powi(xv, 3);
            ad.derivative(f, xv, 4).unwrap()
        };
        let d5 = {
            let mut ad = AutoDiff::new();
            let xv = ad.var(x).unwrap();
            let f = ad.powi(xv, 3);
            ad.derivative(f, xv, 5).unwrap()
        };
        println!("x = {}: d4 = {}  d5 = {}", x, d4, d5);

        // Regression assertions: must be exactly 0, never NaN.
        assert_eq!(d4, 0.0, "d^4[x^3]/dx^4 at x={} should be 0 (was NaN before smart_sub fix)", x);
        assert_eq!(d5, 0.0, "d^5[x^3]/dx^5 at x={} should be 0 (was NaN before smart_sub fix)", x);

        // Also check powi_log variant
        let d4_log = {
            let mut ad = AutoDiff::new();
            let xv = ad.var(x).unwrap();
            let f = ad.powi_log(xv, 3);
            ad.derivative(f, xv, 4).unwrap()
        };
        let d5_log = {
            let mut ad = AutoDiff::new();
            let xv = ad.var(x).unwrap();
            let f = ad.powi_log(xv, 3);
            ad.derivative(f, xv, 5).unwrap()
        };
        println!(
            "x = {}: powi_log d4 = {}  powi_log d5 = {}",
            x, d4_log, d5_log
        );
    }
}
