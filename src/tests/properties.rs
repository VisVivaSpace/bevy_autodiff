//! Property-based tests following functional programming principles.
//!
//! These tests verify algebraic laws, mathematical identities, and composition
//! properties that should hold for any correct implementation.

use crate::AutoDiff;
use approx::assert_relative_eq;

/// Test values covering various numerical ranges
const TEST_VALUES: [f64; 7] = [0.0, 0.5, 1.0, 2.0, -1.0, 0.1, 3.14159];

/// Positive test values (for ln, sqrt, etc.)
const POSITIVE_VALUES: [f64; 5] = [0.1, 0.5, 1.0, 2.0, 10.0];

// ============================================================================
// Algebraic Properties of Operations
// ============================================================================

#[test]
fn test_addition_commutativity() {
    // a + b = b + a
    for &a_val in &TEST_VALUES {
        for &b_val in &TEST_VALUES {
            let mut ad = AutoDiff::new();
            let a = ad.var(a_val);
            let b = ad.var(b_val);

            let ab = ad.add(a, b);
            let ba = ad.add(b, a);

            assert_relative_eq!(ad.eval(ab), ad.eval(ba), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_multiplication_commutativity() {
    // a * b = b * a
    for &a_val in &TEST_VALUES {
        for &b_val in &TEST_VALUES {
            let mut ad = AutoDiff::new();
            let a = ad.var(a_val);
            let b = ad.var(b_val);

            let ab = ad.mul(a, b);
            let ba = ad.mul(b, a);

            assert_relative_eq!(ad.eval(ab), ad.eval(ba), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_addition_associativity() {
    // (a + b) + c = a + (b + c)
    for &a_val in &[1.0, 2.0, -1.0] {
        for &b_val in &[0.5, 1.5, -0.5] {
            for &c_val in &[0.1, 1.0, -0.1] {
                let mut ad = AutoDiff::new();
                let a = ad.var(a_val);
                let b = ad.var(b_val);
                let c = ad.var(c_val);

                let ab = ad.add(a, b);
                let left = ad.add(ab, c); // (a + b) + c

                let bc = ad.add(b, c);
                let right = ad.add(a, bc); // a + (b + c)

                assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_multiplication_associativity() {
    // (a * b) * c = a * (b * c)
    for &a_val in &[1.0, 2.0, 0.5] {
        for &b_val in &[0.5, 1.5, 2.0] {
            for &c_val in &[0.1, 1.0, 3.0] {
                let mut ad = AutoDiff::new();
                let a = ad.var(a_val);
                let b = ad.var(b_val);
                let c = ad.var(c_val);

                let ab = ad.mul(a, b);
                let left = ad.mul(ab, c);

                let bc = ad.mul(b, c);
                let right = ad.mul(a, bc);

                assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_distributivity() {
    // a * (b + c) = a*b + a*c
    for &a_val in &[1.0, 2.0, -1.0] {
        for &b_val in &[0.5, 1.5] {
            for &c_val in &[0.1, 1.0] {
                let mut ad = AutoDiff::new();
                let a = ad.var(a_val);
                let b = ad.var(b_val);
                let c = ad.var(c_val);

                let bc = ad.add(b, c);
                let left = ad.mul(a, bc); // a * (b + c)

                let ab = ad.mul(a, b);
                let ac = ad.mul(a, c);
                let right = ad.add(ab, ac); // a*b + a*c

                assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_additive_identity() {
    // a + 0 = a
    for &a_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let a = ad.var(a_val);
        let zero = ad.constant(0.0);

        let result = ad.add(a, zero);
        assert_relative_eq!(ad.eval(result), a_val, epsilon = 1e-10);
    }
}

#[test]
fn test_multiplicative_identity() {
    // a * 1 = a
    for &a_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let a = ad.var(a_val);
        let one = ad.constant(1.0);

        let result = ad.mul(a, one);
        assert_relative_eq!(ad.eval(result), a_val, epsilon = 1e-10);
    }
}

#[test]
fn test_additive_inverse() {
    // a + (-a) = 0
    for &a_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let a = ad.var(a_val);
        let neg_a = ad.neg(a);

        let result = ad.add(a, neg_a);
        assert_relative_eq!(ad.eval(result), 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_multiplicative_inverse() {
    // a * (1/a) = 1 for a != 0
    for &a_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let a = ad.var(a_val);
        let one = ad.constant(1.0);
        let inv_a = ad.div(one, a);

        let result = ad.mul(a, inv_a);
        assert_relative_eq!(ad.eval(result), 1.0, epsilon = 1e-10);
    }
}

// ============================================================================
// Trigonometric Identities
// ============================================================================

#[test]
fn test_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1
    for &x_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let sin_x = ad.sin(x);
        let cos_x = ad.cos(x);
        let sin2 = ad.square(sin_x);
        let cos2 = ad.square(cos_x);
        let sum = ad.add(sin2, cos2);

        assert_relative_eq!(ad.eval(sum), 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_hyperbolic_identity() {
    // cosh²(x) - sinh²(x) = 1
    for &x_val in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let sinh_x = ad.sinh(x);
        let cosh_x = ad.cosh(x);
        let sinh2 = ad.square(sinh_x);
        let cosh2 = ad.square(cosh_x);
        let diff = ad.sub(cosh2, sinh2);

        assert_relative_eq!(ad.eval(diff), 1.0, epsilon = 1e-10);
    }
}

// ============================================================================
// Exponential and Logarithmic Identities
// ============================================================================

#[test]
fn test_exp_ln_inverse() {
    // exp(ln(x)) = x for x > 0
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let ln_x = ad.ln(x);
        let result = ad.exp(ln_x);

        assert_relative_eq!(ad.eval(result), x_val, epsilon = 1e-10);
    }
}

#[test]
fn test_ln_exp_inverse() {
    // ln(exp(x)) = x
    for &x_val in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let exp_x = ad.exp(x);
        let result = ad.ln(exp_x);

        assert_relative_eq!(ad.eval(result), x_val, epsilon = 1e-10);
    }
}

#[test]
fn test_exp_addition_law() {
    // exp(a + b) = exp(a) * exp(b)
    for &a_val in &[-1.0, 0.0, 0.5, 1.0] {
        for &b_val in &[-0.5, 0.0, 0.5, 1.0] {
            let mut ad = AutoDiff::new();
            let a = ad.var(a_val);
            let b = ad.var(b_val);

            let sum = ad.add(a, b);
            let left = ad.exp(sum); // exp(a + b)

            let exp_a = ad.exp(a);
            let exp_b = ad.exp(b);
            let right = ad.mul(exp_a, exp_b); // exp(a) * exp(b)

            assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_ln_multiplication_law() {
    // ln(a * b) = ln(a) + ln(b) for a, b > 0
    for &a_val in &POSITIVE_VALUES {
        for &b_val in &POSITIVE_VALUES {
            let mut ad = AutoDiff::new();
            let a = ad.var(a_val);
            let b = ad.var(b_val);

            let prod = ad.mul(a, b);
            let left = ad.ln(prod); // ln(a * b)

            let ln_a = ad.ln(a);
            let ln_b = ad.ln(b);
            let right = ad.add(ln_a, ln_b); // ln(a) + ln(b)

            assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-10);
        }
    }
}

// ============================================================================
// Power Function Properties
// ============================================================================

#[test]
fn test_sqrt_square_inverse() {
    // sqrt(x²) = |x| (we test for positive x)
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let x2 = ad.square(x);
        let result = ad.sqrt(x2);

        assert_relative_eq!(ad.eval(result), x_val, epsilon = 1e-10);
    }
}

#[test]
fn test_square_sqrt_inverse() {
    // (sqrt(x))² = x for x >= 0
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let sqrt_x = ad.sqrt(x);
        let result = ad.square(sqrt_x);

        assert_relative_eq!(ad.eval(result), x_val, epsilon = 1e-10);
    }
}

#[test]
fn test_power_multiplication_law() {
    // x^a * x^b = x^(a+b)
    for &x_val in &[2.0, 3.0] {
        for &a_val in &[0.5, 1.0, 2.0] {
            for &b_val in &[0.5, 1.0, 1.5] {
                let mut ad = AutoDiff::new();
                let x = ad.var(x_val);

                let xa = ad.powf(x, a_val);
                let xb = ad.powf(x, b_val);
                let left = ad.mul(xa, xb); // x^a * x^b

                let right = ad.powf(x, a_val + b_val); // x^(a+b)

                assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-9);
            }
        }
    }
}

#[test]
fn test_power_power_law() {
    // (x^a)^b = x^(a*b)
    for &x_val in &[2.0, 3.0] {
        for &a_val in &[0.5, 1.0, 2.0] {
            for &b_val in &[0.5, 1.0, 2.0] {
                let mut ad = AutoDiff::new();
                let x = ad.var(x_val);

                let xa = ad.powf(x, a_val);
                let left = ad.powf(xa, b_val); // (x^a)^b

                let right = ad.powf(x, a_val * b_val); // x^(a*b)

                assert_relative_eq!(ad.eval(left), ad.eval(right), epsilon = 1e-9);
            }
        }
    }
}

// ============================================================================
// Derivative Properties
// ============================================================================

#[test]
fn test_derivative_of_sum_is_sum_of_derivatives() {
    // d(f+g)/dx = df/dx + dg/dx
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let f = ad.sin(x);
        let g = ad.exp(x);
        let fg_sum = ad.add(f, g);

        // d(f+g)/dx
        let d_sum = ad.differentiate(fg_sum, x);

        // df/dx + dg/dx
        let df = ad.differentiate(f, x);
        let dg = ad.differentiate(g, x);
        let sum_d = ad.add(df, dg);

        assert_relative_eq!(ad.eval(d_sum), ad.eval(sum_d), epsilon = 1e-10);
    }
}

#[test]
fn test_derivative_of_product_is_product_rule() {
    // d(fg)/dx = f'g + fg'
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let f = ad.sin(x);
        let g = ad.exp(x);
        let fg = ad.mul(f, g);

        let d_fg = ad.differentiate(fg, x);

        let df = ad.differentiate(f, x);
        let dg = ad.differentiate(g, x);
        let df_g = ad.mul(df, g);
        let f_dg = ad.mul(f, dg);
        let product_rule = ad.add(df_g, f_dg);

        assert_relative_eq!(ad.eval(d_fg), ad.eval(product_rule), epsilon = 1e-10);
    }
}

#[test]
fn test_derivative_of_constant_is_zero() {
    for &c_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(1.0);
        let c = ad.constant(c_val);
        let dc = ad.differentiate(c, x);
        assert_relative_eq!(ad.eval(dc), 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_derivative_linearity() {
    // d(a*f + b*g)/dx = a*df/dx + b*dg/dx
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let a = ad.constant(3.0);
        let b = ad.constant(-2.0);

        let f = ad.sin(x);
        let g = ad.ln(x);

        let af = ad.mul(a, f);
        let bg = ad.mul(b, g);
        let combo = ad.add(af, bg);
        let d_combo = ad.differentiate(combo, x);

        let df = ad.differentiate(f, x);
        let dg = ad.differentiate(g, x);
        let a_df = ad.mul(a, df);
        let b_dg = ad.mul(b, dg);
        let expected = ad.add(a_df, b_dg);

        assert_relative_eq!(ad.eval(d_combo), ad.eval(expected), epsilon = 1e-10);
    }
}

#[test]
fn test_pythagorean_identity_derivative() {
    // d/dx(sin²(x) + cos²(x)) = 0
    for &x_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let sin_x = ad.sin(x);
        let cos_x = ad.cos(x);
        let sin2 = ad.square(sin_x);
        let cos2 = ad.square(cos_x);
        let f = ad.add(sin2, cos2);

        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_mixed_partial_symmetry() {
    // d²f/dxdy = d²f/dydx for f = sin(x*y)
    for &x_val in &[0.5, 1.0, 2.0] {
        for &y_val in &[0.5, 1.0, 1.5] {
            let mut ad = AutoDiff::new();
            let x = ad.var(x_val);
            let y = ad.var(y_val);

            let xy = ad.mul(x, y);
            let f = ad.sin(xy);

            let dfdx = ad.differentiate(f, x);
            let d2fdxdy = ad.differentiate(dfdx, y);

            let dfdy = ad.differentiate(f, y);
            let d2fdydx = ad.differentiate(dfdy, x);

            assert_relative_eq!(ad.eval(d2fdxdy), ad.eval(d2fdydx), epsilon = 1e-10);
        }
    }
}

#[test]
fn test_chain_rule_exp_ln_inverse() {
    // d/dx(exp(ln(x))) should equal d/dx(x) = 1 for x > 0
    for &x_val in &POSITIVE_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);
        let ln_x = ad.ln(x);
        let f = ad.exp(ln_x);
        let df = ad.differentiate(f, x);
        assert_relative_eq!(ad.eval(df), 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_higher_order_derivative_polynomial() {
    // f = x^5, f'=5x^4, f''=20x^3, f'''=60x^2, f''''=120x, f'''''=120
    let mut ad = AutoDiff::new();
    let x = ad.var(2.0);
    let f = ad.powi(x, 5);

    assert_relative_eq!(ad.derivative(f, x, 1), 80.0, epsilon = 1e-8); // 5*16
    assert_relative_eq!(ad.derivative(f, x, 2), 160.0, epsilon = 1e-7); // 20*8
    assert_relative_eq!(ad.derivative(f, x, 3), 240.0, epsilon = 1e-6); // 60*4
    assert_relative_eq!(ad.derivative(f, x, 4), 240.0, epsilon = 1e-5); // 120*2
    assert_relative_eq!(ad.derivative(f, x, 5), 120.0, epsilon = 1e-4); // 120
}

// ============================================================================
// Edge Cases and Special Values
// ============================================================================

#[test]
fn test_zero_handling() {
    let mut ad = AutoDiff::new();
    let zero = ad.var(0.0);

    // sin(0) = 0
    let sin_zero = ad.sin(zero);
    assert_relative_eq!(ad.eval(sin_zero), 0.0, epsilon = 1e-10);

    // cos(0) = 1
    let cos_zero = ad.cos(zero);
    assert_relative_eq!(ad.eval(cos_zero), 1.0, epsilon = 1e-10);

    // exp(0) = 1
    let exp_zero = ad.exp(zero);
    assert_relative_eq!(ad.eval(exp_zero), 1.0, epsilon = 1e-10);

    // 0^2 = 0
    let zero_squared = ad.square(zero);
    assert_relative_eq!(ad.eval(zero_squared), 0.0, epsilon = 1e-10);
}

#[test]
fn test_one_handling() {
    let mut ad = AutoDiff::new();
    let one = ad.var(1.0);

    // ln(1) = 0
    let ln_one = ad.ln(one);
    assert_relative_eq!(ad.eval(ln_one), 0.0, epsilon = 1e-10);

    // sqrt(1) = 1
    let sqrt_one = ad.sqrt(one);
    assert_relative_eq!(ad.eval(sqrt_one), 1.0, epsilon = 1e-10);

    // 1^n = 1 for any n
    for n in 0..=5 {
        let one_n = ad.powi(one, n);
        assert_relative_eq!(ad.eval(one_n), 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_negative_values() {
    let mut ad = AutoDiff::new();
    let neg = ad.var(-2.0);

    // (-2)² = 4
    let neg_squared = ad.square(neg);
    assert_relative_eq!(ad.eval(neg_squared), 4.0, epsilon = 1e-10);

    // exp(-2) = 1/e²
    let exp_neg = ad.exp(neg);
    assert_relative_eq!(ad.eval(exp_neg), (-2.0_f64).exp(), epsilon = 1e-10);

    // sin(-x) = -sin(x)
    let pos = ad.var(2.0);
    let sin_neg = ad.sin(neg);
    let sin_pos = ad.sin(pos);
    assert_relative_eq!(ad.eval(sin_neg), -ad.eval(sin_pos), epsilon = 1e-10);
}
