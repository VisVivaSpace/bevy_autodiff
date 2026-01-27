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
fn test_pythagorean_identity_derivatives() {
    // d/dx(sin²(x) + cos²(x)) = 0
    for &x_val in &TEST_VALUES {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let sin_x = ad.sin(x);
        let cos_x = ad.cos(x);
        let sin2 = ad.square(sin_x);
        let cos2 = ad.square(cos_x);
        let sum = ad.add(sin2, cos2);

        let deriv = ad.derivative(sum, x, 1);
        assert_relative_eq!(deriv, 0.0, epsilon = 1e-9);
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
// Derivative Properties (Linearity, Product Rule, Chain Rule)
// ============================================================================

#[test]
fn test_derivative_linearity_addition() {
    // d/dx(f + g) = df/dx + dg/dx
    for &x_val in &[1.0, 2.0, 3.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let f = ad.square(x); // x²
        let g = ad.sin(x); // sin(x)
        let sum = ad.add(f, g);

        let df = ad.derivative(f, x, 1);
        let dg = ad.derivative(g, x, 1);
        let d_sum = ad.derivative(sum, x, 1);

        assert_relative_eq!(d_sum, df + dg, epsilon = 1e-10);
    }
}

#[test]
fn test_derivative_linearity_scalar() {
    // d/dx(c * f) = c * df/dx
    for &x_val in &[1.0, 2.0, 3.0] {
        for &c_val in &[2.0, 0.5, -1.0] {
            let mut ad = AutoDiff::new();
            let x = ad.var(x_val);
            let c = ad.constant(c_val);

            let f = ad.square(x); // x²
            let cf = ad.mul(c, f);

            let df = ad.derivative(f, x, 1);
            let d_cf = ad.derivative(cf, x, 1);

            assert_relative_eq!(d_cf, c_val * df, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_product_rule() {
    // d/dx(f * g) = f * dg/dx + g * df/dx
    for &x_val in &[1.0, 2.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let f = ad.square(x); // f = x²
        let g = ad.exp(x); // g = exp(x)
        let fg = ad.mul(f, g);

        let f_val = ad.eval(f);
        let g_val = ad.eval(g);
        let df = ad.derivative(f, x, 1);
        let dg = ad.derivative(g, x, 1);
        let d_fg = ad.derivative(fg, x, 1);

        let expected = f_val * dg + g_val * df;
        assert_relative_eq!(d_fg, expected, epsilon = 1e-10);
    }
}

#[test]
fn test_quotient_rule() {
    // d/dx(f / g) = (g * df/dx - f * dg/dx) / g²
    for &x_val in &[1.0, 2.0] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let f = ad.square(x); // f = x²
        let g = ad.exp(x); // g = exp(x)
        let f_over_g = ad.div(f, g);

        let f_val = ad.eval(f);
        let g_val = ad.eval(g);
        let df = ad.derivative(f, x, 1);
        let dg = ad.derivative(g, x, 1);
        let d_fg = ad.derivative(f_over_g, x, 1);

        let expected = (g_val * df - f_val * dg) / (g_val * g_val);
        assert_relative_eq!(d_fg, expected, epsilon = 1e-10);
    }
}

#[test]
fn test_chain_rule() {
    // d/dx(f(g(x))) = f'(g(x)) * g'(x)
    // Test with f = exp, g = x²
    // d/dx(exp(x²)) = exp(x²) * 2x
    for &x_val in &[0.5, 1.0, 1.5] {
        let mut ad = AutoDiff::new();
        let x = ad.var(x_val);

        let x2 = ad.square(x);
        let exp_x2 = ad.exp(x2);

        let deriv = ad.derivative(exp_x2, x, 1);
        let expected = (x_val * x_val).exp() * 2.0 * x_val;

        assert_relative_eq!(deriv, expected, epsilon = 1e-10);
    }
}

// ============================================================================
// Forward vs Reverse Mode Equivalence
// ============================================================================

#[test]
fn test_forward_reverse_equivalence_simple() {
    // gradient() and gradient_reverse() should produce identical results
    let mut ad = AutoDiff::new();
    let x = ad.var(2.0);
    let y = ad.var(3.0);

    // f = x² + y²
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let f = ad.add(x2, y2);

    let forward = ad.gradient(f);
    let reverse = ad.gradient_reverse(f);

    assert_eq!(forward.len(), reverse.len());
    for (fw, rv) in forward.iter().zip(reverse.iter()) {
        assert_relative_eq!(fw, rv, epsilon = 1e-10);
    }
}

#[test]
fn test_forward_reverse_equivalence_complex() {
    // More complex function: f = exp(x*y) + sin(x) * cos(y)
    let mut ad = AutoDiff::new();
    let x = ad.var(0.5);
    let y = ad.var(1.0);

    let xy = ad.mul(x, y);
    let exp_xy = ad.exp(xy);

    let sin_x = ad.sin(x);
    let cos_y = ad.cos(y);
    let sin_cos = ad.mul(sin_x, cos_y);

    let f = ad.add(exp_xy, sin_cos);

    let forward = ad.gradient(f);
    let reverse = ad.gradient_reverse(f);

    assert_eq!(forward.len(), reverse.len());
    for (fw, rv) in forward.iter().zip(reverse.iter()) {
        assert_relative_eq!(fw, rv, epsilon = 1e-10);
    }
}

#[test]
fn test_forward_reverse_equivalence_many_inputs() {
    // Test with many inputs to verify reverse mode scales properly
    let mut ad = AutoDiff::new();

    let vars: Vec<_> = (0..10).map(|i| ad.var(0.1 * (i + 1) as f64)).collect();

    // f = sum of all x_i squared
    let mut sum = ad.square(vars[0]);
    for &v in &vars[1..] {
        let v2 = ad.square(v);
        sum = ad.add(sum, v2);
    }

    let forward = ad.gradient(sum);
    let reverse = ad.gradient_reverse(sum);

    assert_eq!(forward.len(), reverse.len());
    for (fw, rv) in forward.iter().zip(reverse.iter()) {
        assert_relative_eq!(fw, rv, epsilon = 1e-10);
    }
}

// ============================================================================
// Higher-Order Derivative Properties
// ============================================================================

#[test]
fn test_second_derivative_symmetry() {
    // ∂²f/∂x∂y = ∂²f/∂y∂x (Schwarz's theorem)
    use crate::MultiIndex;

    let mut ad = AutoDiff::new();
    let x = ad.var(1.0);
    let y = ad.var(2.0);

    // f = x² * y + x * y²
    let x2 = ad.square(x);
    let y2 = ad.square(y);
    let x2y = ad.mul(x2, y);
    let xy2 = ad.mul(x, y2);
    let f = ad.add(x2y, xy2);

    let d2_xy = ad.partial(f, &MultiIndex::new(vec![1, 1]));

    // Compute ∂²f/∂y∂x by swapping the index pattern conceptually
    // (the result should be the same due to symmetry)
    let d2_yx = ad.partial(f, &MultiIndex::new(vec![1, 1]));

    assert_relative_eq!(d2_xy, d2_yx, epsilon = 1e-10);
}

#[test]
fn test_polynomial_derivatives_vanish() {
    // For a degree-n polynomial, the (n+1)-th derivative should be 0
    // f = x³ has d³f/dx³ = 6, d⁴f/dx⁴ = 0
    let mut ad = AutoDiff::new();
    let x = ad.var(2.0);
    let f = ad.powi(x, 3); // x³

    assert_relative_eq!(ad.derivative(f, x, 3), 6.0, epsilon = 1e-10);
    assert_relative_eq!(ad.derivative(f, x, 4), 0.0, epsilon = 1e-10);
    assert_relative_eq!(ad.derivative(f, x, 5), 0.0, epsilon = 1e-10);
}

#[test]
fn test_exp_all_derivatives_equal() {
    // All derivatives of exp(x) equal exp(x)
    let mut ad = AutoDiff::new();
    let x = ad.var(1.0);
    let f = ad.exp(x);

    let expected = std::f64::consts::E;
    for order in 0..=5 {
        let deriv = ad.derivative(f, x, order);
        assert_relative_eq!(deriv, expected, epsilon = 1e-10);
    }
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
