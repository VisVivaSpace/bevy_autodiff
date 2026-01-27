//! Taylor coefficient rules for power functions.
//!
//! ## General Power: y = u^v
//! Uses the identity: u^v = exp(v · ln(u))
//!
//! This handles the case where both base and exponent are variables.
//! For constant exponents, use `pow_const_taylor` from elementary.rs
//! which is more efficient.

use super::elementary::{exp_taylor, ln_taylor, pow_const_taylor};
use super::{mul_taylor, scale_taylor};

/// Computes Taylor coefficients for u^v (general power).
///
/// Uses the identity: u^v = exp(v · ln(u))
///
/// # Requirements
/// - u₀ > 0 (logarithm of base must be defined)
///
/// # Arguments
/// - `u`: Taylor coefficients of the base
/// - `v`: Taylor coefficients of the exponent
/// - `order`: Maximum order to compute
///
/// # Example
/// ```
/// use bevy_autodiff::taylor::rules::power::pow_taylor;
/// use bevy_autodiff::taylor::polynomial::identity_taylor;
/// use approx::assert_relative_eq;
///
/// // Compute (2)^(3) = 8 (constant case)
/// let u = identity_taylor(2.0, 0.0, 2);  // u = 2 (constant)
/// let v = identity_taylor(3.0, 0.0, 2);  // v = 3 (constant)
/// let y = pow_taylor(&u, &v, 2);
/// assert_relative_eq!(y[0], 8.0, epsilon = 1e-10);
/// ```
pub fn pow_taylor(u: &[f64], v: &[f64], order: usize) -> Vec<f64> {
    let u0 = u.first().copied().unwrap_or(0.0);
    let v0 = v.first().copied().unwrap_or(0.0);

    // Special cases for efficiency
    if v0 == 0.0 && v.iter().skip(1).all(|&x| x == 0.0) {
        // v is constant 0: u^0 = 1
        let mut result = vec![0.0; order + 1];
        result[0] = 1.0;
        return result;
    }

    if u0 == 1.0 && u.iter().skip(1).all(|&x| x == 0.0) {
        // u is constant 1: 1^v = 1
        let mut result = vec![0.0; order + 1];
        result[0] = 1.0;
        return result;
    }

    // Check if v is a constant (all higher coefficients are zero)
    let v_is_constant = v.iter().skip(1).all(|&x| x.abs() < 1e-15);

    if v_is_constant {
        // Use the more efficient constant power formula
        return pow_const_taylor(u, v0, order);
    }

    // General case: u^v = exp(v * ln(u))
    let ln_u = ln_taylor(u, order).expect("ln domain error in pow_taylor");
    let v_ln_u = mul_taylor(v, &ln_u, order);
    exp_taylor(&v_ln_u, order)
}

/// Computes Taylor coefficients for x^n where n is a positive integer.
///
/// This is more efficient than the general formula for small positive integers
/// since it uses repeated multiplication instead of exp/ln.
pub fn pow_int_taylor(u: &[f64], n: i32, order: usize) -> Vec<f64> {
    if n == 0 {
        let mut result = vec![0.0; order + 1];
        result[0] = 1.0;
        return result;
    }

    if n < 0 {
        // x^(-n) = 1 / x^n
        let pos_pow = pow_int_taylor(u, -n, order);
        let one = super::constant_taylor(1.0, order);
        return super::div_taylor(&one, &pos_pow, order).expect("division by zero in pow_int").to_vec();
    }

    // For n=1, just return u
    if n == 1 {
        let mut result = vec![0.0; order + 1];
        let copy_len = (order + 1).min(u.len());
        result[..copy_len].copy_from_slice(&u[..copy_len]);
        return result;
    }

    // For small n, use repeated multiplication
    if n <= 8 {
        let mut result = u.to_vec();
        result.resize(order + 1, 0.0);

        for _ in 1..n {
            result = mul_taylor(&result, u, order).to_vec();
        }
        return result;
    }

    // For larger n, use the general formula
    pow_const_taylor(u, n as f64, order)
}

/// Computes d/dx(x^n) Taylor coefficients using the power rule.
///
/// d/dx(x^n) = n * x^(n-1)
///
/// This is useful for verification and for cases where we know we need
/// the derivative of a power.
pub fn pow_derivative_taylor(u: &[f64], n: i32, order: usize) -> Vec<f64> {
    if n == 0 {
        // d/dx(1) = 0
        return vec![0.0; order + 1];
    }

    // n * x^(n-1)
    let pow_nm1 = pow_int_taylor(u, n - 1, order);
    scale_taylor(&pow_nm1, n as f64, order).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taylor::polynomial::{identity_taylor, mul_taylor};
    use approx::assert_relative_eq;

    #[test]
    fn test_pow_constant_exponent_zero() {
        // u^0 = 1
        let u = identity_taylor(5.0, 1.0, 3);
        let v = identity_taylor(0.0, 0.0, 3);
        let y = pow_taylor(&u, &v, 3);

        assert_eq!(y[0], 1.0);
        for k in 1..=3 {
            assert_eq!(y[k], 0.0);
        }
    }

    #[test]
    fn test_pow_constant_base_one() {
        // 1^v = 1
        let u = identity_taylor(1.0, 0.0, 3);
        let v = identity_taylor(5.0, 1.0, 3);
        let y = pow_taylor(&u, &v, 3);

        assert_eq!(y[0], 1.0);
        for k in 1..=3 {
            assert_eq!(y[k], 0.0);
        }
    }

    #[test]
    fn test_pow_constant_both() {
        // 2^3 = 8
        let u = identity_taylor(2.0, 0.0, 3);
        let v = identity_taylor(3.0, 0.0, 3);
        let y = pow_taylor(&u, &v, 3);

        assert_relative_eq!(y[0], 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pow_square() {
        // x^2 where x varies
        // Should match mul(x, x)
        let x = identity_taylor(3.0, 1.0, 4);
        let two = identity_taylor(2.0, 0.0, 4);

        let y_pow = pow_taylor(&x, &two, 4);
        let y_mul = mul_taylor(&x, &x, 4);

        for k in 0..=4 {
            assert_relative_eq!(y_pow[k], y_mul[k], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_pow_sqrt_via_half() {
        // x^0.5 = sqrt(x)
        let x = identity_taylor(4.0, 1.0, 3);
        let half = identity_taylor(0.5, 0.0, 3);

        let y_pow = pow_taylor(&x, &half, 3);
        let y_sqrt = super::super::sqrt_taylor(&x, 3).unwrap();

        for k in 0..=3 {
            assert_relative_eq!(y_pow[k], y_sqrt[k], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_pow_variable_exponent() {
        // x^y where both vary
        // At (x, y) = (2, 3): f = 8
        // ∂f/∂x = y * x^(y-1) = 3 * 2^2 = 12
        // ∂f/∂y = x^y * ln(x) = 8 * ln(2) ≈ 5.545
        let x = identity_taylor(2.0, 1.0, 2);
        let y = identity_taylor(3.0, 1.0, 2);

        // For testing purposes, let's use direction (1, 0) and (0, 1) separately
        // Here we're testing with both directions = 1

        let result = pow_taylor(&x, &y, 2);

        // Value at (2, 3) = 8
        assert_relative_eq!(result[0], 8.0, epsilon = 1e-10);

        // First directional derivative in direction (1, 1):
        // ∂f/∂x + ∂f/∂y = 12 + 8*ln(2) ≈ 17.545
        let expected_deriv = 12.0 + 8.0 * 2.0_f64.ln();
        assert_relative_eq!(result[1], expected_deriv, epsilon = 1e-8);
    }

    #[test]
    fn test_pow_int_zero() {
        let u = identity_taylor(5.0, 1.0, 3);
        let y = pow_int_taylor(&u, 0, 3);

        assert_eq!(y[0], 1.0);
        for k in 1..=3 {
            assert_eq!(y[k], 0.0);
        }
    }

    #[test]
    fn test_pow_int_one() {
        let u = identity_taylor(5.0, 1.0, 3);
        let y = pow_int_taylor(&u, 1, 3);

        // Should be identical to u
        for k in 0..=3 {
            assert_eq!(y[k], u.get(k).copied().unwrap_or(0.0));
        }
    }

    #[test]
    fn test_pow_int_two() {
        // x^2 via integer power
        let u = identity_taylor(3.0, 1.0, 3);
        let y = pow_int_taylor(&u, 2, 3);

        // (3 + t)^2 = 9 + 6t + t^2
        assert_eq!(y[0], 9.0);
        assert_eq!(y[1], 6.0);
        assert_eq!(y[2], 1.0);
    }

    #[test]
    fn test_pow_int_three() {
        // x^3 via integer power
        let u = identity_taylor(2.0, 1.0, 4);
        let y = pow_int_taylor(&u, 3, 4);

        // (2 + t)^3 = 8 + 12t + 6t^2 + t^3
        assert_eq!(y[0], 8.0);
        assert_eq!(y[1], 12.0);
        assert_eq!(y[2], 6.0);
        assert_relative_eq!(y[3], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pow_int_negative() {
        // x^(-1) = 1/x
        let u = identity_taylor(2.0, 1.0, 3);
        let y = pow_int_taylor(&u, -1, 3);

        // 1/(2+t) at t=0
        assert_relative_eq!(y[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(y[1], -0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_pow_int_negative_two() {
        // x^(-2) = 1/x^2
        let u = identity_taylor(2.0, 1.0, 3);
        let y = pow_int_taylor(&u, -2, 3);

        // Value: 1/4 = 0.25
        assert_relative_eq!(y[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_pow_derivative() {
        // d/dx(x^3) = 3x^2
        let u = identity_taylor(2.0, 1.0, 3);
        let y = pow_derivative_taylor(&u, 3, 3);

        // 3 * (2+t)^2 = 3 * (4 + 4t + t^2) = 12 + 12t + 3t^2
        assert_eq!(y[0], 12.0);
        assert_eq!(y[1], 12.0);
        assert_eq!(y[2], 3.0);
    }
}
