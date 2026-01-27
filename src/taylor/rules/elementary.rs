//! Taylor coefficient rules for elementary functions.
//!
//! All functions use recurrence relations that are O(n²) for order n.
//!
//! ## Exponential: y = e^u
//! y_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·y_{k-j}
//!
//! ## Logarithm: y = ln(u)
//! y_k = (u_k - (1/k) Σⱼ₌₁ᵏ⁻¹ j·yⱼ·u_{k-j}) / u_0
//!
//! ## Square Root: y = √u
//! y_k = (u_k - Σⱼ₌₁ᵏ⁻¹ yⱼ·y_{k-j}) / (2·y_0)

use smallvec::smallvec;

use crate::error::{TaylorError, TaylorResult};
use crate::taylor::polynomial::TaylorCoeffs;

/// Computes Taylor coefficients for exp(u).
///
/// Uses the recurrence: y_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·y_{k-j}
/// Base case: y_0 = exp(u_0)
///
/// # Example
/// ```
/// use bevy_autodiff::taylor::rules::exp_taylor;
/// use bevy_autodiff::taylor::polynomial::identity_taylor;
/// use approx::assert_relative_eq;
///
/// // exp(0 + t) = e^t = 1 + t + t²/2 + t³/6 + ...
/// let u = identity_taylor(0.0, 1.0, 4);
/// let exp_coeffs = exp_taylor(&u, 4);
///
/// assert_relative_eq!(exp_coeffs[0], 1.0, epsilon = 1e-10);  // e^0 = 1
/// assert_relative_eq!(exp_coeffs[1], 1.0, epsilon = 1e-10);  // 1!/1! = 1
/// assert_relative_eq!(exp_coeffs[2], 0.5, epsilon = 1e-10);  // 1/2!
/// assert_relative_eq!(exp_coeffs[3], 1.0/6.0, epsilon = 1e-10);  // 1/3!
/// ```
pub fn exp_taylor(u: &[f64], order: usize) -> Vec<f64> {
    let u0 = u.first().copied().unwrap_or(0.0);
    let mut y = vec![0.0; order + 1];

    // Base case
    y[0] = u0.exp();

    // Recurrence: y_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·y_{k-j}
    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            sum += (j as f64) * u[j] * y[k - j];
        }
        y[k] = sum / (k as f64);
    }

    y
}

/// Computes Taylor coefficients for ln(u).
///
/// Uses the recurrence: y_k = (u_k - (1/k) Σⱼ₌₁ᵏ⁻¹ j·yⱼ·u_{k-j}) / u_0
/// Base case: y_0 = ln(u_0)
///
/// # Errors
/// Returns [`TaylorError::LogDomainError`] if u_0 <= 0 (logarithm undefined for non-positive values).
///
/// # Example
/// ```
/// use bevy_autodiff::taylor::rules::ln_taylor;
/// use bevy_autodiff::taylor::polynomial::identity_taylor;
/// use approx::assert_relative_eq;
///
/// // ln(1 + t) = t - t²/2 + t³/3 - t⁴/4 + ...
/// let u = identity_taylor(1.0, 1.0, 4);
/// let ln_coeffs = ln_taylor(&u, 4).unwrap();
///
/// assert_relative_eq!(ln_coeffs[0], 0.0, epsilon = 1e-10);   // ln(1) = 0
/// assert_relative_eq!(ln_coeffs[1], 1.0, epsilon = 1e-10);   // 1
/// assert_relative_eq!(ln_coeffs[2], -0.5, epsilon = 1e-10);  // -1/2
/// assert_relative_eq!(ln_coeffs[3], 1.0/3.0, epsilon = 1e-10);  // 1/3
/// ```
pub fn ln_taylor(u: &[f64], order: usize) -> TaylorResult<TaylorCoeffs> {
    let u0 = u.first().copied().unwrap_or(0.0);
    if u0 <= 0.0 {
        return Err(TaylorError::LogDomainError(u0));
    }

    let mut y: TaylorCoeffs = smallvec![0.0; order + 1];

    // Base case
    y[0] = u0.ln();

    // Recurrence: y_k = (u_k - (1/k) Σⱼ₌₁ᵏ⁻¹ j·yⱼ·u_{k-j}) / u_0
    for k in 1..=order {
        let uk = u.get(k).copied().unwrap_or(0.0);
        let mut sum = 0.0;
        // Loop needs index j to access both y[j] and u[k-j]
        #[allow(clippy::needless_range_loop)]
        for j in 1..k {
            let ukj = u.get(k - j).copied().unwrap_or(0.0);
            sum += (j as f64) * y[j] * ukj;
        }
        y[k] = (uk - sum / (k as f64)) / u0;
    }

    Ok(y)
}

/// Computes Taylor coefficients for sqrt(u).
///
/// Uses the recurrence: y_k = (u_k - Σⱼ₌₁ᵏ⁻¹ yⱼ·y_{k-j}) / (2·y_0)
/// Base case: y_0 = sqrt(u_0)
///
/// # Errors
/// Returns [`TaylorError::SqrtDomainError`] if u_0 < 0 (square root undefined for negative values).
///
/// # Example
/// ```
/// use bevy_autodiff::taylor::rules::sqrt_taylor;
/// use bevy_autodiff::taylor::polynomial::identity_taylor;
/// use approx::assert_relative_eq;
///
/// // sqrt(4 + t) at t=0
/// let u = identity_taylor(4.0, 1.0, 3);
/// let sqrt_coeffs = sqrt_taylor(&u, 3).unwrap();
///
/// assert_relative_eq!(sqrt_coeffs[0], 2.0, epsilon = 1e-10);  // sqrt(4) = 2
/// assert_relative_eq!(sqrt_coeffs[1], 0.25, epsilon = 1e-10); // 1/(2*sqrt(4)) = 0.25
/// ```
pub fn sqrt_taylor(u: &[f64], order: usize) -> TaylorResult<TaylorCoeffs> {
    let u0 = u.first().copied().unwrap_or(0.0);
    if u0 < 0.0 {
        return Err(TaylorError::SqrtDomainError(u0));
    }

    if u0 == 0.0 {
        // Special case: sqrt(0 + ...) - only valid if all coefficients are 0
        let y: TaylorCoeffs = smallvec![0.0; order + 1];
        // For sqrt(t) at 0, we can't compute higher derivatives (infinite at 0)
        // Just return zeros
        return Ok(y);
    }

    let mut y: TaylorCoeffs = smallvec![0.0; order + 1];
    let y0 = u0.sqrt();
    y[0] = y0;

    // Recurrence: y_k = (u_k - Σⱼ₌₁ᵏ⁻¹ yⱼ·y_{k-j}) / (2·y_0)
    for k in 1..=order {
        let uk = u.get(k).copied().unwrap_or(0.0);
        let mut sum = 0.0;
        for j in 1..k {
            sum += y[j] * y[k - j];
        }
        y[k] = (uk - sum) / (2.0 * y0);
    }

    Ok(y)
}

/// Computes Taylor coefficients for u^p where p is a constant.
///
/// Uses the identity: u^p = exp(p * ln(u))
///
/// This handles fractional and negative powers.
pub fn pow_const_taylor(u: &[f64], p: f64, order: usize) -> Vec<f64> {
    // Special cases
    if p == 0.0 {
        let mut result = vec![0.0; order + 1];
        result[0] = 1.0;
        return result;
    }
    if p == 1.0 {
        let mut result = vec![0.0; order + 1];
        let copy_len = (order + 1).min(u.len());
        result[..copy_len].copy_from_slice(&u[..copy_len]);
        return result;
    }
    if p == 0.5 {
        return sqrt_taylor(u, order).expect("sqrt domain error in pow_const").to_vec();
    }
    if p == -1.0 {
        // 1/u - use division
        let one = super::constant_taylor(1.0, order);
        return super::div_taylor(&one, u, order).expect("division by zero in pow_const").to_vec();
    }

    // General case: u^p = exp(p * ln(u))
    let ln_u = ln_taylor(u, order).expect("ln domain error in pow_const");
    let p_ln_u = super::scale_taylor(&ln_u, p, order);
    exp_taylor(&p_ln_u, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taylor::polynomial::{identity_taylor, mul_taylor};
    use crate::util::factorial;
    use approx::assert_relative_eq;

    // ===================
    // Exp tests
    // ===================

    #[test]
    fn test_exp_at_zero() {
        // e^t = 1 + t + t²/2! + t³/3! + ...
        let u = identity_taylor(0.0, 1.0, 6);
        let y = exp_taylor(&u, 6);

        for k in 0..=6 {
            let expected = 1.0 / factorial(k);
            assert_relative_eq!(y[k], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_exp_at_one() {
        // e^(1+t) at t=0 should give e * (Taylor of e^t)
        let u = identity_taylor(1.0, 1.0, 4);
        let y = exp_taylor(&u, 4);

        let e = std::f64::consts::E;
        for k in 0..=4 {
            let expected = e / factorial(k);
            assert_relative_eq!(y[k], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_exp_derivative_is_exp() {
        // d/dx e^x = e^x
        for x0 in [0.0, 0.5, 1.0, 2.0] {
            let u = identity_taylor(x0, 1.0, 2);
            let y = exp_taylor(&u, 2);

            let value = y[0];
            let first_deriv = y[1] * 1.0; // coeff[1] * 1!

            assert_relative_eq!(first_deriv, value, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_exp_inverse_property() {
        // e^(-x) * e^x = 1
        let x0 = 0.5;
        let u = identity_taylor(x0, 1.0, 4);
        let neg_u = identity_taylor(-x0, -1.0, 4);

        let exp_u = exp_taylor(&u, 4);
        let exp_neg_u = exp_taylor(&neg_u, 4);
        let product = mul_taylor(&exp_u, &exp_neg_u, 4);

        assert_relative_eq!(product[0], 1.0, epsilon = 1e-10);
        for k in 1..=4 {
            assert_relative_eq!(product[k], 0.0, epsilon = 1e-9);
        }
    }

    // ===================
    // Ln tests
    // ===================

    #[test]
    fn test_ln_at_one() {
        // ln(1+t) = t - t²/2 + t³/3 - t⁴/4 + ...
        // Normalized: [0, 1, -1/2, 1/3, -1/4, ...]
        let u = identity_taylor(1.0, 1.0, 5);
        let y = ln_taylor(&u, 5).unwrap();

        assert_relative_eq!(y[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], -0.5, epsilon = 1e-10);
        assert_relative_eq!(y[3], 1.0 / 3.0, epsilon = 1e-10);
        assert_relative_eq!(y[4], -0.25, epsilon = 1e-10);
        assert_relative_eq!(y[5], 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_ln_at_e() {
        // ln(e + t) at t=0: value is 1, derivative is 1/e
        let e = std::f64::consts::E;
        let u = identity_taylor(e, 1.0, 3);
        let y = ln_taylor(&u, 3).unwrap();

        assert_relative_eq!(y[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 1.0 / e, epsilon = 1e-10);
    }

    #[test]
    fn test_ln_exp_inverse() {
        // ln(e^x) = x
        for x0 in [0.1, 0.5, 1.0, 2.0] {
            let u = identity_taylor(x0, 1.0, 4);
            let exp_u = exp_taylor(&u, 4);
            let ln_exp_u = ln_taylor(&exp_u, 4).unwrap();

            // Should recover x0 + t
            assert_relative_eq!(ln_exp_u[0], x0, epsilon = 1e-10);
            assert_relative_eq!(ln_exp_u[1], 1.0, epsilon = 1e-9);
            for k in 2..=4 {
                assert_relative_eq!(ln_exp_u[k], 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_ln_negative_returns_error() {
        use crate::error::TaylorError;
        let u = identity_taylor(-1.0, 1.0, 2);
        let result = ln_taylor(&u, 2);
        assert!(result.is_err());
        assert!(matches!(result, Err(TaylorError::LogDomainError(_))));
    }

    // ===================
    // Sqrt tests
    // ===================

    #[test]
    fn test_sqrt_at_one() {
        // sqrt(1+t) = 1 + t/2 - t²/8 + t³/16 - ...
        // Using binomial series: (1+t)^{1/2} = Σ C(1/2, k) t^k
        let u = identity_taylor(1.0, 1.0, 4);
        let y = sqrt_taylor(&u, 4).unwrap();

        assert_relative_eq!(y[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(y[2], -0.125, epsilon = 1e-10);
        assert_relative_eq!(y[3], 0.0625, epsilon = 1e-10);
    }

    #[test]
    fn test_sqrt_at_four() {
        // sqrt(4+t)
        // y_0 = 2
        // y_1 = 1/(2*2) = 0.25
        let u = identity_taylor(4.0, 1.0, 3);
        let y = sqrt_taylor(&u, 3).unwrap();

        assert_relative_eq!(y[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_sqrt_squared() {
        // sqrt(x)² = x
        for x0 in [1.0, 4.0, 9.0] {
            let u = identity_taylor(x0, 1.0, 4);
            let sqrt_u = sqrt_taylor(&u, 4).unwrap();
            let squared = mul_taylor(&sqrt_u, &sqrt_u, 4);

            // Should recover x0 + t
            assert_relative_eq!(squared[0], x0, epsilon = 1e-10);
            assert_relative_eq!(squared[1], 1.0, epsilon = 1e-9);
            for k in 2..=4 {
                assert_relative_eq!(squared[k], 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_sqrt_negative_returns_error() {
        use crate::error::TaylorError;
        let u = identity_taylor(-1.0, 1.0, 2);
        let result = sqrt_taylor(&u, 2);
        assert!(result.is_err());
        assert!(matches!(result, Err(TaylorError::SqrtDomainError(_))));
    }

    // ===================
    // pow_const tests
    // ===================

    #[test]
    fn test_pow_const_zero() {
        let u = identity_taylor(3.0, 1.0, 3);
        let y = pow_const_taylor(&u, 0.0, 3);

        assert_eq!(y[0], 1.0);
        for k in 1..=3 {
            assert_eq!(y[k], 0.0);
        }
    }

    #[test]
    fn test_pow_const_one() {
        let u = identity_taylor(3.0, 1.0, 3);
        let y = pow_const_taylor(&u, 1.0, 3);

        assert_eq!(y[0], 3.0);
        assert_eq!(y[1], 1.0);
        for k in 2..=3 {
            assert_eq!(y[k], 0.0);
        }
    }

    #[test]
    fn test_pow_const_half() {
        // x^0.5 = sqrt(x)
        let u = identity_taylor(4.0, 1.0, 3);
        let y_pow = pow_const_taylor(&u, 0.5, 3);
        let y_sqrt = sqrt_taylor(&u, 3).unwrap();

        for k in 0..=3 {
            assert_relative_eq!(y_pow[k], y_sqrt[k], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_pow_const_negative_one() {
        // x^(-1) = 1/x
        let u = identity_taylor(2.0, 1.0, 3);
        let y = pow_const_taylor(&u, -1.0, 3);

        // 1/(2+t) at t=0: value = 0.5, derivative = -0.25, ...
        assert_relative_eq!(y[0], 0.5, epsilon = 1e-10);
        assert_relative_eq!(y[1], -0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_pow_const_two() {
        // x^2
        let u = identity_taylor(3.0, 1.0, 3);
        let y = pow_const_taylor(&u, 2.0, 3);

        // (3+t)² = 9 + 6t + t²
        assert_relative_eq!(y[0], 9.0, epsilon = 1e-10);
        assert_relative_eq!(y[1], 6.0, epsilon = 1e-10);
        assert_relative_eq!(y[2], 1.0, epsilon = 1e-10);
    }
}
