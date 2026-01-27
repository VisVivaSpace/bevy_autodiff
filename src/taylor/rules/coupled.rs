//! Coupled recurrence rules for trigonometric and hyperbolic functions.
//!
//! Some functions (sin/cos, sinh/cosh) satisfy coupled differential equations
//! and must be computed together for numerical stability. This module provides
//! a trait-based framework for such coupled recurrences.
//!
//! ## Theory
//!
//! For sin and cos:
//! - d/dt sin(u(t)) = cos(u(t)) · u'(t)
//! - d/dt cos(u(t)) = -sin(u(t)) · u'(t)
//!
//! The Taylor coefficient recurrences are:
//! - s_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·c_{k-j}
//! - c_k = -(1/k) Σⱼ₌₁ᵏ j·uⱼ·s_{k-j}

/// Trait for coupled function recurrences.
///
/// Implement this trait to add new coupled function pairs that can be
/// computed using the generic `coupled_taylor` driver.
pub trait CoupledRecurrence {
    /// Number of coupled outputs (e.g., 2 for sin/cos).
    const NUM_OUTPUTS: usize;

    /// Compute base values (k=0 coefficients) from the input base value.
    fn base_values(u0: f64) -> Vec<f64>;

    /// Compute k-th coefficients for all outputs.
    ///
    /// # Arguments
    /// - `k`: The coefficient index to compute (k >= 1)
    /// - `u`: Input Taylor coefficients [u₀, u₁, ..., u_k]
    /// - `prev`: Previous output coefficients [output0[..k], output1[..k], ...]
    ///
    /// # Returns
    /// A vector of length `NUM_OUTPUTS` containing the k-th coefficient for each output.
    fn recurrence_step(k: usize, u: &[f64], prev: &[Vec<f64>]) -> Vec<f64>;
}

/// Generic driver for coupled recurrences.
///
/// Computes Taylor coefficients for all coupled outputs up to the given order.
///
/// # Returns
/// A vector of `R::NUM_OUTPUTS` coefficient vectors.
pub fn coupled_taylor<R: CoupledRecurrence>(u: &[f64], order: usize) -> Vec<Vec<f64>> {
    let u0 = u.first().copied().unwrap_or(0.0);
    let base = R::base_values(u0);

    // Initialize output vectors
    let mut outputs: Vec<Vec<f64>> = base
        .into_iter()
        .map(|b| {
            let mut v = vec![0.0; order + 1];
            v[0] = b;
            v
        })
        .collect();

    // Compute higher-order coefficients
    for k in 1..=order {
        let new_coeffs = R::recurrence_step(k, u, &outputs);
        for (i, coeff) in new_coeffs.into_iter().enumerate() {
            if i < outputs.len() {
                outputs[i][k] = coeff;
            }
        }
    }

    outputs
}

/// Coupled recurrence for sin/cos.
///
/// Computes Taylor coefficients for both sin(u) and cos(u) simultaneously.
/// Returns [sin_coeffs, cos_coeffs].
pub struct SinCos;

impl CoupledRecurrence for SinCos {
    const NUM_OUTPUTS: usize = 2;

    fn base_values(u0: f64) -> Vec<f64> {
        vec![u0.sin(), u0.cos()]
    }

    fn recurrence_step(k: usize, u: &[f64], prev: &[Vec<f64>]) -> Vec<f64> {
        let s = &prev[0]; // sin coefficients
        let c = &prev[1]; // cos coefficients

        let mut s_k = 0.0;
        let mut c_k = 0.0;

        // s_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·c_{k-j}
        // c_k = -(1/k) Σⱼ₌₁ᵏ j·uⱼ·s_{k-j}
        for j in 1..=k.min(u.len() - 1) {
            let ju = (j as f64) * u[j];
            s_k += ju * c[k - j];
            c_k -= ju * s[k - j];
        }

        vec![s_k / (k as f64), c_k / (k as f64)]
    }
}

/// Coupled recurrence for sinh/cosh.
///
/// Computes Taylor coefficients for both sinh(u) and cosh(u) simultaneously.
/// Returns [sinh_coeffs, cosh_coeffs].
///
/// Note: Unlike sin/cos, the signs are the same (both positive).
pub struct SinhCosh;

impl CoupledRecurrence for SinhCosh {
    const NUM_OUTPUTS: usize = 2;

    fn base_values(u0: f64) -> Vec<f64> {
        vec![u0.sinh(), u0.cosh()]
    }

    fn recurrence_step(k: usize, u: &[f64], prev: &[Vec<f64>]) -> Vec<f64> {
        let sh = &prev[0]; // sinh coefficients
        let ch = &prev[1]; // cosh coefficients

        let mut sh_k = 0.0;
        let mut ch_k = 0.0;

        // sh_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·ch_{k-j}
        // ch_k = (1/k) Σⱼ₌₁ᵏ j·uⱼ·sh_{k-j}  (note: positive, unlike cos)
        for j in 1..=k.min(u.len() - 1) {
            let ju = (j as f64) * u[j];
            sh_k += ju * ch[k - j];
            ch_k += ju * sh[k - j];
        }

        vec![sh_k / (k as f64), ch_k / (k as f64)]
    }
}

/// Convenience function to compute sin Taylor coefficients.
#[inline]
pub fn sin_taylor(u: &[f64], order: usize) -> Vec<f64> {
    coupled_taylor::<SinCos>(u, order).into_iter().next().unwrap()
}

/// Convenience function to compute cos Taylor coefficients.
#[inline]
pub fn cos_taylor(u: &[f64], order: usize) -> Vec<f64> {
    coupled_taylor::<SinCos>(u, order).into_iter().nth(1).unwrap()
}

/// Convenience function to compute sinh Taylor coefficients.
#[inline]
pub fn sinh_taylor(u: &[f64], order: usize) -> Vec<f64> {
    coupled_taylor::<SinhCosh>(u, order)
        .into_iter()
        .next()
        .unwrap()
}

/// Convenience function to compute cosh Taylor coefficients.
#[inline]
pub fn cosh_taylor(u: &[f64], order: usize) -> Vec<f64> {
    coupled_taylor::<SinhCosh>(u, order)
        .into_iter()
        .nth(1)
        .unwrap()
}

/// Compute both sin and cos Taylor coefficients.
pub fn sin_cos_taylor(u: &[f64], order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut results = coupled_taylor::<SinCos>(u, order);
    let cos = results.pop().unwrap();
    let sin = results.pop().unwrap();
    (sin, cos)
}

/// Compute both sinh and cosh Taylor coefficients.
pub fn sinh_cosh_taylor(u: &[f64], order: usize) -> (Vec<f64>, Vec<f64>) {
    let mut results = coupled_taylor::<SinhCosh>(u, order);
    let cosh = results.pop().unwrap();
    let sinh = results.pop().unwrap();
    (sinh, cosh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taylor::polynomial::identity_taylor;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    // Helper to verify sin²(x) + cos²(x) = 1 for Taylor coefficients
    fn verify_pythagorean(sin: &[f64], cos: &[f64], tol: f64) {
        // The product sin(u)·sin(u) + cos(u)·cos(u) should have coefficient [1, 0, 0, ...]
        let order = sin.len().min(cos.len()) - 1;
        let sin2 = crate::taylor::polynomial::mul_taylor(sin, sin, order);
        let cos2 = crate::taylor::polynomial::mul_taylor(cos, cos, order);
        let sum = crate::taylor::polynomial::add_taylor(&sin2, &cos2, order);

        assert_relative_eq!(sum[0], 1.0, epsilon = tol);
        for k in 1..=order {
            assert_relative_eq!(sum[k], 0.0, epsilon = tol);
        }
    }

    // Helper to verify cosh²(x) - sinh²(x) = 1 for Taylor coefficients
    fn verify_hyperbolic_identity(sinh: &[f64], cosh: &[f64], tol: f64) {
        let order = sinh.len().min(cosh.len()) - 1;
        let sinh2 = crate::taylor::polynomial::mul_taylor(sinh, sinh, order);
        let cosh2 = crate::taylor::polynomial::mul_taylor(cosh, cosh, order);
        let diff = crate::taylor::polynomial::sub_taylor(&cosh2, &sinh2, order);

        assert_relative_eq!(diff[0], 1.0, epsilon = tol);
        for k in 1..=order {
            assert_relative_eq!(diff[k], 0.0, epsilon = tol);
        }
    }

    #[test]
    fn test_sin_cos_at_zero() {
        // u(t) = 0 + 1*t = t, so we're computing sin(t) and cos(t) around 0
        let u = identity_taylor(0.0, 1.0, 5);
        let (sin, cos) = sin_cos_taylor(&u, 5);

        // sin(t) at t=0: Taylor series is t - t³/6 + t⁵/120 - ...
        // Normalized coefficients: [0, 1, 0, -1/6, 0, 1/120]
        assert_relative_eq!(sin[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sin[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin[3], -1.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(sin[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sin[5], 1.0 / 120.0, epsilon = 1e-10);

        // cos(t) at t=0: Taylor series is 1 - t²/2 + t⁴/24 - ...
        // Normalized coefficients: [1, 0, -1/2, 0, 1/24, 0]
        assert_relative_eq!(cos[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cos[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos[2], -0.5, epsilon = 1e-10);
        assert_relative_eq!(cos[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos[4], 1.0 / 24.0, epsilon = 1e-10);
        assert_relative_eq!(cos[5], 0.0, epsilon = 1e-10);

        verify_pythagorean(&sin, &cos, 1e-10);
    }

    #[test]
    fn test_sin_cos_at_pi_4() {
        // u(t) = π/4 + t
        let u = identity_taylor(FRAC_PI_4, 1.0, 4);
        let (sin, cos) = sin_cos_taylor(&u, 4);

        // At t=0: sin(π/4) = cos(π/4) = √2/2
        let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert_relative_eq!(sin[0], sqrt2_2, epsilon = 1e-10);
        assert_relative_eq!(cos[0], sqrt2_2, epsilon = 1e-10);

        // First derivatives: d/dt sin(π/4+t) = cos(π/4+t), at t=0 = √2/2
        //                    d/dt cos(π/4+t) = -sin(π/4+t), at t=0 = -√2/2
        assert_relative_eq!(sin[1], sqrt2_2, epsilon = 1e-10);
        assert_relative_eq!(cos[1], -sqrt2_2, epsilon = 1e-10);

        verify_pythagorean(&sin, &cos, 1e-10);
    }

    #[test]
    fn test_sin_cos_at_pi_2() {
        // u(t) = π/2 + t
        let u = identity_taylor(FRAC_PI_2, 1.0, 4);
        let (sin, cos) = sin_cos_taylor(&u, 4);

        // sin(π/2) = 1, cos(π/2) = 0
        assert_relative_eq!(sin[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cos[0], 0.0, epsilon = 1e-10);

        // d/dt sin(π/2+t) = cos(π/2+t) = 0 at t=0
        // d/dt cos(π/2+t) = -sin(π/2+t) = -1 at t=0
        assert_relative_eq!(sin[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cos[1], -1.0, epsilon = 1e-10);

        verify_pythagorean(&sin, &cos, 1e-10);
    }

    #[test]
    fn test_sin_cos_pythagorean_various() {
        for x0 in [0.0, 0.5, 1.0, PI / 3.0, PI] {
            let u = identity_taylor(x0, 1.0, 6);
            let (sin, cos) = sin_cos_taylor(&u, 6);
            verify_pythagorean(&sin, &cos, 1e-9);
        }
    }

    #[test]
    fn test_sinh_cosh_at_zero() {
        // u(t) = 0 + t
        let u = identity_taylor(0.0, 1.0, 5);
        let (sinh, cosh) = sinh_cosh_taylor(&u, 5);

        // sinh(t) at t=0: Taylor series is t + t³/6 + t⁵/120 + ...
        assert_relative_eq!(sinh[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sinh[2], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh[3], 1.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(sinh[4], 0.0, epsilon = 1e-10);
        assert_relative_eq!(sinh[5], 1.0 / 120.0, epsilon = 1e-10);

        // cosh(t) at t=0: Taylor series is 1 + t²/2 + t⁴/24 + ...
        assert_relative_eq!(cosh[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(cosh[1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cosh[2], 0.5, epsilon = 1e-10);
        assert_relative_eq!(cosh[3], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cosh[4], 1.0 / 24.0, epsilon = 1e-10);
        assert_relative_eq!(cosh[5], 0.0, epsilon = 1e-10);

        verify_hyperbolic_identity(&sinh, &cosh, 1e-10);
    }

    #[test]
    fn test_sinh_cosh_identity_various() {
        for x0 in [0.0, 0.5, 1.0, 2.0] {
            let u = identity_taylor(x0, 1.0, 6);
            let (sinh, cosh) = sinh_cosh_taylor(&u, 6);
            verify_hyperbolic_identity(&sinh, &cosh, 1e-9);
        }
    }

    #[test]
    fn test_sin_derivative_is_cos() {
        // d/dx sin(x) = cos(x)
        // For sin(x) Taylor coeffs, coeff[1] * 1! should equal cos(x) value
        for x0 in [0.0, FRAC_PI_4, FRAC_PI_2, 1.0] {
            let u = identity_taylor(x0, 1.0, 2);
            let (sin, cos) = sin_cos_taylor(&u, 2);

            // First derivative of sin is cos
            let sin_deriv = sin[1] * 1.0; // coeff[1] * 1!
            let cos_value = cos[0];
            assert_relative_eq!(sin_deriv, cos_value, epsilon = 1e-10);

            // First derivative of cos is -sin
            let cos_deriv = cos[1] * 1.0;
            let neg_sin_value = -sin[0];
            assert_relative_eq!(cos_deriv, neg_sin_value, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sinh_derivative_is_cosh() {
        // d/dx sinh(x) = cosh(x)
        // d/dx cosh(x) = sinh(x)
        for x0 in [0.0, 0.5, 1.0, 2.0] {
            let u = identity_taylor(x0, 1.0, 2);
            let (sinh, cosh) = sinh_cosh_taylor(&u, 2);

            // First derivative of sinh is cosh
            let sinh_deriv = sinh[1] * 1.0;
            let cosh_value = cosh[0];
            assert_relative_eq!(sinh_deriv, cosh_value, epsilon = 1e-10);

            // First derivative of cosh is sinh (not negative!)
            let cosh_deriv = cosh[1] * 1.0;
            let sinh_value = sinh[0];
            assert_relative_eq!(cosh_deriv, sinh_value, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_convenience_functions() {
        let u = identity_taylor(0.5, 1.0, 3);

        let sin_only = sin_taylor(&u, 3);
        let cos_only = cos_taylor(&u, 3);
        let sinh_only = sinh_taylor(&u, 3);
        let cosh_only = cosh_taylor(&u, 3);

        let (sin_pair, cos_pair) = sin_cos_taylor(&u, 3);
        let (sinh_pair, cosh_pair) = sinh_cosh_taylor(&u, 3);

        assert_eq!(sin_only, sin_pair);
        assert_eq!(cos_only, cos_pair);
        assert_eq!(sinh_only, sinh_pair);
        assert_eq!(cosh_only, cosh_pair);
    }
}
