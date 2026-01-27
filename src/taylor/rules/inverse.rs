//! Taylor coefficient rules for inverse trigonometric and hyperbolic functions.
//!
//! ## Inverse Trig Derivatives
//! - d/dx asin(x) = 1/sqrt(1-x²)
//! - d/dx acos(x) = -1/sqrt(1-x²)
//! - d/dx atan(x) = 1/(1+x²)
//!
//! ## Inverse Hyperbolic Derivatives
//! - d/dx asinh(x) = 1/sqrt(x²+1)
//! - d/dx acosh(x) = 1/sqrt(x²-1)  (for x > 1)
//! - d/dx atanh(x) = 1/(1-x²)  (for |x| < 1)
//!
//! The Taylor coefficient recurrences are derived from y = f(u) where f is the inverse function.
//! Using the chain rule: y' = f'(u) * u'
//! For normalized Taylor coefficients: y_k = (1/k) * sum_{j=1}^{k} j * u_j * (f'(u))_{k-j}

// Allow index-based loops as they're clearer for these convolution-like mathematical formulas
#![allow(clippy::needless_range_loop)]

use super::elementary::sqrt_taylor;
use crate::taylor::polynomial::{
    add_taylor, constant_taylor, div_taylor, mul_taylor, sub_taylor, TaylorCoeffs,
};

/// Helper that unwraps div_taylor result (panics on division by zero).
#[inline]
fn div_taylor_unwrap(u: &[f64], v: &[f64], order: usize) -> TaylorCoeffs {
    div_taylor(u, v, order).expect("division by zero in inverse trig function")
}

/// Computes Taylor coefficients for tan(u).
///
/// tan(u) = sin(u) / cos(u)
pub fn tan_taylor(u: &[f64], order: usize) -> TaylorCoeffs {
    use super::coupled::{cos_taylor, sin_taylor};

    let sin_coeffs = sin_taylor(u, order);
    let cos_coeffs = cos_taylor(u, order);

    // tan = sin / cos
    div_taylor_unwrap(&sin_coeffs, &cos_coeffs, order)
}

/// Computes Taylor coefficients for tanh(u).
///
/// tanh(u) = sinh(u) / cosh(u)
pub fn tanh_taylor(u: &[f64], order: usize) -> TaylorCoeffs {
    use super::coupled::{cosh_taylor, sinh_taylor};

    let sinh_coeffs = sinh_taylor(u, order);
    let cosh_coeffs = cosh_taylor(u, order);

    // tanh = sinh / cosh
    div_taylor_unwrap(&sinh_coeffs, &cosh_coeffs, order)
}

/// Computes Taylor coefficients for asin(u).
///
/// Uses the recurrence: y_k = (1/k) * sum_{j=1}^{k} j * u_j * w_{k-j}
/// where w = 1/sqrt(1-u²), the derivative of asin.
pub fn asin_taylor(u: &[f64], order: usize) -> Option<TaylorCoeffs> {
    let u0 = u.first().copied().unwrap_or(0.0);

    // Check domain: |u0| < 1
    if u0.abs() >= 1.0 {
        return None;
    }

    // Compute 1 - u²
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_minus_u2 = sub_taylor(&one, &u_squared, order);

    // Compute sqrt(1 - u²)
    let sqrt_term = sqrt_taylor(&one_minus_u2, order).ok()?;

    // Compute w = 1 / sqrt(1 - u²) = derivative of asin
    let w = div_taylor_unwrap(&one, &sqrt_term, order);

    // Build y using the recurrence
    // y_0 = asin(u_0)
    // y_k = (1/k) * sum_{j=1}^{k} j * u_j * w_{k-j}
    let mut y = vec![0.0; order + 1];
    y[0] = u0.asin();

    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            let w_idx = k - j;
            if w_idx < w.len() {
                sum += (j as f64) * u[j] * w[w_idx];
            }
        }
        y[k] = sum / (k as f64);
    }

    Some(y.into())
}

/// Computes Taylor coefficients for acos(u).
///
/// acos(u) = π/2 - asin(u), so acos' = -asin'
pub fn acos_taylor(u: &[f64], order: usize) -> Option<TaylorCoeffs> {
    let u0 = u.first().copied().unwrap_or(0.0);

    // Check domain: |u0| < 1
    if u0.abs() >= 1.0 {
        return None;
    }

    // Compute 1 - u²
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_minus_u2 = sub_taylor(&one, &u_squared, order);

    // Compute sqrt(1 - u²)
    let sqrt_term = sqrt_taylor(&one_minus_u2, order).ok()?;

    // Compute w = -1 / sqrt(1 - u²) = derivative of acos
    let neg_one = constant_taylor(-1.0, order);
    let w = div_taylor_unwrap(&neg_one, &sqrt_term, order);

    // Build y using the recurrence
    let mut y = vec![0.0; order + 1];
    y[0] = u0.acos();

    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            let w_idx = k - j;
            if w_idx < w.len() {
                sum += (j as f64) * u[j] * w[w_idx];
            }
        }
        y[k] = sum / (k as f64);
    }

    Some(y.into())
}

/// Computes Taylor coefficients for atan(u).
///
/// Uses the recurrence with w = 1/(1+u²), the derivative of atan.
pub fn atan_taylor(u: &[f64], order: usize) -> TaylorCoeffs {
    let u0 = u.first().copied().unwrap_or(0.0);

    // Compute 1 + u²
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_plus_u2 = add_taylor(&one, &u_squared, order);

    // Compute w = 1 / (1 + u²) = derivative of atan
    let w = div_taylor_unwrap(&one, &one_plus_u2, order);

    // Build y using the recurrence
    let mut y = vec![0.0; order + 1];
    y[0] = u0.atan();

    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            let w_idx = k - j;
            if w_idx < w.len() {
                sum += (j as f64) * u[j] * w[w_idx];
            }
        }
        y[k] = sum / (k as f64);
    }

    y.into()
}

/// Computes Taylor coefficients for asinh(u).
///
/// Uses w = 1/sqrt(u²+1), the derivative of asinh.
pub fn asinh_taylor(u: &[f64], order: usize) -> TaylorCoeffs {
    let u0 = u.first().copied().unwrap_or(0.0);

    // Compute u² + 1
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let u2_plus_one = add_taylor(&u_squared, &one, order);

    // Compute sqrt(u² + 1)
    let sqrt_term = sqrt_taylor(&u2_plus_one, order).expect("asinh sqrt always valid");

    // Compute w = 1 / sqrt(u² + 1) = derivative of asinh
    let w = div_taylor_unwrap(&one, &sqrt_term, order);

    // Build y using the recurrence
    let mut y = vec![0.0; order + 1];
    y[0] = u0.asinh();

    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            let w_idx = k - j;
            if w_idx < w.len() {
                sum += (j as f64) * u[j] * w[w_idx];
            }
        }
        y[k] = sum / (k as f64);
    }

    y.into()
}

/// Computes Taylor coefficients for acosh(u).
///
/// Uses w = 1/sqrt(u²-1), the derivative of acosh.
/// Requires u > 1.
pub fn acosh_taylor(u: &[f64], order: usize) -> Option<TaylorCoeffs> {
    let u0 = u.first().copied().unwrap_or(0.0);

    // Check domain: u0 > 1
    if u0 <= 1.0 {
        return None;
    }

    // Compute u² - 1
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let u2_minus_one = sub_taylor(&u_squared, &one, order);

    // Compute sqrt(u² - 1)
    let sqrt_term = sqrt_taylor(&u2_minus_one, order).ok()?;

    // Compute w = 1 / sqrt(u² - 1) = derivative of acosh
    let w = div_taylor_unwrap(&one, &sqrt_term, order);

    // Build y using the recurrence
    let mut y = vec![0.0; order + 1];
    y[0] = u0.acosh();

    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            let w_idx = k - j;
            if w_idx < w.len() {
                sum += (j as f64) * u[j] * w[w_idx];
            }
        }
        y[k] = sum / (k as f64);
    }

    Some(y.into())
}

/// Computes Taylor coefficients for atanh(u).
///
/// Uses w = 1/(1-u²), the derivative of atanh.
/// Requires |u| < 1.
pub fn atanh_taylor(u: &[f64], order: usize) -> Option<TaylorCoeffs> {
    let u0 = u.first().copied().unwrap_or(0.0);

    // Check domain: |u0| < 1
    if u0.abs() >= 1.0 {
        return None;
    }

    // Compute 1 - u²
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_minus_u2 = sub_taylor(&one, &u_squared, order);

    // Compute w = 1 / (1 - u²) = derivative of atanh
    let w = div_taylor_unwrap(&one, &one_minus_u2, order);

    // Build y using the recurrence
    let mut y = vec![0.0; order + 1];
    y[0] = u0.atanh();

    for k in 1..=order {
        let mut sum = 0.0;
        for j in 1..=k.min(u.len() - 1) {
            let w_idx = k - j;
            if w_idx < w.len() {
                sum += (j as f64) * u[j] * w[w_idx];
            }
        }
        y[k] = sum / (k as f64);
    }

    Some(y.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::taylor::polynomial::identity_taylor;
    use approx::assert_relative_eq;

    #[test]
    fn test_tan_at_zero() {
        // tan(0) = 0, tan'(0) = 1
        let u = identity_taylor(0.0, 1.0, 3);
        let tan = tan_taylor(&u, 3);

        assert_relative_eq!(tan[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tan[1], 1.0, epsilon = 1e-10); // tan'(0) = sec²(0) = 1
    }

    #[test]
    fn test_tanh_at_zero() {
        // tanh(0) = 0, tanh'(0) = 1
        let u = identity_taylor(0.0, 1.0, 3);
        let tanh = tanh_taylor(&u, 3);

        assert_relative_eq!(tanh[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(tanh[1], 1.0, epsilon = 1e-10); // tanh'(0) = sech²(0) = 1
    }

    #[test]
    fn test_asin_at_zero() {
        // asin(0) = 0, asin'(0) = 1
        let u = identity_taylor(0.0, 1.0, 3);
        let asin = asin_taylor(&u, 3).unwrap();

        assert_relative_eq!(asin[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(asin[1], 1.0, epsilon = 1e-10); // asin'(0) = 1/sqrt(1-0) = 1
    }

    #[test]
    fn test_acos_at_zero() {
        // acos(0) = π/2, acos'(0) = -1
        let u = identity_taylor(0.0, 1.0, 3);
        let acos = acos_taylor(&u, 3).unwrap();

        assert_relative_eq!(acos[0], std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
        assert_relative_eq!(acos[1], -1.0, epsilon = 1e-10); // acos'(0) = -1
    }

    #[test]
    fn test_atan_at_zero() {
        // atan(0) = 0, atan'(0) = 1
        let u = identity_taylor(0.0, 1.0, 3);
        let atan = atan_taylor(&u, 3);

        assert_relative_eq!(atan[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(atan[1], 1.0, epsilon = 1e-10); // atan'(0) = 1/(1+0) = 1
    }

    #[test]
    fn test_atan_at_one() {
        // atan(1) = π/4, atan'(1) = 1/2
        let u = identity_taylor(1.0, 1.0, 3);
        let atan = atan_taylor(&u, 3);

        assert_relative_eq!(atan[0], std::f64::consts::FRAC_PI_4, epsilon = 1e-10);
        assert_relative_eq!(atan[1], 0.5, epsilon = 1e-10); // atan'(1) = 1/(1+1) = 0.5
    }

    #[test]
    fn test_asinh_at_zero() {
        // asinh(0) = 0, asinh'(0) = 1
        let u = identity_taylor(0.0, 1.0, 3);
        let asinh = asinh_taylor(&u, 3);

        assert_relative_eq!(asinh[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(asinh[1], 1.0, epsilon = 1e-10); // asinh'(0) = 1/sqrt(1) = 1
    }

    #[test]
    fn test_acosh_at_two() {
        // acosh(2) = ln(2 + sqrt(3)), acosh'(2) = 1/sqrt(3)
        let u = identity_taylor(2.0, 1.0, 3);
        let acosh = acosh_taylor(&u, 3).unwrap();

        assert_relative_eq!(acosh[0], 2.0_f64.acosh(), epsilon = 1e-10);
        assert_relative_eq!(acosh[1], 1.0 / 3.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_atanh_at_zero() {
        // atanh(0) = 0, atanh'(0) = 1
        let u = identity_taylor(0.0, 1.0, 3);
        let atanh = atanh_taylor(&u, 3).unwrap();

        assert_relative_eq!(atanh[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(atanh[1], 1.0, epsilon = 1e-10); // atanh'(0) = 1/(1-0) = 1
    }

    #[test]
    fn test_asin_acos_sum() {
        // asin(x) + acos(x) = π/2 for all x in domain
        let u = identity_taylor(0.5, 1.0, 3);
        let asin = asin_taylor(&u, 3).unwrap();
        let acos = acos_taylor(&u, 3).unwrap();

        // Sum of values should be π/2
        assert_relative_eq!(asin[0] + acos[0], std::f64::consts::FRAC_PI_2, epsilon = 1e-10);

        // Sum of all higher coefficients should be 0
        for k in 1..=3 {
            assert_relative_eq!(asin[k] + acos[k], 0.0, epsilon = 1e-10);
        }
    }
}
