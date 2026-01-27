//! Pure functions for Taylor coefficient arithmetic.
//!
//! All functions take normalized Taylor coefficients (divided by k!)
//! and return normalized coefficients.
//!
//! ## Key Formulas
//!
//! For Taylor series f(t) = Σₖ fₖ tᵏ (where fₖ = f⁽ᵏ⁾(a)/k!):
//!
//! - **Addition**: (f+g)ₖ = fₖ + gₖ
//! - **Subtraction**: (f-g)ₖ = fₖ - gₖ
//! - **Negation**: (-f)ₖ = -fₖ
//! - **Multiplication** (Cauchy product): (f·g)ₖ = Σⱼ₌₀ᵏ fⱼ · g_{k-j}
//! - **Division** (recurrence): yₖ = (uₖ - Σⱼ₌₁ᵏ y_{k-j}·vⱼ) / v₀
//!
//! ## Performance
//!
//! This module uses `SmallVec<[f64; 8]>` for Taylor coefficients, which stores
//! coefficients inline (on the stack) for orders up to 7, avoiding heap allocation.
//! This significantly reduces allocation overhead for typical derivative orders (1-4).

use smallvec::{smallvec, SmallVec};

use crate::error::{TaylorError, TaylorResult};

/// Taylor coefficient storage optimized for small orders.
///
/// Uses inline storage for orders up to 7 (8 coefficients), falling back to
/// heap allocation for higher orders. This reduces allocation overhead for
/// typical use cases.
pub type TaylorCoeffs = SmallVec<[f64; 8]>;

/// Adds two Taylor series coefficient-wise.
///
/// Given u = [u₀, u₁, ...] and v = [v₀, v₁, ...], computes y = u + v.
/// The result has length `order + 1`, padding shorter inputs with zeros.
///
/// # Example
/// ```
/// use vvad::taylor::add_taylor;
/// let u = vec![1.0, 2.0, 3.0];
/// let v = vec![4.0, 5.0, 6.0];
/// let y = add_taylor(&u, &v, 2);
/// assert_eq!(y.as_slice(), &[5.0, 7.0, 9.0]);
/// ```
pub fn add_taylor(u: &[f64], v: &[f64], order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    for (k, res) in result.iter_mut().enumerate() {
        let uk = u.get(k).copied().unwrap_or(0.0);
        let vk = v.get(k).copied().unwrap_or(0.0);
        *res = uk + vk;
    }
    result
}

/// Subtracts two Taylor series coefficient-wise.
///
/// Given u = [u₀, u₁, ...] and v = [v₀, v₁, ...], computes y = u - v.
/// The result has length `order + 1`.
///
/// # Example
/// ```
/// use vvad::taylor::sub_taylor;
/// let u = vec![5.0, 7.0, 9.0];
/// let v = vec![4.0, 5.0, 6.0];
/// let y = sub_taylor(&u, &v, 2);
/// assert_eq!(y.as_slice(), &[1.0, 2.0, 3.0]);
/// ```
pub fn sub_taylor(u: &[f64], v: &[f64], order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    for (k, res) in result.iter_mut().enumerate() {
        let uk = u.get(k).copied().unwrap_or(0.0);
        let vk = v.get(k).copied().unwrap_or(0.0);
        *res = uk - vk;
    }
    result
}

/// Negates a Taylor series coefficient-wise.
///
/// Given u = [u₀, u₁, ...], computes y = -u.
/// The result has length `order + 1`.
pub fn neg_taylor(u: &[f64], order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    for (k, res) in result.iter_mut().enumerate() {
        let uk = u.get(k).copied().unwrap_or(0.0);
        *res = -uk;
    }
    result
}

/// Multiplies two Taylor series using the Cauchy product (convolution).
///
/// Given u = [u₀, u₁, ...] and v = [v₀, v₁, ...], computes y = u · v where:
///   yₖ = Σⱼ₌₀ᵏ uⱼ · v_{k-j}
///
/// This is O(n²) for order n. The result has length `order + 1`.
///
/// # Example
/// ```
/// use vvad::taylor::mul_taylor;
/// // (1 + 2t) * (1 + 3t) = 1 + 5t + 6t²
/// let u = vec![1.0, 2.0];  // 1 + 2t
/// let v = vec![1.0, 3.0];  // 1 + 3t
/// let y = mul_taylor(&u, &v, 2);
/// assert_eq!(y.as_slice(), &[1.0, 5.0, 6.0]);
/// ```
pub fn mul_taylor(u: &[f64], v: &[f64], order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    for (k, res) in result.iter_mut().enumerate() {
        let mut sum = 0.0;
        for j in 0..=k {
            let uj = u.get(j).copied().unwrap_or(0.0);
            let vkj = v.get(k - j).copied().unwrap_or(0.0);
            sum += uj * vkj;
        }
        *res = sum;
    }
    result
}

/// Divides two Taylor series using the recurrence relation.
///
/// Given u = [u₀, u₁, ...] and v = [v₀, v₁, ...] where v₀ ≠ 0,
/// computes y = u / v by solving y · v = u:
///   y₀ = u₀ / v₀
///   yₖ = (uₖ - Σⱼ₌₁ᵏ y_{k-j} · vⱼ) / v₀
///
/// This is O(n²) for order n. The result has length `order + 1`.
///
/// # Errors
/// Returns [`TaylorError::DivisionByZero`] if v₀ is too close to zero.
///
/// # Example
/// ```
/// use vvad::taylor::div_taylor;
/// // (1 + 2t) / (1 + t) at order 2
/// // Let's verify: (1 + t)(y₀ + y₁t + y₂t²) = 1 + 2t
/// // y₀ = 1, y₁ + y₀ = 2 → y₁ = 1, y₂ + y₁ = 0 → y₂ = -1
/// let u = vec![1.0, 2.0];
/// let v = vec![1.0, 1.0];
/// let y = div_taylor(&u, &v, 2).unwrap();
/// assert_eq!(y.as_slice(), &[1.0, 1.0, -1.0]);
/// ```
pub fn div_taylor(u: &[f64], v: &[f64], order: usize) -> TaylorResult<TaylorCoeffs> {
    let v0 = v.first().copied().unwrap_or(0.0);
    if v0.abs() <= f64::EPSILON {
        return Err(TaylorError::DivisionByZero(v0));
    }

    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];

    // y₀ = u₀ / v₀
    let u0 = u.first().copied().unwrap_or(0.0);
    result[0] = u0 / v0;

    // yₖ = (uₖ - Σⱼ₌₁ᵏ y_{k-j} · vⱼ) / v₀
    for k in 1..=order {
        let uk = u.get(k).copied().unwrap_or(0.0);
        let mut sum = 0.0;
        for j in 1..=k {
            let vj = v.get(j).copied().unwrap_or(0.0);
            sum += result[k - j] * vj;
        }
        result[k] = (uk - sum) / v0;
    }

    Ok(result)
}

/// Scales a Taylor series by a constant.
///
/// Given u = [u₀, u₁, ...] and scalar c, computes y = c · u.
pub fn scale_taylor(u: &[f64], c: f64, order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    for (k, res) in result.iter_mut().enumerate() {
        let uk = u.get(k).copied().unwrap_or(0.0);
        *res = c * uk;
    }
    result
}

/// Creates a constant Taylor series (all higher coefficients are zero).
pub fn constant_taylor(value: f64, order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    result[0] = value;
    result
}

/// Creates an identity Taylor series for a variable with direction component d.
///
/// For an input variable with value x and direction component d:
/// f(t) = x + d·t
///
/// This is used to initialize Taylor propagation for input variables.
pub fn identity_taylor(value: f64, direction_component: f64, order: usize) -> TaylorCoeffs {
    let mut result: TaylorCoeffs = smallvec![0.0; order + 1];
    result[0] = value;
    if order >= 1 {
        result[1] = direction_component;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Add tests
    #[test]
    fn test_add_basic() {
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        let y = add_taylor(&u, &v, 2);
        assert_eq!(y.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_different_lengths() {
        let u = vec![1.0, 2.0];
        let v = vec![4.0, 5.0, 6.0];
        let y = add_taylor(&u, &v, 2);
        assert_eq!(y.as_slice(), &[5.0, 7.0, 6.0]); // u[2] treated as 0
    }

    #[test]
    fn test_add_identity() {
        let u = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        let y = add_taylor(&u, &zero, 2);
        assert_eq!(y.as_slice(), u.as_slice());
    }

    #[test]
    fn test_add_commutativity() {
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        assert_eq!(add_taylor(&u, &v, 2), add_taylor(&v, &u, 2));
    }

    // Sub tests
    #[test]
    fn test_sub_basic() {
        let u = vec![5.0, 7.0, 9.0];
        let v = vec![4.0, 5.0, 6.0];
        let y = sub_taylor(&u, &v, 2);
        assert_eq!(y.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sub_inverse_of_add() {
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        let sum = add_taylor(&u, &v, 2);
        let back = sub_taylor(&sum, &v, 2);
        assert_eq!(back.as_slice(), u.as_slice());
    }

    #[test]
    fn test_sub_self_is_zero() {
        let u = vec![1.0, 2.0, 3.0];
        let y = sub_taylor(&u, &u, 2);
        assert_eq!(y.as_slice(), &[0.0, 0.0, 0.0]);
    }

    // Neg tests
    #[test]
    fn test_neg_basic() {
        let u = vec![1.0, -2.0, 3.0];
        let y = neg_taylor(&u, 2);
        assert_eq!(y.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_neg_double_is_identity() {
        let u = vec![1.0, 2.0, 3.0];
        let y = neg_taylor(&neg_taylor(&u, 2), 2);
        assert_eq!(y.as_slice(), u.as_slice());
    }

    // Mul tests
    #[test]
    fn test_mul_basic() {
        // (1 + 2t) * (1 + 3t) = 1 + 5t + 6t²
        let u = vec![1.0, 2.0];
        let v = vec![1.0, 3.0];
        let y = mul_taylor(&u, &v, 2);
        assert_eq!(y.as_slice(), &[1.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul_identity() {
        let u = vec![3.0, 4.0, 5.0];
        let one = vec![1.0];
        let y = mul_taylor(&u, &one, 2);
        assert_eq!(y.as_slice(), u.as_slice());
    }

    #[test]
    fn test_mul_zero() {
        let u = vec![3.0, 4.0, 5.0];
        let zero = vec![0.0];
        let y = mul_taylor(&u, &zero, 2);
        assert_eq!(y.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_commutativity() {
        let u = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        assert_eq!(mul_taylor(&u, &v, 2), mul_taylor(&v, &u, 2));
    }

    #[test]
    fn test_mul_associativity() {
        let u = vec![1.0, 2.0];
        let v = vec![3.0, 4.0];
        let w = vec![5.0, 6.0];

        let uv = mul_taylor(&u, &v, 2);
        let uv_w = mul_taylor(&uv, &w, 2);

        let vw = mul_taylor(&v, &w, 2);
        let u_vw = mul_taylor(&u, &vw, 2);

        for k in 0..3 {
            assert!(
                (uv_w[k] - u_vw[k]).abs() < 1e-10,
                "Associativity failed at index {}",
                k
            );
        }
    }

    // Div tests
    #[test]
    fn test_div_basic() {
        // (1 + 2t) / (1 + t) should give [1, 1, -1, 1, -1, ...]
        let u = vec![1.0, 2.0];
        let v = vec![1.0, 1.0];
        let y = div_taylor(&u, &v, 2).unwrap();
        assert_eq!(y.as_slice(), &[1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_div_round_trip() {
        // (u * v) / v = u
        let u = vec![2.0, 3.0, 4.0];
        let v = vec![5.0, 1.0, 2.0];

        let product = mul_taylor(&u, &v, 2);
        let back = div_taylor(&product, &v, 2).unwrap();

        for k in 0..3 {
            assert!(
                (back[k] - u[k]).abs() < 1e-10,
                "Round-trip failed at index {}: got {}, expected {}",
                k,
                back[k],
                u[k]
            );
        }
    }

    #[test]
    fn test_div_by_constant() {
        let u = vec![6.0, 12.0, 18.0];
        let v = vec![3.0];
        let y = div_taylor(&u, &v, 2).unwrap();
        assert_eq!(y.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_div_by_zero_returns_error() {
        let u = vec![1.0, 2.0];
        let v = vec![0.0, 1.0];
        let result = div_taylor(&u, &v, 2);
        assert!(result.is_err());
        assert!(matches!(result, Err(TaylorError::DivisionByZero(_))));
    }

    // Scale tests
    #[test]
    fn test_scale_basic() {
        let u = vec![1.0, 2.0, 3.0];
        let y = scale_taylor(&u, 2.0, 2);
        assert_eq!(y.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_zero() {
        let u = vec![1.0, 2.0, 3.0];
        let y = scale_taylor(&u, 0.0, 2);
        assert_eq!(y.as_slice(), &[0.0, 0.0, 0.0]);
    }

    // Constant tests
    #[test]
    fn test_constant_basic() {
        let y = constant_taylor(5.0, 3);
        assert_eq!(y.as_slice(), &[5.0, 0.0, 0.0, 0.0]);
    }

    // Identity tests
    #[test]
    fn test_identity_taylor() {
        let y = identity_taylor(2.0, 1.0, 3);
        assert_eq!(y.as_slice(), &[2.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_identity_zero_direction() {
        let y = identity_taylor(5.0, 0.0, 2);
        assert_eq!(y.as_slice(), &[5.0, 0.0, 0.0]);
    }

    // Integration test: polynomial f(x) = x² - verified with Taylor series
    #[test]
    fn test_polynomial_x_squared() {
        // For f(x) = x², at x=2, evaluating along direction d=1:
        // f(2 + t) = (2 + t)² = 4 + 4t + t²
        // Normalized coefficients: [4, 4, 1]
        let x = identity_taylor(2.0, 1.0, 2); // x(t) = 2 + t
        let x_squared = mul_taylor(&x, &x, 2);
        assert_eq!(x_squared.as_slice(), &[4.0, 4.0, 1.0]);

        // First derivative: d/dx(x²) = 2x = 4 at x=2
        // Coefficient[1] * 1! = 4
        assert_eq!(x_squared[1], 4.0);

        // Second derivative: d²/dx²(x²) = 2
        // Coefficient[2] * 2! = 1 * 2 = 2
        assert_eq!(x_squared[2] * 2.0, 2.0);
    }
}
