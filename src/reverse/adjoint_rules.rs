//! Adjoint rules for reverse mode Taylor AD.
//!
//! Each adjoint rule takes:
//! - The forward Taylor coefficients of inputs
//! - The adjoint Taylor coefficients of the output (ȳ)
//!
//! And returns/accumulates:
//! - The adjoint contributions to the inputs (ū, v̄, etc.)
//!
//! These rules implement the chain rule for Taylor polynomials.

use crate::taylor::polynomial::{
    add_taylor, constant_taylor, div_taylor, mul_taylor, neg_taylor, scale_taylor, sub_taylor,
    TaylorCoeffs,
};

/// Adjoint rule for y = u + v
///
/// Since dy/du = 1 and dy/dv = 1:
/// - ū += ȳ
/// - v̄ += ȳ
pub fn adjoint_add(y_adj: &[f64], u_adj: &mut [f64], v_adj: &mut [f64]) {
    for (i, &y) in y_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += y;
        }
        if i < v_adj.len() {
            v_adj[i] += y;
        }
    }
}

/// Adjoint rule for y = u - v
///
/// Since dy/du = 1 and dy/dv = -1:
/// - ū += ȳ
/// - v̄ -= ȳ
pub fn adjoint_sub(y_adj: &[f64], u_adj: &mut [f64], v_adj: &mut [f64]) {
    for (i, &y) in y_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += y;
        }
        if i < v_adj.len() {
            v_adj[i] -= y;
        }
    }
}

/// Adjoint rule for y = u * v
///
/// Since dy/du = v and dy/dv = u:
/// - ū += v · ȳ (polynomial multiplication)
/// - v̄ += u · ȳ (polynomial multiplication)
pub fn adjoint_mul(y_adj: &[f64], u: &[f64], v: &[f64], u_adj: &mut [f64], v_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // ū += v · ȳ
    let v_y = mul_taylor(v, y_adj, order);
    for (i, &vy) in v_y.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += vy;
        }
    }

    // v̄ += u · ȳ
    let u_y = mul_taylor(u, y_adj, order);
    for (i, &uy) in u_y.iter().enumerate() {
        if i < v_adj.len() {
            v_adj[i] += uy;
        }
    }
}

/// Adjoint rule for y = u / v
///
/// Since dy/du = 1/v and dy/dv = -u/v²:
/// - ū += ȳ / v
/// - v̄ += -u · ȳ / v²
///
/// We use y = u/v, so:
/// - ū += ȳ / v
/// - v̄ += -y · ȳ / v
pub fn adjoint_div(
    y_adj: &[f64],
    _u: &[f64],
    v: &[f64],
    y: &[f64],
    u_adj: &mut [f64],
    v_adj: &mut [f64],
) {
    let order = y_adj.len().saturating_sub(1);

    // ū += ȳ / v
    let y_over_v = crate::taylor::polynomial::div_taylor(y_adj, v, order).expect("division by zero in adjoint_div");
    for (i, &yv) in y_over_v.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += yv;
        }
    }

    // v̄ += -y · ȳ / v
    let y_times_y_adj = mul_taylor(y, y_adj, order);
    let neg_y_y_adj = neg_taylor(&y_times_y_adj, order);
    let v_contrib = crate::taylor::polynomial::div_taylor(&neg_y_y_adj, v, order).expect("division by zero in adjoint_div");
    for (i, &vc) in v_contrib.iter().enumerate() {
        if i < v_adj.len() {
            v_adj[i] += vc;
        }
    }
}

/// Adjoint rule for y = -u
///
/// Since dy/du = -1:
/// - ū -= ȳ
pub fn adjoint_neg(y_adj: &[f64], u_adj: &mut [f64]) {
    for (i, &y) in y_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] -= y;
        }
    }
}

/// Adjoint rule for (s, c) = (sin(u), cos(u))
///
/// Since ds/du = c and dc/du = -s:
/// - ū += c · s̄ - s · c̄
///
/// Note: Both sin and cos outputs share the same input, so their adjoints
/// contribute to the same input adjoint.
pub fn adjoint_sin_cos(
    sin_adj: &[f64],
    cos_adj: &[f64],
    sin_coeffs: &[f64],
    cos_coeffs: &[f64],
    u_adj: &mut [f64],
) {
    let order = sin_adj.len().saturating_sub(1).max(cos_adj.len().saturating_sub(1));

    // ū += c · s̄ (derivative of sin is cos)
    let cos_sin_adj = mul_taylor(cos_coeffs, sin_adj, order);
    for (i, &c) in cos_sin_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }

    // ū -= s · c̄ (derivative of cos is -sin)
    let sin_cos_adj = mul_taylor(sin_coeffs, cos_adj, order);
    for (i, &s) in sin_cos_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] -= s;
        }
    }
}

/// Adjoint rule for y = exp(u)
///
/// Since dy/du = exp(u) = y:
/// - ū += y · ȳ
pub fn adjoint_exp(y_adj: &[f64], y: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // ū += y · ȳ
    let y_times_y_adj = mul_taylor(y, y_adj, order);
    for (i, &yy) in y_times_y_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += yy;
        }
    }
}

/// Adjoint rule for y = ln(u)
///
/// Since dy/du = 1/u:
/// - ū += ȳ / u
pub fn adjoint_ln(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // ū += ȳ / u
    let y_adj_over_u = crate::taylor::polynomial::div_taylor(y_adj, u, order)
        .expect("division by zero in adjoint_ln");
    for (i, &yu) in y_adj_over_u.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += yu;
        }
    }
}

/// Adjoint rule for y = sqrt(u)
///
/// Since dy/du = 1/(2·sqrt(u)) = 1/(2y):
/// - ū += ȳ / (2y)
pub fn adjoint_sqrt(y_adj: &[f64], y: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // ū += ȳ / (2y)
    let two_y = scale_taylor(y, 2.0, order);
    let y_adj_over_2y = crate::taylor::polynomial::div_taylor(y_adj, &two_y, order)
        .expect("division by zero in adjoint_sqrt");
    for (i, &c) in y_adj_over_2y.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for (sh, ch) = (sinh(u), cosh(u))
///
/// Since d(sinh)/du = cosh and d(cosh)/du = sinh:
/// - ū += ch · sh̄ + sh · ch̄
pub fn adjoint_sinh_cosh(
    sinh_adj: &[f64],
    cosh_adj: &[f64],
    sinh_coeffs: &[f64],
    cosh_coeffs: &[f64],
    u_adj: &mut [f64],
) {
    let order = sinh_adj
        .len()
        .saturating_sub(1)
        .max(cosh_adj.len().saturating_sub(1));

    // ū += ch · sh̄ (derivative of sinh is cosh)
    let cosh_sinh_adj = mul_taylor(cosh_coeffs, sinh_adj, order);
    for (i, &c) in cosh_sinh_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }

    // ū += sh · ch̄ (derivative of cosh is sinh)
    let sinh_cosh_adj = mul_taylor(sinh_coeffs, cosh_adj, order);
    for (i, &s) in sinh_cosh_adj.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += s;
        }
    }
}

/// Adjoint rule for y = u^p (constant exponent)
///
/// Since dy/du = p · u^(p-1):
/// - ū += p · u^(p-1) · ȳ
///
/// We can compute this as: ū += p · y/u · ȳ
pub fn adjoint_pow_const(
    y_adj: &[f64],
    u: &[f64],
    y: &[f64],
    p: f64,
    u_adj: &mut [f64],
) {
    let order = y_adj.len().saturating_sub(1);

    // p · y/u · ȳ
    let y_over_u = crate::taylor::polynomial::div_taylor(y, u, order).expect("division by zero in adjoint_pow_const");
    let p_y_over_u = scale_taylor(&y_over_u, p, order);
    let contrib = mul_taylor(&p_y_over_u, y_adj, order);

    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

// =============================================================================
// Specialized adjoint rules for tan, tanh, and inverse trig/hyperbolic functions
// =============================================================================

/// Adjoint rule for y = tan(u)
///
/// Since dy/du = sec²(u) = 1 + tan²(u) = 1 + y²:
/// - ū += ȳ · (1 + y²)
pub fn adjoint_tan(y_adj: &[f64], y: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute 1 + y²
    let one = constant_taylor(1.0, order);
    let y_squared = mul_taylor(y, y, order);
    let one_plus_y2 = add_taylor(&one, &y_squared, order);

    // ū += ȳ · (1 + y²)
    let contrib = mul_taylor(y_adj, &one_plus_y2, order);
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = tanh(u)
///
/// Since dy/du = sech²(u) = 1 - tanh²(u) = 1 - y²:
/// - ū += ȳ · (1 - y²)
pub fn adjoint_tanh(y_adj: &[f64], y: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute 1 - y²
    let one = constant_taylor(1.0, order);
    let y_squared = mul_taylor(y, y, order);
    let one_minus_y2 = sub_taylor(&one, &y_squared, order);

    // ū += ȳ · (1 - y²)
    let contrib = mul_taylor(y_adj, &one_minus_y2, order);
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = asin(u)
///
/// Since dy/du = 1/√(1-u²):
/// - ū += ȳ / √(1-u²)
pub fn adjoint_asin(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute √(1 - u²)
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_minus_u2 = sub_taylor(&one, &u_squared, order);
    let sqrt_term = crate::taylor::rules::elementary::sqrt_taylor(&one_minus_u2, order)
        .expect("domain error in adjoint_asin");

    // ū += ȳ / √(1-u²)
    let contrib = div_taylor(y_adj, &sqrt_term, order).expect("division by zero in adjoint_asin");
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = acos(u)
///
/// Since dy/du = -1/√(1-u²):
/// - ū += -ȳ / √(1-u²)
pub fn adjoint_acos(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute √(1 - u²)
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_minus_u2 = sub_taylor(&one, &u_squared, order);
    let sqrt_term = crate::taylor::rules::elementary::sqrt_taylor(&one_minus_u2, order)
        .expect("domain error in adjoint_acos");

    // ū += -ȳ / √(1-u²)
    let neg_y_adj = neg_taylor(y_adj, order);
    let contrib = div_taylor(&neg_y_adj, &sqrt_term, order).expect("division by zero in adjoint_acos");
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = atan(u)
///
/// Since dy/du = 1/(1+u²):
/// - ū += ȳ / (1+u²)
pub fn adjoint_atan(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute 1 + u²
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_plus_u2 = add_taylor(&one, &u_squared, order);

    // ū += ȳ / (1+u²)
    let contrib = div_taylor(y_adj, &one_plus_u2, order).expect("division by zero in adjoint_atan");
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = asinh(u)
///
/// Since dy/du = 1/√(u²+1):
/// - ū += ȳ / √(u²+1)
pub fn adjoint_asinh(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute √(u² + 1)
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let u2_plus_one = add_taylor(&u_squared, &one, order);
    let sqrt_term = crate::taylor::rules::elementary::sqrt_taylor(&u2_plus_one, order)
        .expect("domain error in adjoint_asinh");

    // ū += ȳ / √(u²+1)
    let contrib = div_taylor(y_adj, &sqrt_term, order).expect("division by zero in adjoint_asinh");
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = acosh(u)
///
/// Since dy/du = 1/√(u²-1):
/// - ū += ȳ / √(u²-1)
pub fn adjoint_acosh(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute √(u² - 1)
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let u2_minus_one = sub_taylor(&u_squared, &one, order);
    let sqrt_term = crate::taylor::rules::elementary::sqrt_taylor(&u2_minus_one, order)
        .expect("domain error in adjoint_acosh");

    // ū += ȳ / √(u²-1)
    let contrib = div_taylor(y_adj, &sqrt_term, order).expect("division by zero in adjoint_acosh");
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

/// Adjoint rule for y = atanh(u)
///
/// Since dy/du = 1/(1-u²):
/// - ū += ȳ / (1-u²)
pub fn adjoint_atanh(y_adj: &[f64], u: &[f64], u_adj: &mut [f64]) {
    let order = y_adj.len().saturating_sub(1);

    // Compute 1 - u²
    let one = constant_taylor(1.0, order);
    let u_squared = mul_taylor(u, u, order);
    let one_minus_u2 = sub_taylor(&one, &u_squared, order);

    // ū += ȳ / (1-u²)
    let contrib = div_taylor(y_adj, &one_minus_u2, order).expect("division by zero in adjoint_atanh");
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

// =============================================================================
// Improved generic adjoint rule (fallback for unknown functions)
// =============================================================================

/// Differentiates a Taylor series with respect to t.
///
/// For y(t) = Σₖ yₖ tᵏ, returns y'(t) = Σₖ (k+1)·y_{k+1} tᵏ
///
/// This shifts coefficients: y'_k = (k+1) · y_{k+1}
#[allow(dead_code)] // Used in adjoint_chain_rule and tests
fn differentiate_taylor(y: &[f64], order: usize) -> TaylorCoeffs {
    let mut result = TaylorCoeffs::new();
    for k in 0..=order {
        let coeff = y.get(k + 1).copied().unwrap_or(0.0) * ((k + 1) as f64);
        result.push(coeff);
    }
    result
}

/// Generic adjoint rule using the chain rule for y = f(u).
///
/// This is a fallback for functions where we don't have a specialized adjoint rule.
/// It computes ū += f'(u) · ȳ where f'(u) is derived from the Taylor coefficients.
///
/// ## Method
///
/// For y = f(u(t)), we have by chain rule: y' = f'(u) · u'
///
/// Therefore: f'(u) = y' / u' (as Taylor polynomial division)
///
/// This gives us the full Taylor expansion of f'(u), not just the constant term.
///
/// ## Fallback
///
/// If u' is too small (constant u), we fall back to extracting f'(u₀) from
/// the first Taylor coefficient.
///
/// ## Note
///
/// All current unary operations have specialized adjoint rules, but this function
/// is kept as a fallback for future operations or external use.
#[allow(dead_code)] // Kept as fallback for future operations
pub fn adjoint_chain_rule(
    y_adj: &[f64],
    u: &[f64],
    y: &[f64],
    u_adj: &mut [f64],
    order: usize,
) {
    // Compute y' = dy/dt and u' = du/dt (derivatives of Taylor series w.r.t. t)
    let y_prime = differentiate_taylor(y, order);
    let u_prime = differentiate_taylor(u, order);

    // Check if u' is non-trivial (u is not constant)
    let u_prime_magnitude: f64 = u_prime.iter().map(|x| x.abs()).sum();

    let f_prime_taylor: TaylorCoeffs = if u_prime_magnitude > 1e-12 {
        // f'(u) = y' / u' (Taylor polynomial division)
        match div_taylor(&y_prime, &u_prime, order) {
            Ok(result) => result,
            Err(_) => {
                // Division failed, fall back to constant approximation
                let u1 = u.get(1).copied().unwrap_or(0.0);
                let y1 = y.get(1).copied().unwrap_or(0.0);
                let f_prime = if u1.abs() > 1e-15 { y1 / u1 } else { y1 };
                constant_taylor(f_prime, order)
            }
        }
    } else {
        // u is essentially constant, use scalar derivative
        let u1 = u.get(1).copied().unwrap_or(0.0);
        let y1 = y.get(1).copied().unwrap_or(0.0);
        let f_prime = if u1.abs() > 1e-15 { y1 / u1 } else { y1 };
        constant_taylor(f_prime, order)
    };

    // ū += f'(u) · ȳ (Taylor polynomial multiplication)
    let contrib = mul_taylor(&f_prime_taylor, y_adj, order);
    for (i, &c) in contrib.iter().enumerate() {
        if i < u_adj.len() {
            u_adj[i] += c;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_adjoint_add() {
        let y_adj = vec![1.0, 2.0, 3.0];
        let mut u_adj = vec![0.0, 0.0, 0.0];
        let mut v_adj = vec![0.0, 0.0, 0.0];

        adjoint_add(&y_adj, &mut u_adj, &mut v_adj);

        assert_eq!(u_adj, vec![1.0, 2.0, 3.0]);
        assert_eq!(v_adj, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_adjoint_sub() {
        let y_adj = vec![1.0, 2.0, 3.0];
        let mut u_adj = vec![0.0, 0.0, 0.0];
        let mut v_adj = vec![0.0, 0.0, 0.0];

        adjoint_sub(&y_adj, &mut u_adj, &mut v_adj);

        assert_eq!(u_adj, vec![1.0, 2.0, 3.0]);
        assert_eq!(v_adj, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_adjoint_neg() {
        let y_adj = vec![1.0, 2.0, 3.0];
        let mut u_adj = vec![0.0, 0.0, 0.0];

        adjoint_neg(&y_adj, &mut u_adj);

        assert_eq!(u_adj, vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_adjoint_mul_constant() {
        // y = u * v where u = [2], v = [3], y = [6]
        // dy/du = v = 3, dy/dv = u = 2
        // With ȳ = [1], ū = 3, v̄ = 2
        let u = vec![2.0];
        let v = vec![3.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];
        let mut v_adj = vec![0.0];

        adjoint_mul(&y_adj, &u, &v, &mut u_adj, &mut v_adj);

        assert_relative_eq!(u_adj[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(v_adj[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_mul_linear() {
        // y = u * v where u = [1, 1] (1 + t), v = [2, 0] (constant 2)
        // y = [2, 2] (2 + 2t)
        // dy/du = v, so ū += v · ȳ
        // dy/dv = u, so v̄ += u · ȳ
        let u = vec![1.0, 1.0];
        let v = vec![2.0, 0.0];
        let y_adj = vec![1.0, 0.0];
        let mut u_adj = vec![0.0, 0.0];
        let mut v_adj = vec![0.0, 0.0];

        adjoint_mul(&y_adj, &u, &v, &mut u_adj, &mut v_adj);

        // ū = v · ȳ = [2, 0] · [1, 0] = [2, ...]
        assert_relative_eq!(u_adj[0], 2.0, epsilon = 1e-10);
        // v̄ = u · ȳ = [1, 1] · [1, 0] = [1, ...]
        assert_relative_eq!(v_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_exp_at_zero() {
        // y = exp(u) at u=0
        // y = [1, ...] (exp(0) = 1)
        // dy/du = y, so ū += y · ȳ
        let y = vec![1.0, 1.0, 0.5]; // exp Taylor at 0
        let y_adj = vec![1.0, 0.0, 0.0];
        let mut u_adj = vec![0.0, 0.0, 0.0];

        adjoint_exp(&y_adj, &y, &mut u_adj);

        // ū = y · ȳ = [1, 1, 0.5] · [1, 0, 0] = [1, ...]
        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_ln_at_one() {
        // y = ln(u) at u=1
        // dy/du = 1/u = 1 at u=1
        // ū += ȳ / u
        let u = vec![1.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_ln(&y_adj, &u, &mut u_adj);

        // ū = ȳ / u = 1 / 1 = 1
        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_sqrt_at_four() {
        // y = sqrt(u) at u=4
        // y = 2
        // dy/du = 1/(2y) = 1/4
        // ū += ȳ / (2y)
        let y = vec![2.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_sqrt(&y_adj, &y, &mut u_adj);

        // ū = ȳ / (2y) = 1 / 4 = 0.25
        assert_relative_eq!(u_adj[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_pow_const_square() {
        // y = u^2 at u=3
        // y = 9
        // dy/du = 2u = 6
        // ū += 2 · y/u · ȳ = 2 · 9/3 · 1 = 6
        let u = vec![3.0];
        let y = vec![9.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_pow_const(&y_adj, &u, &y, 2.0, &mut u_adj);

        assert_relative_eq!(u_adj[0], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_accumulation() {
        // Test that multiple calls accumulate correctly
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];
        let mut v_adj = vec![0.0];

        // First contribution
        adjoint_add(&y_adj, &mut u_adj, &mut v_adj);
        // Second contribution
        adjoint_add(&y_adj, &mut u_adj, &mut v_adj);

        assert_eq!(u_adj, vec![2.0]);
        assert_eq!(v_adj, vec![2.0]);
    }

    // =========================================================================
    // Tests for new specialized adjoint rules
    // =========================================================================

    #[test]
    fn test_adjoint_tan_at_zero() {
        // y = tan(u) at u=0
        // y = 0, dy/du = sec²(0) = 1
        // ū += ȳ · (1 + y²) = 1 · (1 + 0) = 1
        let y = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_tan(&y_adj, &y, &mut u_adj);

        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_tan_at_pi_over_4() {
        // y = tan(π/4) = 1, dy/du = sec²(π/4) = 2
        // ū += ȳ · (1 + y²) = 1 · (1 + 1) = 2
        let y = vec![1.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_tan(&y_adj, &y, &mut u_adj);

        assert_relative_eq!(u_adj[0], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_tanh_at_zero() {
        // y = tanh(u) at u=0
        // y = 0, dy/du = sech²(0) = 1
        // ū += ȳ · (1 - y²) = 1 · (1 - 0) = 1
        let y = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_tanh(&y_adj, &y, &mut u_adj);

        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_asin_at_zero() {
        // y = asin(u) at u=0
        // dy/du = 1/√(1-0²) = 1
        // ū += ȳ / √(1-u²) = 1 / 1 = 1
        let u = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_asin(&y_adj, &u, &mut u_adj);

        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_asin_at_half() {
        // y = asin(0.5), dy/du = 1/√(1-0.25) = 1/√0.75 ≈ 1.1547
        let u = vec![0.5];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_asin(&y_adj, &u, &mut u_adj);

        let expected = 1.0 / (0.75_f64).sqrt();
        assert_relative_eq!(u_adj[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_acos_at_zero() {
        // y = acos(u) at u=0
        // dy/du = -1/√(1-0²) = -1
        // ū += -ȳ / √(1-u²) = -1 / 1 = -1
        let u = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_acos(&y_adj, &u, &mut u_adj);

        assert_relative_eq!(u_adj[0], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_atan_at_zero() {
        // y = atan(u) at u=0
        // dy/du = 1/(1+0²) = 1
        let u = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_atan(&y_adj, &u, &mut u_adj);

        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_atan_at_one() {
        // y = atan(1) = π/4, dy/du = 1/(1+1) = 0.5
        let u = vec![1.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_atan(&y_adj, &u, &mut u_adj);

        assert_relative_eq!(u_adj[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_asinh_at_zero() {
        // y = asinh(u) at u=0
        // dy/du = 1/√(0²+1) = 1
        let u = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_asinh(&y_adj, &u, &mut u_adj);

        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_acosh_at_two() {
        // y = acosh(2), dy/du = 1/√(4-1) = 1/√3 ≈ 0.5774
        let u = vec![2.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_acosh(&y_adj, &u, &mut u_adj);

        let expected = 1.0 / (3.0_f64).sqrt();
        assert_relative_eq!(u_adj[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_atanh_at_zero() {
        // y = atanh(u) at u=0
        // dy/du = 1/(1-0²) = 1
        let u = vec![0.0];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_atanh(&y_adj, &u, &mut u_adj);

        assert_relative_eq!(u_adj[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adjoint_atanh_at_half() {
        // y = atanh(0.5), dy/du = 1/(1-0.25) = 1/0.75 ≈ 1.333
        let u = vec![0.5];
        let y_adj = vec![1.0];
        let mut u_adj = vec![0.0];

        adjoint_atanh(&y_adj, &u, &mut u_adj);

        let expected = 1.0 / 0.75;
        assert_relative_eq!(u_adj[0], expected, epsilon = 1e-10);
    }

    #[test]
    fn test_differentiate_taylor() {
        // y(t) = 1 + 2t + 3t² → y'(t) = 2 + 6t
        // Coefficients: [1, 2, 3] → [2, 6]
        let y = vec![1.0, 2.0, 3.0];
        let y_prime = differentiate_taylor(&y, 1);

        assert_relative_eq!(y_prime[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(y_prime[1], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_improved_chain_rule() {
        // Test that the improved chain rule produces the same result as scalar case
        // for simple first-order derivatives
        let u = vec![1.0, 1.0]; // u(t) = 1 + t
        let y = vec![1.0_f64.exp(), 1.0_f64.exp()]; // y = exp(1+t) at t=0, derivative = e
        let y_adj = vec![1.0, 0.0];
        let mut u_adj = vec![0.0, 0.0];

        adjoint_chain_rule(&y_adj, &u, &y, &mut u_adj, 1);

        // dy/du = exp(u) = e at u=1
        assert_relative_eq!(u_adj[0], std::f64::consts::E, epsilon = 1e-10);
    }
}
