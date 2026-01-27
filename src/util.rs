//! Numerical utilities for Taylor-mode autodiff.
//!
//! Provides compile-time factorial lookup, binomial coefficients,
//! and Horner's method for polynomial evaluation.

/// Precomputed factorial lookup table (compile-time const).
/// 170! ≈ 7.26e306 is the largest that fits in f64.
/// 171! overflows f64 to infinity.
pub const FACTORIAL: [f64; 171] = {
    let mut table = [1.0; 171];
    let mut i = 1;
    while i < 171 {
        table[i] = table[i - 1] * (i as f64);
        i += 1;
    }
    table
};

/// O(1) factorial lookup.
///
/// # Panics
/// Panics if n > 170.
#[inline]
pub fn factorial(n: usize) -> f64 {
    FACTORIAL[n]
}

/// Binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Uses the multiplicative formula to avoid overflow for large n:
/// C(n, k) = ∏(i=0..k-1) (n-i)/(i+1)
///
/// Returns 0.0 if k > n.
#[inline]
pub fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    // Use symmetry: C(n, k) = C(n, n-k)
    let k = k.min(n - k);
    let mut result = 1.0;
    for i in 0..k {
        result = result * ((n - i) as f64) / ((i + 1) as f64);
    }
    result
}

/// Horner's method for polynomial evaluation.
///
/// Evaluates p(x) = c₀ + c₁x + c₂x² + ... as c₀ + x(c₁ + x(c₂ + ...))
///
/// This is O(n) multiplications and numerically more stable than
/// directly computing each power.
///
/// Returns 0.0 for empty coefficient slice.
#[inline]
pub fn horner_eval(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_base_cases() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(2), 2.0);
        assert_eq!(factorial(3), 6.0);
        assert_eq!(factorial(4), 24.0);
        assert_eq!(factorial(5), 120.0);
    }

    #[test]
    fn test_factorial_larger_values() {
        assert_eq!(factorial(10), 3628800.0);
        assert_eq!(factorial(20), 2432902008176640000.0);
        // Check 170! doesn't overflow
        assert!(factorial(170).is_finite());
        assert!(factorial(170) > 1e300);
    }

    #[test]
    fn test_factorial_table_consistency() {
        // Verify each entry is previous * index
        for i in 1..171 {
            let expected = FACTORIAL[i - 1] * (i as f64);
            assert!(
                (FACTORIAL[i] - expected).abs() < expected * 1e-10,
                "Factorial table inconsistent at index {}",
                i
            );
        }
    }

    #[test]
    fn test_binomial_base_cases() {
        assert_eq!(binomial(0, 0), 1.0);
        assert_eq!(binomial(1, 0), 1.0);
        assert_eq!(binomial(1, 1), 1.0);
        assert_eq!(binomial(5, 0), 1.0);
        assert_eq!(binomial(5, 5), 1.0);
    }

    #[test]
    fn test_binomial_pascal_triangle() {
        // Row 5: 1, 5, 10, 10, 5, 1
        assert_eq!(binomial(5, 0), 1.0);
        assert_eq!(binomial(5, 1), 5.0);
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(5, 3), 10.0);
        assert_eq!(binomial(5, 4), 5.0);
        assert_eq!(binomial(5, 5), 1.0);
    }

    #[test]
    fn test_binomial_symmetry() {
        for n in 0..20 {
            for k in 0..=n {
                assert_eq!(
                    binomial(n, k),
                    binomial(n, n - k),
                    "Symmetry failed for C({}, {})",
                    n,
                    k
                );
            }
        }
    }

    #[test]
    fn test_binomial_k_greater_than_n() {
        assert_eq!(binomial(5, 6), 0.0);
        assert_eq!(binomial(0, 1), 0.0);
        assert_eq!(binomial(10, 100), 0.0);
    }

    #[test]
    fn test_binomial_large_values() {
        // C(100, 50) is a large but computable value
        let c_100_50 = binomial(100, 50);
        assert!(c_100_50.is_finite());
        // Approximately 1.009e29
        assert!(c_100_50 > 1e29);
        assert!(c_100_50 < 1.1e29);
    }

    #[test]
    fn test_horner_constant() {
        assert_eq!(horner_eval(&[5.0], 10.0), 5.0);
        assert_eq!(horner_eval(&[5.0], 0.0), 5.0);
    }

    #[test]
    fn test_horner_linear() {
        // p(x) = 2 + 3x
        let coeffs = [2.0, 3.0];
        assert_eq!(horner_eval(&coeffs, 0.0), 2.0);
        assert_eq!(horner_eval(&coeffs, 1.0), 5.0);
        assert_eq!(horner_eval(&coeffs, 2.0), 8.0);
    }

    #[test]
    fn test_horner_quadratic() {
        // p(x) = 1 + 2x + 3x²
        let coeffs = [1.0, 2.0, 3.0];
        assert_eq!(horner_eval(&coeffs, 0.0), 1.0);
        assert_eq!(horner_eval(&coeffs, 1.0), 6.0); // 1 + 2 + 3
        assert_eq!(horner_eval(&coeffs, 2.0), 17.0); // 1 + 4 + 12
    }

    #[test]
    fn test_horner_empty() {
        assert_eq!(horner_eval(&[], 5.0), 0.0);
    }

    #[test]
    fn test_horner_taylor_exp() {
        // Taylor series for e^x at x=0: 1 + x + x²/2! + x³/3! + ...
        // Evaluate at x=1, should approximate e
        let coeffs: Vec<f64> = (0..20).map(|k| 1.0 / factorial(k)).collect();
        let result = horner_eval(&coeffs, 1.0);
        let expected = std::f64::consts::E;
        assert!((result - expected).abs() < 1e-10);
    }
}
