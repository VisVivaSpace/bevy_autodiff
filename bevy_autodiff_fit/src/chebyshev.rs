//! Core Chebyshev polynomial math.
//!
//! Pure math utilities — no bevy_autodiff dependency within this module.
//! Provides Chebyshev nodes, coefficient computation via DCT, Clenshaw
//! evaluation, and derivative coefficient recurrence.

use std::f64::consts::PI;

/// Chebyshev nodes of the first kind on [-1, 1].
///
/// Returns n points: x_k = cos(π(2k+1) / (2n)) for k = 0..n-1.
/// These are the zeros of T_n(x) and the optimal interpolation points.
pub fn chebyshev_nodes(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| (PI * (2 * k + 1) as f64 / (2 * n) as f64).cos())
        .collect()
}

/// Compute Chebyshev coefficients from function values at Chebyshev nodes.
///
/// Given values f(x_k) at the n Chebyshev nodes (from [`chebyshev_nodes`]),
/// computes coefficients c_0, ..., c_{n-1} such that:
///   f(x) ≈ c_0/2 + Σ_{k=1}^{n-1} c_k T_k(x)
///
/// Uses a direct O(n²) discrete cosine transform. Adequate for degree 10-40.
pub fn chebyshev_coefficients(values_at_nodes: &[f64]) -> Vec<f64> {
    let n = values_at_nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut coeffs = Vec::with_capacity(n);

    for k in 0..n {
        let mut sum = 0.0;
        for (j, &val) in values_at_nodes.iter().enumerate() {
            // T_k(x_j) = cos(k * arccos(x_j)) = cos(k * π(2j+1)/(2n))
            let angle = k as f64 * PI * (2 * j + 1) as f64 / (2 * n) as f64;
            sum += val * angle.cos();
        }
        coeffs.push(2.0 * sum / n as f64);
    }

    coeffs
}

/// Evaluate a Chebyshev series at point t ∈ [-1, 1] via Clenshaw's algorithm.
///
/// Given coefficients c_0, ..., c_N (where the series is c_0/2 + Σ c_k T_k),
/// evaluates using the backward recurrence:
///   b_{N+1} = b_{N+2} = 0
///   b_k = 2·t·b_{k+1} - b_{k+2} + c_k   for k = N, N-1, ..., 1
///   result = t·b_1 - b_2 + c_0/2
pub fn clenshaw_eval(coeffs: &[f64], t: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    if coeffs.len() == 1 {
        return coeffs[0] / 2.0;
    }

    let n = coeffs.len() - 1; // degree
    let mut b_next = 0.0; // b_{k+1}
    let mut b_next2 = 0.0; // b_{k+2}

    // Sweep from k = N down to k = 1
    for k in (1..=n).rev() {
        let b_curr = 2.0 * t * b_next - b_next2 + coeffs[k];
        b_next2 = b_next;
        b_next = b_curr;
    }

    // Final step: result = t * b_1 - b_2 + c_0/2
    t * b_next - b_next2 + coeffs[0] / 2.0
}

/// Compute the Chebyshev coefficients of the derivative.
///
/// Given coefficients c_0, ..., c_N of a Chebyshev series on [-1, 1],
/// returns coefficients c'_0, ..., c'_{N-1} of the derivative series.
///
/// Uses the backward recurrence:
///   c'_{N} = 0  (degree drops by one)
///   c'_{N-1} = 2N · c_N
///   c'_k = c'_{k+2} + 2(k+1) · c_{k+1}   for k = N-2, ..., 0
///
/// The output coefficients follow the same convention: f'(x) = c'_0/2 + Σ c'_k T_k(x).
pub fn derivative_coefficients(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![0.0];
    }

    let mut dc = vec![0.0; n - 1];

    // c'_{N-1} = 2N * c_N
    dc[n - 2] = 2.0 * (n - 1) as f64 * coeffs[n - 1];

    // Backward recurrence: c'_k = c'_{k+2} + 2(k+1) * c_{k+1}
    if n >= 3 {
        for k in (0..n - 2).rev() {
            let c_prime_k_plus_2 = if k + 2 < dc.len() { dc[k + 2] } else { 0.0 };
            dc[k] = c_prime_k_plus_2 + 2.0 * (k + 1) as f64 * coeffs[k + 1];
        }
    }

    dc
}

/// Piecewise linear interpolation: given sorted (x_data, y_data), evaluate at x_eval.
///
/// For dense data, this is the appropriate resampling method — it avoids the
/// Runge phenomenon that polynomial interpolation through many points causes.
/// The error is O(h²) where h is the data spacing, which is negligible when
/// the data is dense relative to the polynomial degree.
///
/// `x_data` must be sorted in ascending order.
pub fn linear_interpolate(x_data: &[f64], y_data: &[f64], x_eval: &[f64]) -> Vec<f64> {
    let n = x_data.len();
    debug_assert_eq!(n, y_data.len());
    debug_assert!(n >= 2);

    x_eval
        .iter()
        .map(|&x| {
            // Clamp to data range
            if x <= x_data[0] {
                return y_data[0];
            }
            if x >= x_data[n - 1] {
                return y_data[n - 1];
            }

            // Binary search for the interval containing x
            let i = match x_data.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
                Ok(i) => return y_data[i],
                Err(i) => i,
            };

            // Linear interpolation in [x_data[i-1], x_data[i]]
            let t = (x - x_data[i - 1]) / (x_data[i] - x_data[i - 1]);
            y_data[i - 1] + t * (y_data[i] - y_data[i - 1])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_count_and_range() {
        for n in [1, 5, 10, 20] {
            let nodes = chebyshev_nodes(n);
            assert_eq!(nodes.len(), n);
            for &x in &nodes {
                assert!(x >= -1.0 && x <= 1.0, "node {x} out of [-1, 1]");
            }
        }
    }

    #[test]
    fn nodes_symmetric() {
        let nodes = chebyshev_nodes(8);
        for i in 0..4 {
            assert!(
                (nodes[i] + nodes[7 - i]).abs() < 1e-14,
                "nodes not symmetric: {} vs {}",
                nodes[i],
                nodes[7 - i]
            );
        }
    }

    #[test]
    fn coefficients_constant_function() {
        // f(x) = 5.0 on all nodes → c_0 = 10.0 (because series is c_0/2 + ...), c_k = 0
        let n = 8;
        let values: Vec<f64> = vec![5.0; n];
        let coeffs = chebyshev_coefficients(&values);
        assert!((coeffs[0] - 10.0).abs() < 1e-12, "c_0 = {}", coeffs[0]);
        for k in 1..n {
            assert!(coeffs[k].abs() < 1e-12, "c_{k} = {}", coeffs[k]);
        }
    }

    #[test]
    fn coefficients_t1() {
        // f(x) = x = T_1(x) → c_1 = 1.0, rest zero (except c_0 for convention)
        let n = 8;
        let nodes = chebyshev_nodes(n);
        let values: Vec<f64> = nodes.clone(); // f(x) = x
        let coeffs = chebyshev_coefficients(&values);
        assert!(coeffs[0].abs() < 1e-12, "c_0 = {}", coeffs[0]);
        assert!((coeffs[1] - 1.0).abs() < 1e-12, "c_1 = {}", coeffs[1]);
        for k in 2..n {
            assert!(coeffs[k].abs() < 1e-12, "c_{k} = {}", coeffs[k]);
        }
    }

    #[test]
    fn coefficients_t2() {
        // f(x) = 2x² - 1 = T_2(x) → c_2 = 1.0
        let n = 8;
        let nodes = chebyshev_nodes(n);
        let values: Vec<f64> = nodes.iter().map(|&x| 2.0 * x * x - 1.0).collect();
        let coeffs = chebyshev_coefficients(&values);
        assert!(coeffs[0].abs() < 1e-12, "c_0 = {}", coeffs[0]);
        assert!(coeffs[1].abs() < 1e-12, "c_1 = {}", coeffs[1]);
        assert!((coeffs[2] - 1.0).abs() < 1e-12, "c_2 = {}", coeffs[2]);
        for k in 3..n {
            assert!(coeffs[k].abs() < 1e-12, "c_{k} = {}", coeffs[k]);
        }
    }

    #[test]
    fn clenshaw_constant() {
        let coeffs = [6.0]; // c_0/2 = 3.0
        assert!((clenshaw_eval(&coeffs, 0.5) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn clenshaw_linear() {
        // c_0 = 0, c_1 = 1 → f(x) = T_1(x) = x
        let coeffs = [0.0, 1.0];
        assert!((clenshaw_eval(&coeffs, 0.7) - 0.7).abs() < 1e-14);
    }

    #[test]
    fn clenshaw_t2() {
        // c_0 = 0, c_1 = 0, c_2 = 1 → f(x) = T_2(x) = 2x²-1
        let coeffs = [0.0, 0.0, 1.0];
        let t = 0.6;
        let expected = 2.0 * t * t - 1.0;
        assert!(
            (clenshaw_eval(&coeffs, t) - expected).abs() < 1e-14,
            "got {}",
            clenshaw_eval(&coeffs, t)
        );
    }

    #[test]
    fn round_trip_values_to_coeffs_to_eval() {
        // Evaluate sin at Chebyshev nodes, compute coefficients, then re-evaluate
        let n = 16;
        let nodes = chebyshev_nodes(n);
        let values: Vec<f64> = nodes.iter().map(|x| x.sin()).collect();
        let coeffs = chebyshev_coefficients(&values);

        for &node in &nodes {
            let reconstructed = clenshaw_eval(&coeffs, node);
            let original = node.sin();
            assert!(
                (reconstructed - original).abs() < 1e-12,
                "at {node}: got {reconstructed}, expected {original}"
            );
        }
    }

    #[test]
    fn round_trip_exp() {
        // exp(x) on [-1, 1] — well-approximated by degree 16
        let n = 20;
        let nodes = chebyshev_nodes(n);
        let values: Vec<f64> = nodes.iter().map(|x| x.exp()).collect();
        let coeffs = chebyshev_coefficients(&values);

        // Evaluate at non-node points
        for t in [-0.9, -0.5, 0.0, 0.3, 0.8] {
            let approx = clenshaw_eval(&coeffs, t);
            let exact = t.exp();
            assert!(
                (approx - exact).abs() < 1e-13,
                "at {t}: got {approx}, expected {exact}, err = {}",
                (approx - exact).abs()
            );
        }
    }

    #[test]
    fn derivative_coefficients_of_t1() {
        // T_1(x) = x → d/dx = 1 → coefficients [2.0] (c_0 = 2 so c_0/2 = 1)
        let coeffs = [0.0, 1.0];
        let dc = derivative_coefficients(&coeffs);
        assert_eq!(dc.len(), 1);
        assert!((dc[0] - 2.0).abs() < 1e-14, "dc[0] = {}", dc[0]);
    }

    #[test]
    fn derivative_coefficients_of_t2() {
        // T_2(x) = 2x²-1 → d/dx = 4x = 4·T_1(x) → c_1 = 4
        let coeffs = [0.0, 0.0, 1.0];
        let dc = derivative_coefficients(&coeffs);
        assert_eq!(dc.len(), 2);
        assert!(dc[0].abs() < 1e-14, "dc[0] = {}", dc[0]);
        assert!((dc[1] - 4.0).abs() < 1e-14, "dc[1] = {}", dc[1]);
    }

    #[test]
    fn derivative_of_sin_approximation() {
        // Fit sin(x), differentiate coefficients, compare derivative against cos(x)
        let n = 20;
        let nodes = chebyshev_nodes(n);
        let values: Vec<f64> = nodes.iter().map(|x| x.sin()).collect();
        let coeffs = chebyshev_coefficients(&values);
        let dc = derivative_coefficients(&coeffs);

        for t in [-0.8, -0.3, 0.0, 0.5, 0.9] {
            let approx_deriv = clenshaw_eval(&dc, t);
            let exact_deriv = t.cos();
            assert!(
                (approx_deriv - exact_deriv).abs() < 1e-11,
                "d/dx sin at {t}: got {approx_deriv}, expected {exact_deriv}"
            );
        }
    }

    #[test]
    fn derivative_of_constant_is_zero() {
        let coeffs = [4.0];
        let dc = derivative_coefficients(&coeffs);
        assert_eq!(dc.len(), 1);
        assert!((dc[0]).abs() < 1e-14);
    }

    #[test]
    fn linear_interpolate_exact_at_data_points() {
        let x_data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();
        let result = linear_interpolate(&x_data, &y_data, &x_data);
        for (i, (&r, &y)) in result.iter().zip(y_data.iter()).enumerate() {
            assert!(
                (r - y).abs() < 1e-12,
                "at x_data[{i}]: got {r}, expected {y}"
            );
        }
    }

    #[test]
    fn linear_interpolate_midpoints() {
        // Linear function: interpolation should be exact
        let x_data = vec![0.0, 1.0, 2.0, 3.0];
        let y_data = vec![0.0, 2.0, 4.0, 6.0]; // f(x) = 2x
        let x_eval = vec![0.5, 1.5, 2.5];
        let result = linear_interpolate(&x_data, &y_data, &x_eval);
        for (&x, &r) in x_eval.iter().zip(result.iter()) {
            let expected = 2.0 * x;
            assert!(
                (r - expected).abs() < 1e-12,
                "at {x}: got {r}, expected {expected}"
            );
        }
    }
}
