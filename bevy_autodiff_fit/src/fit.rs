//! Fitting pipelines: dense (DCT) and sparse (least-squares QR).

use bevy_autodiff::Float;

use crate::chebyshev;
use crate::error::FitError;
use crate::piecewise::PiecewiseFit;
use crate::reliability::{self, SegmentReliability};

/// A single Chebyshev segment: coefficients on a domain [a, b].
///
/// The Chebyshev series uses the convention:
///   f(x) ≈ c_0/2 + Σ_{k=1}^{N} c_k T_k(t)
/// where t = (2x - (a+b)) / (b-a) maps [a, b] → [-1, 1].
#[derive(Clone, Debug)]
pub struct ChebyshevSegment<F: Float> {
    /// Chebyshev coefficients c_0, ..., c_N.
    pub coeffs: Vec<F>,
    /// Left endpoint of the domain.
    pub a: F,
    /// Right endpoint of the domain.
    pub b: F,
}

impl<F: Float> ChebyshevSegment<F> {
    /// Map x from [a, b] to t in [-1, 1].
    pub fn map_to_unit(&self, x: F) -> F {
        let two = F::from_f64(2.0);
        (two * x - self.a - self.b) / (self.b - self.a)
    }

    /// Evaluate the Chebyshev series at x ∈ [a, b].
    pub fn eval(&self, x: F) -> F {
        let t = self.map_to_unit(x);
        clenshaw_eval_generic(&self.coeffs, t)
    }
}

/// Options controlling the fitting process.
#[derive(Clone, Debug)]
pub struct FitOptions {
    /// Polynomial degree (same for all segments). Must be >= 1.
    pub degree: usize,
}

/// The result of a fitting operation: fit + reliability metadata.
#[derive(Clone, Debug)]
pub struct FitResult<F: Float> {
    /// The piecewise fit.
    pub fit: PiecewiseFit<F>,
    /// Reliability metadata per segment.
    pub reliability: Vec<SegmentReliability>,
}

/// Fit dense or regularly sampled data with user-specified breakpoints.
///
/// For each segment [a_i, a_{i+1}]:
/// 1. Maps Chebyshev nodes from [-1, 1] to the segment domain
/// 2. Resamples data onto these nodes via barycentric interpolation
/// 3. Computes Chebyshev coefficients via discrete cosine transform
///
/// # Errors
///
/// Returns `FitError` if:
/// - `x_data` and `y_data` have different lengths
/// - fewer than 2 data points
/// - breakpoints are not strictly increasing
/// - degree < 1
pub fn fit_dense(
    x_data: &[f64],
    y_data: &[f64],
    breakpoints: &[f64],
    options: &FitOptions,
) -> Result<FitResult<f64>, FitError> {
    // Validate inputs
    if x_data.len() != y_data.len() {
        return Err(FitError::LengthMismatch {
            x_len: x_data.len(),
            y_len: y_data.len(),
        });
    }
    if x_data.len() < 2 {
        return Err(FitError::InsufficientData {
            min: 2,
            got: x_data.len(),
        });
    }
    if options.degree < 1 {
        return Err(FitError::InvalidDegree(options.degree));
    }
    validate_breakpoints(breakpoints)?;

    let n_segments = breakpoints.len() - 1;
    let n_nodes = options.degree + 1;

    let mut segments = Vec::with_capacity(n_segments);
    let mut reliabilities = Vec::with_capacity(n_segments);

    for i in 0..n_segments {
        let a = breakpoints[i];
        let b = breakpoints[i + 1];

        // Map Chebyshev nodes from [-1, 1] to [a, b]
        let unit_nodes = chebyshev::chebyshev_nodes(n_nodes);
        let mapped_nodes: Vec<f64> = unit_nodes
            .iter()
            .map(|&t| (b - a) / 2.0 * t + (a + b) / 2.0)
            .collect();

        // Resample data onto Chebyshev nodes via linear interpolation.
        // For dense data, linear interpolation is appropriate — it avoids the
        // Runge phenomenon and has O(h²) error where h is the data spacing.
        let values_at_nodes = chebyshev::linear_interpolate(x_data, y_data, &mapped_nodes);

        // Compute Chebyshev coefficients via DCT
        let coeffs = chebyshev::chebyshev_coefficients(&values_at_nodes);

        let segment = ChebyshevSegment {
            coeffs: coeffs.clone(),
            a,
            b,
        };
        reliabilities.push(reliability::estimate_reliability(&segment));
        segments.push(segment);
    }

    Ok(FitResult {
        fit: PiecewiseFit::new(segments, breakpoints.to_vec()),
        reliability: reliabilities,
    })
}

/// Generate n equally-spaced breakpoints covering [x_min, x_max].
///
/// Returns n_segments + 1 breakpoints (n_segments intervals).
pub fn uniform_breakpoints(x_min: f64, x_max: f64, n_segments: usize) -> Vec<f64> {
    assert!(n_segments >= 1, "need at least 1 segment");
    (0..=n_segments)
        .map(|i| x_min + (x_max - x_min) * i as f64 / n_segments as f64)
        .collect()
}

/// Validate that breakpoints are strictly increasing.
pub(crate) fn validate_breakpoints(breakpoints: &[f64]) -> Result<(), FitError> {
    if breakpoints.len() < 2 {
        return Err(FitError::InvalidBreakpoints);
    }
    for w in breakpoints.windows(2) {
        if w[1] <= w[0] {
            return Err(FitError::InvalidBreakpoints);
        }
    }
    Ok(())
}

/// Fit sparse or scattered data with user-specified breakpoints.
///
/// For each segment [a_i, a_{i+1}]:
/// 1. Selects data points within the segment
/// 2. Maps them to [-1, 1]
/// 3. Builds a Chebyshev Vandermonde matrix
/// 4. Solves the least-squares problem via Householder QR
///
/// This is the appropriate path when data is scattered (not on a regular grid)
/// or when there are too few points per segment for the DCT path.
///
/// # Errors
///
/// Returns `FitError` if:
/// - `x_data` and `y_data` have different lengths
/// - fewer than 2 data points total
/// - breakpoints are not strictly increasing
/// - degree < 1
/// - any segment has fewer data points than required (degree + 1)
pub fn fit_sparse(
    x_data: &[f64],
    y_data: &[f64],
    breakpoints: &[f64],
    options: &FitOptions,
) -> Result<FitResult<f64>, FitError> {
    // Validate inputs
    if x_data.len() != y_data.len() {
        return Err(FitError::LengthMismatch {
            x_len: x_data.len(),
            y_len: y_data.len(),
        });
    }
    if x_data.len() < 2 {
        return Err(FitError::InsufficientData {
            min: 2,
            got: x_data.len(),
        });
    }
    if options.degree < 1 {
        return Err(FitError::InvalidDegree(options.degree));
    }
    validate_breakpoints(breakpoints)?;

    let n_segments = breakpoints.len() - 1;
    let n_basis = options.degree + 1;

    let mut segments = Vec::with_capacity(n_segments);
    let mut reliabilities = Vec::with_capacity(n_segments);

    for i in 0..n_segments {
        let a = breakpoints[i];
        let b = breakpoints[i + 1];

        // Collect data points in this segment
        // Points on the left boundary belong to this segment (except the very first breakpoint)
        // Points on the right boundary belong to the next segment (except the last)
        let mut seg_x = Vec::new();
        let mut seg_y = Vec::new();
        for j in 0..x_data.len() {
            let x = x_data[j];
            let in_segment = if i == n_segments - 1 {
                x >= a && x <= b // last segment includes right endpoint
            } else {
                x >= a && x < b
            };
            if in_segment {
                seg_x.push(x);
                seg_y.push(y_data[j]);
            }
        }

        if seg_x.len() < n_basis {
            return Err(FitError::InsufficientData {
                min: n_basis,
                got: seg_x.len(),
            });
        }

        // Map data points to [-1, 1]
        let t_points: Vec<f64> = seg_x.iter().map(|&x| (2.0 * x - a - b) / (b - a)).collect();

        // Build Chebyshev Vandermonde matrix: V[i][k] = T_k(t_i)
        let vandermonde = chebyshev_vandermonde(&t_points, options.degree);

        // Solve least-squares: min ||V * c_raw - y||² via Householder QR
        let c_raw = householder_qr_solve(&vandermonde, &seg_y);

        // Convert from raw basis to Clenshaw convention:
        // Clenshaw uses f(x) = c_0/2 + Σ c_k T_k, so c_0 = 2*c_raw_0
        let mut coeffs = c_raw;
        coeffs[0] *= 2.0;

        let segment = ChebyshevSegment { coeffs, a, b };
        reliabilities.push(reliability::estimate_reliability(&segment));
        segments.push(segment);
    }

    Ok(FitResult {
        fit: PiecewiseFit::new(segments, breakpoints.to_vec()),
        reliability: reliabilities,
    })
}

/// Build the Chebyshev Vandermonde matrix.
///
/// V[i][k] = T_k(t_i) for i = 0..m, k = 0..n
/// where T_k is the k-th Chebyshev polynomial of the first kind.
///
/// Returns a row-major matrix as Vec<Vec<f64>> (m rows, n+1 columns).
pub(crate) fn chebyshev_vandermonde(t_points: &[f64], degree: usize) -> Vec<Vec<f64>> {
    let m = t_points.len();
    let n = degree + 1;

    let mut v = vec![vec![0.0; n]; m];

    for (i, &t) in t_points.iter().enumerate() {
        // T_0 = 1
        v[i][0] = 1.0;
        if n > 1 {
            // T_1 = t
            v[i][1] = t;
        }
        // T_{k+1} = 2t * T_k - T_{k-1}
        for k in 2..n {
            v[i][k] = 2.0 * t * v[i][k - 1] - v[i][k - 2];
        }
    }

    v
}

/// Solve the least-squares problem min ||A*x - b||² via Householder QR.
///
/// A is m×n with m >= n. Returns the n-element solution vector.
///
/// Algorithm: factor A = Q*R via Householder reflections, then solve R*x = Q^T*b
/// by back-substitution.
#[allow(clippy::needless_range_loop)] // col indexes 2nd dimension r[row][col], not r itself
pub(crate) fn householder_qr_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let m = a.len();
    let n = a[0].len();
    assert!(m >= n, "need at least as many rows as columns");
    assert_eq!(b.len(), m);

    // Copy A and b so we can modify in-place
    let mut r: Vec<Vec<f64>> = a.to_vec();
    let mut qt_b: Vec<f64> = b.to_vec();

    for k in 0..n {
        // Extract column k below the diagonal
        let mut col_norm_sq = 0.0;
        for row in &r[k..m] {
            col_norm_sq += row[k] * row[k];
        }
        let col_norm = col_norm_sq.sqrt();

        if col_norm < 1e-15 {
            continue; // skip near-zero column
        }

        // Householder vector v: v = x + sign(x_k)*||x||*e_k
        let sign = if r[k][k] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * col_norm;

        // Build v (stored in-place below diagonal of r, conceptually)
        let mut v = vec![0.0; m - k];
        v[0] = r[k][k] - alpha;
        for i in 1..v.len() {
            v[i] = r[k + i][k];
        }

        let v_norm_sq: f64 = v.iter().map(|&vi| vi * vi).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }

        let scale = 2.0 / v_norm_sq;

        // Apply reflection to remaining columns of R
        for col in k..n {
            let dot: f64 = v
                .iter()
                .enumerate()
                .map(|(i, &vi)| vi * r[k + i][col])
                .sum();
            let factor = scale * dot;
            for (i, &vi) in v.iter().enumerate() {
                r[k + i][col] -= factor * vi;
            }
        }

        // Apply reflection to Q^T * b
        let dot: f64 = (0..v.len()).map(|i| v[i] * qt_b[k + i]).sum();
        let factor = scale * dot;
        for i in 0..v.len() {
            qt_b[k + i] -= factor * v[i];
        }
    }

    // Back-substitution: R * x = qt_b (upper n×n part of R)
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        let mut sum = qt_b[k];
        for j in (k + 1)..n {
            sum -= r[k][j] * x[j];
        }
        if r[k][k].abs() > 1e-15 {
            x[k] = sum / r[k][k];
        }
    }

    x
}

/// Generic Clenshaw evaluation for any Float type.
pub(crate) fn clenshaw_eval_generic<F: Float>(coeffs: &[F], t: F) -> F {
    if coeffs.is_empty() {
        return F::zero();
    }
    if coeffs.len() == 1 {
        return coeffs[0] / F::from_f64(2.0);
    }

    let two = F::from_f64(2.0);
    let n = coeffs.len() - 1;
    let mut b_next = F::zero();
    let mut b_next2 = F::zero();

    for k in (1..=n).rev() {
        let b_curr = two * t * b_next - b_next2 + coeffs[k];
        b_next2 = b_next;
        b_next = b_curr;
    }

    t * b_next - b_next2 + coeffs[0] / two
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_breakpoints_basic() {
        let bp = uniform_breakpoints(0.0, 1.0, 4);
        assert_eq!(bp.len(), 5);
        assert!((bp[0] - 0.0).abs() < 1e-14);
        assert!((bp[1] - 0.25).abs() < 1e-14);
        assert!((bp[4] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn fit_dense_polynomial() {
        // Fit f(x) = x² on [0, 1] with a single segment and degree 4
        // Should recover exact coefficients (x² is degree 2 in power basis)
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();
        let breakpoints = vec![0.0, 1.0];
        let options = FitOptions { degree: 10 };

        let result = fit_dense(&x_data, &y_data, &breakpoints, &options).unwrap();
        assert_eq!(result.fit.num_segments(), 1);

        // Evaluate at test points (tolerance accounts for linear interpolation error)
        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let approx = result.fit.eval(x);
            let exact = x * x;
            assert!(
                (approx - exact).abs() < 1e-4,
                "at {x}: got {approx}, expected {exact}, err = {}",
                (approx - exact).abs()
            );
        }
    }

    #[test]
    fn fit_dense_sin() {
        // Fit sin(x) on [0, π] with 2 segments
        let n = 100;
        let x_data: Vec<f64> = (0..=n)
            .map(|i| std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
        let breakpoints = uniform_breakpoints(0.0, std::f64::consts::PI, 2);
        let options = FitOptions { degree: 16 };

        let result = fit_dense(&x_data, &y_data, &breakpoints, &options).unwrap();

        for &x in &[0.1, 0.5, 1.0, 2.0, 3.0] {
            let approx = result.fit.eval(x);
            let exact = x.sin();
            assert!(
                (approx - exact).abs() < 1e-4,
                "at {x}: got {approx}, expected {exact}, err = {}",
                (approx - exact).abs()
            );
        }
    }

    #[test]
    fn fit_dense_error_length_mismatch() {
        let result = fit_dense(&[1.0, 2.0], &[1.0], &[0.0, 3.0], &FitOptions { degree: 5 });
        assert!(matches!(result, Err(FitError::LengthMismatch { .. })));
    }

    #[test]
    fn fit_dense_error_insufficient_data() {
        let result = fit_dense(&[1.0], &[1.0], &[0.0, 2.0], &FitOptions { degree: 5 });
        assert!(matches!(result, Err(FitError::InsufficientData { .. })));
    }

    #[test]
    fn fit_dense_error_invalid_breakpoints() {
        let result = fit_dense(
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[1.0, 0.0],
            &FitOptions { degree: 5 },
        );
        assert!(matches!(result, Err(FitError::InvalidBreakpoints)));
    }

    #[test]
    fn fit_dense_error_invalid_degree() {
        let result = fit_dense(
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[0.0, 3.0],
            &FitOptions { degree: 0 },
        );
        assert!(matches!(result, Err(FitError::InvalidDegree(0))));
    }

    // ========================================================================
    // Sparse fitting tests
    // ========================================================================

    #[test]
    fn fit_sparse_polynomial() {
        // Fit f(x) = x² on [0, 1] with scattered data
        // QR should recover exact coefficients for a quadratic
        let x_data: Vec<f64> = vec![0.0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.85, 0.95, 1.0];
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();
        let result = fit_sparse(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 4 }).unwrap();

        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let approx = result.fit.eval(x);
            let exact = x * x;
            assert!(
                (approx - exact).abs() < 1e-10,
                "at {x}: got {approx}, expected {exact}, err = {}",
                (approx - exact).abs()
            );
        }
    }

    #[test]
    fn fit_sparse_sin() {
        // Fit sin(x) on [0, π] with 20 scattered points
        let x_data: Vec<f64> = (0..20)
            .map(|i| std::f64::consts::PI * i as f64 / 19.0)
            .collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
        let result = fit_sparse(
            &x_data,
            &y_data,
            &[0.0, std::f64::consts::PI],
            &FitOptions { degree: 10 },
        )
        .unwrap();

        for &x in &[0.3, 1.0, 1.5, 2.5] {
            let approx = result.fit.eval(x);
            let exact = x.sin();
            assert!(
                (approx - exact).abs() < 1e-4,
                "at {x}: got {approx}, expected {exact}, err = {}",
                (approx - exact).abs()
            );
        }
    }

    #[test]
    fn fit_sparse_matches_dense() {
        // Both paths should produce similar fits on the same data
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.exp()).collect();
        let bp = vec![0.0, 1.0];
        let opts = FitOptions { degree: 10 };

        let dense = fit_dense(&x_data, &y_data, &bp, &opts).unwrap();
        let sparse = fit_sparse(&x_data, &y_data, &bp, &opts).unwrap();

        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let d = dense.fit.eval(x);
            let s = sparse.fit.eval(x);
            assert!(
                (d - s).abs() < 1e-3,
                "at {x}: dense={d}, sparse={s}, diff={}",
                (d - s).abs()
            );
        }
    }

    #[test]
    fn fit_sparse_insufficient_data() {
        // 3 points for degree 5 (needs 6) — should fail
        let result = fit_sparse(
            &[0.0, 0.5, 1.0],
            &[0.0, 0.25, 1.0],
            &[0.0, 1.0],
            &FitOptions { degree: 5 },
        );
        assert!(matches!(result, Err(FitError::InsufficientData { .. })));
    }

    #[test]
    fn fit_sparse_two_segments() {
        // Fit exp(x) on [0, 2] with 2 segments
        let x_data: Vec<f64> = (0..30).map(|i| 2.0 * i as f64 / 29.0).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.exp()).collect();
        let bp = uniform_breakpoints(0.0, 2.0, 2);
        let result = fit_sparse(&x_data, &y_data, &bp, &FitOptions { degree: 8 }).unwrap();

        for &x in &[0.1, 0.5, 1.0, 1.5, 1.9] {
            let approx = result.fit.eval(x);
            let exact = x.exp();
            assert!(
                (approx - exact).abs() < 1e-4,
                "at {x}: got {approx}, expected {exact}, err = {}",
                (approx - exact).abs()
            );
        }
    }

    // ========================================================================
    // Internal helper tests
    // ========================================================================

    #[test]
    fn chebyshev_vandermonde_basic() {
        // At t=0: T_0=1, T_1=0, T_2=-1, T_3=0
        let v = chebyshev_vandermonde(&[0.0], 3);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 4);
        assert!((v[0][0] - 1.0).abs() < 1e-14);
        assert!(v[0][1].abs() < 1e-14);
        assert!((v[0][2] - (-1.0)).abs() < 1e-14);
        assert!(v[0][3].abs() < 1e-14);
    }

    #[test]
    fn householder_qr_exact_system() {
        // 3×3 system: identity matrix * x = [1, 2, 3]
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![1.0, 2.0, 3.0];
        let x = householder_qr_solve(&a, &b);
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn householder_qr_overdetermined() {
        // Fit y = 2x + 1 through 4 points (overdetermined, 2 unknowns)
        // A = [[1, 0], [1, 1], [1, 2], [1, 3]], b = [1, 3, 5, 7]
        let a = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let b = vec![1.0, 3.0, 5.0, 7.0];
        let x = householder_qr_solve(&a, &b);
        assert!(
            (x[0] - 1.0).abs() < 1e-12,
            "intercept: got {}, expected 1.0",
            x[0]
        );
        assert!(
            (x[1] - 2.0).abs() < 1e-12,
            "slope: got {}, expected 2.0",
            x[1]
        );
    }
}
