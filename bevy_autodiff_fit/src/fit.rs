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
fn validate_breakpoints(breakpoints: &[f64]) -> Result<(), FitError> {
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
}
