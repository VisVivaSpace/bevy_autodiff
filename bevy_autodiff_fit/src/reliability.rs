//! Derivative reliability estimation from Chebyshev coefficient decay.
//!
//! When you differentiate a fit, noise gets amplified. The rate of Chebyshev
//! coefficient decay tells us how many derivatives are trustworthy.

use crate::chebyshev;
use crate::fit::ChebyshevSegment;
use bevy_autodiff::Float;

/// Reliability metadata for a single segment.
#[derive(Clone, Debug)]
pub struct SegmentReliability {
    /// Estimated maximum derivative order with meaningful accuracy.
    ///
    /// Derivatives of order <= this value are expected to be reliable.
    /// Higher-order derivatives may be dominated by noise amplification.
    pub max_reliable_order: usize,

    /// Estimated noise floor (magnitude of the smallest significant coefficient).
    ///
    /// Coefficients below this level are likely noise rather than signal.
    pub noise_floor: f64,

    /// Chebyshev coefficient magnitudes (for user inspection).
    ///
    /// Users can examine this to understand the quality of the fit
    /// and make their own reliability judgments.
    pub coefficient_magnitudes: Vec<f64>,
}

/// Estimate the reliability of derivatives for a segment.
///
/// Algorithm:
/// 1. Compute coefficient magnitudes |c_k|
/// 2. Estimate the noise floor from the tail of the spectrum
/// 3. For each derivative order d, compute derivative coefficients
///    and check if they are above the amplified noise floor
/// 4. Report the highest order where derivatives are still meaningful
pub fn estimate_reliability<F: Float>(segment: &ChebyshevSegment<F>) -> SegmentReliability {
    let coeffs_f64: Vec<f64> = segment.coeffs.iter().map(|c| c.to_f64()).collect();
    let magnitudes: Vec<f64> = coeffs_f64.iter().map(|c| c.abs()).collect();

    if magnitudes.is_empty() {
        return SegmentReliability {
            max_reliable_order: 0,
            noise_floor: 0.0,
            coefficient_magnitudes: magnitudes,
        };
    }

    // Estimate noise floor: median of the last quarter of coefficients
    let noise_floor = estimate_noise_floor(&magnitudes);

    // Find the maximum reliable derivative order
    let domain_width = (segment.b - segment.a).to_f64();
    let jacobian = 2.0 / domain_width;
    let max_order = find_max_reliable_order(&coeffs_f64, noise_floor, jacobian);

    SegmentReliability {
        max_reliable_order: max_order,
        noise_floor,
        coefficient_magnitudes: magnitudes,
    }
}

/// Estimate the noise floor from the coefficient spectrum.
///
/// Uses the median of the last quarter of coefficients as the noise estimate.
/// For smooth functions, these high-degree coefficients are at or below the noise level.
pub(crate) fn estimate_noise_floor(magnitudes: &[f64]) -> f64 {
    let n = magnitudes.len();
    if n <= 2 {
        return magnitudes.last().copied().unwrap_or(0.0);
    }

    // Take the last quarter (at least 2 coefficients)
    let quarter = (n / 4).max(2).min(n);
    let tail_start = n - quarter;
    let mut tail: Vec<f64> = magnitudes[tail_start..].to_vec();
    tail.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Median
    if tail.len().is_multiple_of(2) {
        (tail[tail.len() / 2 - 1] + tail[tail.len() / 2]) / 2.0
    } else {
        tail[tail.len() / 2]
    }
}

/// Find the maximum derivative order where the signal is above the noise.
///
/// For each derivative order d:
/// 1. Compute derivative coefficients (applying the recurrence d times)
/// 2. Scale the noise floor by jacobian^d (derivative amplification)
/// 3. Check if the leading derivative coefficients are above the noise
pub(crate) fn find_max_reliable_order(coeffs: &[f64], noise_floor: f64, jacobian: f64) -> usize {
    if noise_floor == 0.0 {
        // Perfect data (e.g., exact polynomial) — all derivatives reliable
        // up to degree
        return coeffs.len().saturating_sub(1);
    }

    let max_c = coeffs.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
    if max_c == 0.0 {
        return 0;
    }

    let mut dc = coeffs.to_vec();
    let mut order = 0;

    loop {
        let next_dc = chebyshev::derivative_coefficients(&dc);
        if next_dc.is_empty() || next_dc.len() <= 1 {
            break;
        }

        let amplified_noise = noise_floor * jacobian.powi((order + 1) as i32);

        // Check if the derivative signal is above the amplified noise
        let max_dc = next_dc.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        let signal_to_noise = max_dc / amplified_noise;

        // Require at least 10x signal-to-noise ratio
        if signal_to_noise < 10.0 {
            break;
        }

        order += 1;
        dc = next_dc;

        // Safety limit: don't claim more derivatives than the degree
        if order >= coeffs.len() - 1 {
            break;
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reliability_exact_polynomial() {
        // Polynomial with exact Chebyshev representation — all derivatives reliable
        // Use enough coefficients that the noise floor algorithm works well
        let mut coeffs = vec![0.0; 10];
        coeffs[0] = 2.0;
        coeffs[1] = 1.0;
        coeffs[2] = 0.5;
        // Rest are exactly zero → noise floor is 0 → all derivatives reliable up to degree

        let seg = ChebyshevSegment {
            coeffs,
            a: -1.0,
            b: 1.0,
        };
        let rel = estimate_reliability(&seg);
        // With exact coefficients (no noise), noise floor should be 0
        // and all derivatives should be reliable up to the degree
        assert_eq!(
            rel.noise_floor, 0.0,
            "exact polynomial should have zero noise floor, got {}",
            rel.noise_floor
        );
        assert!(
            rel.max_reliable_order >= 2,
            "expected reliable derivatives for exact polynomial, got order {}",
            rel.max_reliable_order
        );
        assert_eq!(rel.coefficient_magnitudes.len(), 10);
    }

    #[test]
    fn reliability_noisy_data() {
        // Simulate noisy data: large c_0, decaying, then noise floor
        let mut coeffs = vec![0.0; 20];
        coeffs[0] = 10.0;
        coeffs[1] = 5.0;
        coeffs[2] = 1.0;
        coeffs[3] = 0.1;
        // Rest are at noise level
        for c in &mut coeffs[4..] {
            *c = 1e-10;
        }

        let seg = ChebyshevSegment {
            coeffs,
            a: -1.0,
            b: 1.0,
        };
        let rel = estimate_reliability(&seg);

        // Should detect the noise floor
        assert!(rel.noise_floor > 0.0);
        // The reliability should be limited (not all 19 derivatives)
        assert!(
            rel.max_reliable_order < 19,
            "expected limited reliability, got {}",
            rel.max_reliable_order
        );
    }

    #[test]
    fn reliability_smooth_function() {
        // sin(x) on [-1, 1] — coefficients decay exponentially
        use crate::chebyshev::{chebyshev_coefficients, chebyshev_nodes};

        let n = 20;
        let nodes = chebyshev_nodes(n);
        let values: Vec<f64> = nodes.iter().map(|x| x.sin()).collect();
        let coeffs = chebyshev_coefficients(&values);

        let seg = ChebyshevSegment {
            coeffs,
            a: -1.0,
            b: 1.0,
        };
        let rel = estimate_reliability(&seg);

        // sin(x) is infinitely smooth — should report high reliability
        assert!(
            rel.max_reliable_order >= 3,
            "expected high reliability for sin, got {}",
            rel.max_reliable_order
        );
    }

    #[test]
    fn noise_floor_estimation() {
        // Known noise floor
        let mags = vec![10.0, 5.0, 1.0, 0.1, 1e-8, 1e-8, 1e-8, 1e-8];
        let nf = estimate_noise_floor(&mags);
        assert!(nf < 1e-5, "noise floor should be ~1e-8, got {nf}");
    }
}
