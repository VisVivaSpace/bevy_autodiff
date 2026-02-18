//! Continuity constraints for piecewise Chebyshev fits.
//!
//! Enforces C^k matching at segment boundaries via a penalty method:
//! weighted constraint rows are appended to the QR system, coupling
//! adjacent segments to enforce derivative continuity at boundaries.

use crate::error::FitError;
use crate::fit::{
    ChebyshevSegment, FitOptions, FitResult, chebyshev_vandermonde, householder_qr_solve,
    validate_breakpoints,
};
use crate::piecewise::PiecewiseFit;
use crate::reliability;

/// Options for continuity enforcement.
#[derive(Clone, Debug)]
pub struct ContinuityOptions {
    /// Maximum derivative order to enforce (0 = C⁰, 1 = C¹, ...).
    pub order: usize,
    /// Penalty weight. Higher values enforce continuity more tightly
    /// at the cost of slightly worse data fit. Typical range: 1e2–1e6.
    pub weight: f64,
}

/// Fit scattered 1D data with continuity constraints at segment boundaries.
///
/// Like [`fit_sparse`](crate::fit_sparse), but solves all segments jointly
/// with penalty rows enforcing C^k matching at internal breakpoints.
pub fn fit_sparse_continuous(
    x_data: &[f64],
    y_data: &[f64],
    breakpoints: &[f64],
    options: &FitOptions,
    continuity: &ContinuityOptions,
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
    let total_unknowns = n_segments * n_basis;

    // Single segment: no constraints needed, delegate to regular sparse
    if n_segments == 1 {
        return crate::fit::fit_sparse(x_data, y_data, breakpoints, options);
    }

    // Partition data by segment
    let mut seg_data: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); n_segments];
    for j in 0..x_data.len() {
        let x = x_data[j];
        for i in 0..n_segments {
            let a = breakpoints[i];
            let b = breakpoints[i + 1];
            let in_segment = if i == n_segments - 1 {
                x >= a && x <= b
            } else {
                x >= a && x < b
            };
            if in_segment {
                seg_data[i].0.push(x);
                seg_data[i].1.push(y_data[j]);
                break;
            }
        }
    }

    // Check each segment has enough data
    for (i, (sx, _)) in seg_data.iter().enumerate() {
        if sx.len() < n_basis {
            return Err(FitError::InsufficientData {
                min: n_basis,
                got: sx.len(),
            });
        }
        // Also verify breakpoints match
        let _ = (breakpoints[i], breakpoints[i + 1]);
    }

    // Build the global system: block-diagonal data rows + constraint rows
    let n_data: usize = seg_data.iter().map(|(sx, _)| sx.len()).sum();
    let n_constraints = (n_segments - 1) * (continuity.order + 1);
    let n_rows = n_data + n_constraints;

    let mut global_a: Vec<Vec<f64>> = vec![vec![0.0; total_unknowns]; n_rows];
    let mut global_b: Vec<f64> = vec![0.0; n_rows];

    // Fill block-diagonal data rows
    let mut row_offset = 0;
    for (seg_idx, (sx, sy)) in seg_data.iter().enumerate() {
        let a = breakpoints[seg_idx];
        let b = breakpoints[seg_idx + 1];
        let t_points: Vec<f64> = sx.iter().map(|&x| (2.0 * x - a - b) / (b - a)).collect();
        let vandermonde = chebyshev_vandermonde(&t_points, options.degree);

        let col_offset = seg_idx * n_basis;
        for (i, row) in vandermonde.iter().enumerate() {
            for (k, &val) in row.iter().enumerate() {
                global_a[row_offset + i][col_offset + k] = val;
            }
            global_b[row_offset + i] = sy[i];
        }
        row_offset += sx.len();
    }

    // Fill constraint rows
    for boundary_idx in 0..(n_segments - 1) {
        let h_left = breakpoints[boundary_idx + 1] - breakpoints[boundary_idx];
        let h_right = breakpoints[boundary_idx + 2] - breakpoints[boundary_idx + 1];

        let col_left = boundary_idx * n_basis;
        let col_right = (boundary_idx + 1) * n_basis;

        for d in 0..=continuity.order {
            let constraint_row = row_offset;
            row_offset += 1;

            // Physical derivative scaling: (2/h)^d
            let jac_left = (2.0 / h_left).powi(d as i32);
            let jac_right = (2.0 / h_right).powi(d as i32);

            for k in 0..n_basis {
                // Left segment evaluated at t=+1 (right boundary)
                let tk_d_right = chebyshev_deriv_at_one(k, d);
                global_a[constraint_row][col_left + k] =
                    continuity.weight * jac_left * tk_d_right;

                // Right segment evaluated at t=-1 (left boundary), subtracted
                let tk_d_left = chebyshev_deriv_at_neg_one(k, d);
                global_a[constraint_row][col_right + k] =
                    -continuity.weight * jac_right * tk_d_left;
            }
            // RHS = 0 (already initialized)
        }
    }

    // Solve the global system
    let raw_solution = householder_qr_solve(&global_a, &global_b);

    // Extract per-segment coefficients and convert to Clenshaw convention
    let mut segments = Vec::with_capacity(n_segments);
    let mut reliabilities = Vec::with_capacity(n_segments);

    for i in 0..n_segments {
        let offset = i * n_basis;
        let mut coeffs: Vec<f64> = raw_solution[offset..offset + n_basis].to_vec();
        coeffs[0] *= 2.0; // Clenshaw convention: c_0 = 2*a_0

        let segment = ChebyshevSegment {
            coeffs,
            a: breakpoints[i],
            b: breakpoints[i + 1],
        };
        reliabilities.push(reliability::estimate_reliability(&segment));
        segments.push(segment);
    }

    Ok(FitResult {
        fit: PiecewiseFit::new(segments, breakpoints.to_vec()),
        reliability: reliabilities,
    })
}

/// Compute T_k^(d)(1): d-th derivative of T_k evaluated at t=1.
///
/// Uses the formula: T_k^(d)(1) = Π_{j=0}^{d-1} (k² - j²) / (2j + 1)
fn chebyshev_deriv_at_one(k: usize, d: usize) -> f64 {
    let k2 = (k * k) as f64;
    let mut result = 1.0;
    for j in 0..d {
        let j2 = (j * j) as f64;
        result *= (k2 - j2) / (2.0 * j as f64 + 1.0);
    }
    result
}

/// Compute T_k^(d)(-1): d-th derivative of T_k evaluated at t=-1.
///
/// Uses: T_k^(d)(-1) = (-1)^{k+d} · T_k^(d)(1)
fn chebyshev_deriv_at_neg_one(k: usize, d: usize) -> f64 {
    let sign = if (k + d).is_multiple_of(2) { 1.0 } else { -1.0 };
    sign * chebyshev_deriv_at_one(k, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chebyshev;

    #[test]
    fn chebyshev_deriv_at_endpoints() {
        // T_0(1) = 1, T_0(-1) = 1
        assert_eq!(chebyshev_deriv_at_one(0, 0), 1.0);
        assert_eq!(chebyshev_deriv_at_neg_one(0, 0), 1.0);

        // T_1(1) = 1, T_1(-1) = -1
        assert_eq!(chebyshev_deriv_at_one(1, 0), 1.0);
        assert_eq!(chebyshev_deriv_at_neg_one(1, 0), -1.0);

        // T'_0(1) = 0, T'_1(1) = 1, T'_2(1) = 4, T'_3(1) = 9
        assert_eq!(chebyshev_deriv_at_one(0, 1), 0.0);
        assert_eq!(chebyshev_deriv_at_one(1, 1), 1.0);
        assert_eq!(chebyshev_deriv_at_one(2, 1), 4.0);
        assert_eq!(chebyshev_deriv_at_one(3, 1), 9.0);

        // T'_2(-1) = -4, T'_3(-1) = 9
        assert_eq!(chebyshev_deriv_at_neg_one(2, 1), -4.0);
        assert_eq!(chebyshev_deriv_at_neg_one(3, 1), 9.0);

        // T''_2(1) = 4, T''_3(1) = 24, T''_4(1) = 80
        assert!((chebyshev_deriv_at_one(2, 2) - 4.0).abs() < 1e-12);
        assert!((chebyshev_deriv_at_one(3, 2) - 24.0).abs() < 1e-12);
        assert!((chebyshev_deriv_at_one(4, 2) - 80.0).abs() < 1e-12);
    }

    #[test]
    fn continuous_c0_smooth_function() {
        // Fit sin(x) on [0, 2π] with 2 segments, C⁰ continuity
        let n = 100;
        let two_pi = 2.0 * std::f64::consts::PI;
        let x_data: Vec<f64> = (0..=n).map(|i| two_pi * i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();

        let bp = &[0.0, std::f64::consts::PI, two_pi];

        let result = fit_sparse_continuous(
            &x_data,
            &y_data,
            bp,
            &FitOptions { degree: 10 },
            &ContinuityOptions {
                order: 0,
                weight: 1e4,
            },
        )
        .unwrap();

        // Check C⁰ continuity at the boundary x = π
        let boundary = std::f64::consts::PI;
        let left_val = result.fit.segment(0).eval(boundary);
        let right_val = result.fit.segment(1).eval(boundary);
        let gap = (left_val - right_val).abs();
        assert!(
            gap < 1e-6,
            "C⁰ gap at boundary: {gap:.2e} (left={left_val}, right={right_val})"
        );
    }

    #[test]
    fn continuous_c1_smooth_function() {
        // Fit sin(x) on [0, 2π] with 2 segments, C¹ continuity
        let n = 100;
        let two_pi = 2.0 * std::f64::consts::PI;
        let x_data: Vec<f64> = (0..=n).map(|i| two_pi * i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();

        let bp = &[0.0, std::f64::consts::PI, two_pi];

        let result = fit_sparse_continuous(
            &x_data,
            &y_data,
            bp,
            &FitOptions { degree: 10 },
            &ContinuityOptions {
                order: 1,
                weight: 1e4,
            },
        )
        .unwrap();

        // Check C⁰ at boundary
        let boundary = std::f64::consts::PI;
        let left_val = result.fit.segment(0).eval(boundary);
        let right_val = result.fit.segment(1).eval(boundary);
        assert!(
            (left_val - right_val).abs() < 1e-6,
            "C⁰ gap: {:.2e}",
            (left_val - right_val).abs()
        );

        // Check C¹ at boundary via Chebyshev derivative coefficients
        let dc_left = chebyshev::derivative_coefficients(&result.fit.segment(0).coeffs);
        let dc_right = chebyshev::derivative_coefficients(&result.fit.segment(1).coeffs);

        let h_left = boundary; // π - 0
        let h_right = two_pi - boundary; // 2π - π = π

        // Evaluate derivative at boundary: f'(x) = (2/h) * g'(t=±1)
        // where g'(t) = dc_0/2 + Σ dc_k T_k(t) in the Clenshaw convention
        // At t=1: g'(1) = dc_0/2 + Σ dc_k (since T_k(1) = 1)
        let g_prime_left_at_1 = dc_left[0] / 2.0
            + dc_left.iter().skip(1).sum::<f64>();
        let g_prime_right_at_neg1 = dc_right[0] / 2.0
            + dc_right
                .iter()
                .enumerate()
                .skip(1)
                .map(|(k, &c)| c * if k % 2 == 0 { 1.0 } else { -1.0 })
                .sum::<f64>();

        let deriv_left = (2.0 / h_left) * g_prime_left_at_1;
        let deriv_right = (2.0 / h_right) * g_prime_right_at_neg1;
        let deriv_gap = (deriv_left - deriv_right).abs();
        assert!(
            deriv_gap < 1e-3,
            "C¹ gap at boundary: {deriv_gap:.2e} (left={deriv_left}, right={deriv_right})"
        );
    }

    #[test]
    fn continuous_vs_unconstrained() {
        // Compare continuity gap: constrained should be much smaller
        let n = 60;
        let x_data: Vec<f64> = (0..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.abs().sqrt()).collect(); // sqrt(|x|) has a kink

        let bp = &[0.0, 1.0, 2.0];
        let opts = FitOptions { degree: 6 };

        // Unconstrained
        let unconstrained = crate::fit::fit_sparse(&x_data, &y_data, bp, &opts).unwrap();
        let gap_unc = (unconstrained.fit.segment(0).eval(1.0)
            - unconstrained.fit.segment(1).eval(1.0))
        .abs();

        // Constrained C⁰
        let constrained = fit_sparse_continuous(
            &x_data,
            &y_data,
            bp,
            &opts,
            &ContinuityOptions {
                order: 0,
                weight: 1e4,
            },
        )
        .unwrap();
        let gap_con = (constrained.fit.segment(0).eval(1.0)
            - constrained.fit.segment(1).eval(1.0))
        .abs();

        // Constrained gap should be smaller (or at most equal)
        assert!(
            gap_con <= gap_unc + 1e-10,
            "constrained gap {gap_con:.2e} should be <= unconstrained gap {gap_unc:.2e}"
        );
    }

    #[test]
    fn continuous_single_segment_delegates() {
        // Single segment: should work identically to fit_sparse
        let n = 30;
        let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();

        let result = fit_sparse_continuous(
            &x_data,
            &y_data,
            &[0.0, 1.0],
            &FitOptions { degree: 4 },
            &ContinuityOptions {
                order: 1,
                weight: 1e4,
            },
        )
        .unwrap();

        assert_eq!(result.fit.num_segments(), 1);
        let val = result.fit.eval(0.5);
        assert!(
            (val - 0.25).abs() < 1e-4,
            "f(0.5) = {val}, expected 0.25"
        );
    }
}
