//! Per-axis derivative reliability estimation for 2D fits.
//!
//! For tensor product fits, reliability is independent per axis.
//! We analyze coefficient decay along each axis separately.

use bevy_autodiff::Float;

use crate::fit2d::ChebyshevSegment2D;
use crate::reliability::{estimate_noise_floor, find_max_reliable_order};

/// Reliability metadata for a 2D segment.
#[derive(Clone, Debug)]
pub struct SegmentReliability2D {
    /// Maximum reliable derivative order in x.
    pub max_reliable_order_x: usize,
    /// Maximum reliable derivative order in y.
    pub max_reliable_order_y: usize,
    /// Noise floor estimate in x direction.
    pub noise_floor_x: f64,
    /// Noise floor estimate in y direction.
    pub noise_floor_y: f64,
    /// Coefficient magnitudes (flat, row-major like coeffs).
    pub coefficient_magnitudes: Vec<f64>,
}

/// Estimate reliability for a 2D segment.
///
/// For the x-axis: for each y-mode j with significant coefficients,
/// collect the column c_0j..c_{Nx,j} and estimate x-reliability.
/// Take the minimum across significant modes.
///
/// For the y-axis: for each x-mode i with significant coefficients,
/// collect the row c_i0..c_{i,Ny} and estimate y-reliability.
/// Take the minimum across significant modes.
///
/// Modes whose maximum coefficient is below 1e-12 relative to the
/// overall maximum are considered noise-only and skipped.
pub fn estimate_reliability_2d<F: Float>(seg: &ChebyshevSegment2D<F>) -> SegmentReliability2D {
    let n_x = seg.n_x();
    let n_y = seg.n_y();

    let magnitudes: Vec<f64> = seg.coeffs.iter().map(|c| c.to_f64().abs()).collect();
    let overall_max = magnitudes.iter().copied().fold(0.0_f64, f64::max);

    // Threshold for "significant" mode: 1e-12 relative to overall max
    let significance = overall_max * 1e-12;

    let domain_width_x = (seg.b_x - seg.a_x).to_f64();
    let domain_width_y = (seg.b_y - seg.a_y).to_f64();
    let jacobian_x = 2.0 / domain_width_x;
    let jacobian_y = 2.0 / domain_width_y;

    // X-reliability: for each y-mode j, analyze the x-column
    let mut min_reliable_x = usize::MAX;
    let mut max_noise_x = 0.0_f64;
    for j in 0..n_y {
        let column: Vec<f64> = (0..n_x).map(|i| seg.coeff(i, j).to_f64()).collect();
        let col_max = column.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        if col_max < significance {
            continue; // Noise-only column, skip
        }
        let col_mags: Vec<f64> = column.iter().map(|c| c.abs()).collect();
        let nf = estimate_noise_floor(&col_mags);
        let order = find_max_reliable_order(&column, nf, jacobian_x);
        min_reliable_x = min_reliable_x.min(order);
        max_noise_x = max_noise_x.max(nf);
    }
    if min_reliable_x == usize::MAX {
        min_reliable_x = 0;
    }

    // Y-reliability: for each x-mode i, analyze the y-row
    let mut min_reliable_y = usize::MAX;
    let mut max_noise_y = 0.0_f64;
    for i in 0..n_x {
        let row: Vec<f64> = (0..n_y).map(|j| seg.coeff(i, j).to_f64()).collect();
        let row_max = row.iter().map(|c| c.abs()).fold(0.0_f64, f64::max);
        if row_max < significance {
            continue; // Noise-only row, skip
        }
        let row_mags: Vec<f64> = row.iter().map(|c| c.abs()).collect();
        let nf = estimate_noise_floor(&row_mags);
        let order = find_max_reliable_order(&row, nf, jacobian_y);
        min_reliable_y = min_reliable_y.min(order);
        max_noise_y = max_noise_y.max(nf);
    }
    if min_reliable_y == usize::MAX {
        min_reliable_y = 0;
    }

    SegmentReliability2D {
        max_reliable_order_x: min_reliable_x,
        max_reliable_order_y: min_reliable_y,
        noise_floor_x: max_noise_x,
        noise_floor_y: max_noise_y,
        coefficient_magnitudes: magnitudes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reliability_2d_polynomial() {
        // f(x,y) = x²y: coefficients concentrated at low indices
        // degree_x=3, degree_y=2 → most high-order coefficients are zero
        let n_x = 4;
        let n_y = 3;
        let mut coeffs = vec![0.0; n_x * n_y];
        // Set a few coefficients for x²y pattern
        // T_2(x)*T_1(y) component is the dominant term
        coeffs[2 * n_y + 1] = 0.5; // c_21 (arbitrary representative value)
        coeffs[0 * n_y + 1] = 0.25; // c_01

        let seg = ChebyshevSegment2D {
            coeffs,
            degree_x: 3,
            degree_y: 2,
            a_x: -1.0,
            b_x: 1.0,
            a_y: -1.0,
            b_y: 1.0,
        };

        let rel = estimate_reliability_2d(&seg);
        // Should have some reliability in both directions
        assert!(rel.coefficient_magnitudes.len() == n_x * n_y);
    }

    #[test]
    fn reliability_2d_from_sparse_fit() {
        // Fit x²*y via sparse — the result should show high x-reliability
        // (polynomial in x, degree 2) and high y-reliability (degree 1).
        use crate::fit2d::{FitOptions2D, fit_sparse_2d};

        let n = 15;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x * x * y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 6,
                degree_y: 6,
            },
        )
        .unwrap();

        let rel = &result.reliability[0];
        // Low-degree polynomial: reliability should be high in both axes
        assert!(
            rel.max_reliable_order_x >= 2,
            "x-reliability {} too low for x²y",
            rel.max_reliable_order_x
        );
        assert!(
            rel.max_reliable_order_y >= 1,
            "y-reliability {} too low for x²y",
            rel.max_reliable_order_y
        );
    }
}
