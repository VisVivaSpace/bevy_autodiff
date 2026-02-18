//! 2D tensor product Chebyshev fitting.
//!
//! Extends the 1D fitting pipeline to 2D: f(x,y) ≈ Σ_ij c_ij T_i(t_x) T_j(t_y).
//! Coefficients are stored row-major: `coeffs[i * n_y + j] = c_ij`.

use bevy_autodiff::Float;

use crate::chebyshev;
use crate::error::FitError;
use crate::fit::{self, clenshaw_eval_generic};
use crate::piecewise2d::PiecewiseFit2D;
use crate::reliability2d::{self, SegmentReliability2D};

/// A single 2D Chebyshev segment on [a_x, b_x] × [a_y, b_y].
///
/// The tensor product Chebyshev series uses the convention:
///   f(x,y) ≈ Σ_ij c_ij T_i(t_x) T_j(t_y)
/// where each axis applies the c_0/2 halving independently, and
///   t_x = (2x - (a_x+b_x)) / (b_x-a_x)
///   t_y = (2y - (a_y+b_y)) / (b_y-a_y)
///
/// Coefficients are stored row-major: `coeffs[i * n_y + j] = c_ij`.
#[derive(Clone, Debug)]
pub struct ChebyshevSegment2D<F: Float> {
    /// Chebyshev coefficients c_00, c_01, ..., c_{Nx,Ny} (flat row-major).
    pub coeffs: Vec<F>,
    /// Polynomial degree in x (n_x = degree_x + 1 coefficients per row).
    pub degree_x: usize,
    /// Polynomial degree in y (n_y = degree_y + 1 coefficients per column).
    pub degree_y: usize,
    /// Left endpoint of x domain.
    pub a_x: F,
    /// Right endpoint of x domain.
    pub b_x: F,
    /// Left endpoint of y domain.
    pub a_y: F,
    /// Right endpoint of y domain.
    pub b_y: F,
}

impl<F: Float> ChebyshevSegment2D<F> {
    /// Number of x coefficients (degree_x + 1).
    pub fn n_x(&self) -> usize {
        self.degree_x + 1
    }

    /// Number of y coefficients (degree_y + 1).
    pub fn n_y(&self) -> usize {
        self.degree_y + 1
    }

    /// Access coefficient c_ij.
    pub fn coeff(&self, i: usize, j: usize) -> F {
        self.coeffs[i * self.n_y() + j]
    }

    /// Map x from [a_x, b_x] to t_x in [-1, 1].
    pub fn map_to_unit_x(&self, x: F) -> F {
        let two = F::from_f64(2.0);
        (two * x - self.a_x - self.b_x) / (self.b_x - self.a_x)
    }

    /// Map y from [a_y, b_y] to t_y in [-1, 1].
    pub fn map_to_unit_y(&self, y: F) -> F {
        let two = F::from_f64(2.0);
        (two * y - self.a_y - self.b_y) / (self.b_y - self.a_y)
    }

    /// Evaluate the 2D Chebyshev series at (x, y) via nested Clenshaw.
    ///
    /// For each x-mode i, evaluates inner Clenshaw in y → r_i.
    /// Then evaluates outer Clenshaw in x with r_i as coefficients.
    pub fn eval(&self, x: F, y: F) -> F {
        let t_x = self.map_to_unit_x(x);
        let t_y = self.map_to_unit_y(y);

        // Inner Clenshaw: for each x-mode i, evaluate Σ_j c_ij T_j(t_y)
        // clenshaw_eval_generic handles the c_0/2 convention in y
        let r: Vec<F> = (0..self.n_x())
            .map(|i| {
                let row_start = i * self.n_y();
                let row = &self.coeffs[row_start..row_start + self.n_y()];
                clenshaw_eval_generic(row, t_y)
            })
            .collect();

        // Outer Clenshaw: evaluate Σ_i r_i T_i(t_x)
        // The r_i values from inner Clenshaw already account for the y-halving convention.
        // The outer Clenshaw naturally applies x-halving to r_0, which correctly
        // produces the double-halved constant term.
        clenshaw_eval_generic(&r, t_x)
    }
}

/// Options controlling the 2D fitting process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitOptions2D {
    /// Polynomial degree in x. Must be >= 1.
    pub degree_x: usize,
    /// Polynomial degree in y. Must be >= 1.
    pub degree_y: usize,
}

/// The result of a 2D fitting operation: fit + reliability metadata.
#[derive(Clone, Debug)]
pub struct FitResult2D<F: Float> {
    /// The piecewise 2D fit.
    pub fit: PiecewiseFit2D<F>,
    /// Reliability metadata per segment.
    pub reliability: Vec<SegmentReliability2D>,
}

/// Fit dense 2D data on a rectangular grid.
///
/// Data is on a rectangular grid: `z_data[iy][ix] = f(x_data[ix], y_data[iy])`.
/// Uses a separable approach:
/// 1. For each row (fixed y_j): resample in x → Chebyshev nodes, DCT
/// 2. For each x-mode i: resample the intermediate coefficients in y → DCT
///
/// # Errors
///
/// Returns `FitError` if:
/// - `z_data` dimensions don't match `x_data.len()` × `y_data.len()`
/// - fewer than 2 data points in either direction
/// - breakpoints are not strictly increasing
/// - degree < 1 in either direction
pub fn fit_dense_2d(
    x_data: &[f64],
    y_data: &[f64],
    z_data: &[Vec<f64>],
    breakpoints_x: &[f64],
    breakpoints_y: &[f64],
    options: &FitOptions2D,
) -> Result<FitResult2D<f64>, FitError> {
    // Validate inputs
    if x_data.len() < 2 {
        return Err(FitError::InsufficientData {
            min: 2,
            got: x_data.len(),
        });
    }
    if y_data.len() < 2 {
        return Err(FitError::InsufficientData {
            min: 2,
            got: y_data.len(),
        });
    }
    if options.degree_x < 1 {
        return Err(FitError::InvalidDegree(options.degree_x));
    }
    if options.degree_y < 1 {
        return Err(FitError::InvalidDegree(options.degree_y));
    }
    if z_data.len() != y_data.len() {
        return Err(FitError::GridDimensionMismatch {
            rows: z_data.len(),
            cols: if z_data.is_empty() {
                0
            } else {
                z_data[0].len()
            },
            expected_rows: y_data.len(),
            expected_cols: x_data.len(),
        });
    }
    for row in z_data {
        if row.len() != x_data.len() {
            return Err(FitError::GridDimensionMismatch {
                rows: z_data.len(),
                cols: row.len(),
                expected_rows: y_data.len(),
                expected_cols: x_data.len(),
            });
        }
    }
    fit::validate_breakpoints(breakpoints_x)?;
    fit::validate_breakpoints(breakpoints_y)?;

    let n_seg_x = breakpoints_x.len() - 1;
    let n_seg_y = breakpoints_y.len() - 1;
    let n_x = options.degree_x + 1;
    let n_y = options.degree_y + 1;

    let mut segments = Vec::with_capacity(n_seg_x * n_seg_y);
    let mut reliabilities = Vec::with_capacity(n_seg_x * n_seg_y);

    for iy in 0..n_seg_y {
        let ay = breakpoints_y[iy];
        let by = breakpoints_y[iy + 1];

        for ix in 0..n_seg_x {
            let ax = breakpoints_x[ix];
            let bx = breakpoints_x[ix + 1];

            // Step 1: For each data row (fixed y_j), resample in x and DCT
            // Chebyshev nodes in x mapped to [ax, bx]
            let x_nodes = chebyshev::chebyshev_nodes(n_x);
            let x_mapped: Vec<f64> = x_nodes
                .iter()
                .map(|&t| (bx - ax) / 2.0 * t + (ax + bx) / 2.0)
                .collect();

            // For each y data row, resample onto x Chebyshev nodes and compute x-coefficients
            let mut intermediate = Vec::with_capacity(y_data.len());
            for (jy, row) in z_data.iter().enumerate() {
                let y_val = y_data[jy];
                // Only process rows within the y segment
                // Use tolerance relative to segment width to handle domains of different scales
                let y_tol = (by - ay) * 1e-12;
                if y_val < ay - y_tol || y_val > by + y_tol {
                    continue;
                }
                let resampled = chebyshev::linear_interpolate(x_data, row, &x_mapped);
                let x_coeffs = chebyshev::chebyshev_coefficients(&resampled);
                intermediate.push((y_val, x_coeffs));
            }

            // Need at least 2 rows for linear interpolation in y
            if intermediate.len() < 2 {
                return Err(FitError::InsufficientData {
                    min: 2,
                    got: intermediate.len(),
                });
            }

            // Step 2: For each x-mode i, collect intermediate coefficients across y,
            // resample in y, and DCT
            let y_nodes = chebyshev::chebyshev_nodes(n_y);
            let y_mapped: Vec<f64> = y_nodes
                .iter()
                .map(|&t| (by - ay) / 2.0 * t + (ay + by) / 2.0)
                .collect();

            let mut coeffs = vec![0.0; n_x * n_y];

            for i in 0..n_x {
                // Collect the i-th x-coefficient as a function of y
                let inter_y: Vec<f64> = intermediate.iter().map(|(y, _)| *y).collect();
                let inter_vals: Vec<f64> = intermediate.iter().map(|(_, xc)| xc[i]).collect();

                // Resample onto y Chebyshev nodes
                let resampled_y = chebyshev::linear_interpolate(&inter_y, &inter_vals, &y_mapped);
                let y_coeffs = chebyshev::chebyshev_coefficients(&resampled_y);

                // Store in row-major: coeffs[i * n_y + j]
                for (j, &c) in y_coeffs.iter().enumerate() {
                    coeffs[i * n_y + j] = c;
                }
            }

            let seg = ChebyshevSegment2D {
                coeffs,
                degree_x: options.degree_x,
                degree_y: options.degree_y,
                a_x: ax,
                b_x: bx,
                a_y: ay,
                b_y: by,
            };
            reliabilities.push(reliability2d::estimate_reliability_2d(&seg));
            segments.push(seg);
        }
    }

    Ok(FitResult2D {
        fit: PiecewiseFit2D::new(
            segments,
            breakpoints_x.to_vec(),
            breakpoints_y.to_vec(),
            n_seg_x,
            n_seg_y,
        ),
        reliability: reliabilities,
    })
}

/// Fit scattered 2D data via least-squares.
///
/// For scattered (x_i, y_i, z_i) data. Builds a 2D Chebyshev Vandermonde matrix
/// and solves via Householder QR.
///
/// # Arguments
///
/// - `x_data`, `y_data`, `z_data`: coordinate triples, same length
/// - `x_range`: [a_x, b_x] domain in x
/// - `y_range`: [a_y, b_y] domain in y
///
/// # Errors
///
/// Returns `FitError` if arrays have different lengths, too few points,
/// or invalid degree.
pub fn fit_sparse_2d(
    x_data: &[f64],
    y_data: &[f64],
    z_data: &[f64],
    x_range: [f64; 2],
    y_range: [f64; 2],
    options: &FitOptions2D,
) -> Result<FitResult2D<f64>, FitError> {
    if x_data.len() != y_data.len() || x_data.len() != z_data.len() {
        return Err(FitError::ScatterLengthMismatch {
            x_len: x_data.len(),
            y_len: y_data.len(),
            z_len: z_data.len(),
        });
    }
    if options.degree_x < 1 {
        return Err(FitError::InvalidDegree(options.degree_x));
    }
    if options.degree_y < 1 {
        return Err(FitError::InvalidDegree(options.degree_y));
    }

    let n_x = options.degree_x + 1;
    let n_y = options.degree_y + 1;
    let n_coeffs = n_x * n_y;

    if x_data.len() < n_coeffs {
        return Err(FitError::InsufficientData {
            min: n_coeffs,
            got: x_data.len(),
        });
    }

    let ax = x_range[0];
    let bx = x_range[1];
    let ay = y_range[0];
    let by = y_range[1];

    // Map data to [-1, 1] in each dimension
    let t_x: Vec<f64> = x_data
        .iter()
        .map(|&x| (2.0 * x - ax - bx) / (bx - ax))
        .collect();
    let t_y: Vec<f64> = y_data
        .iter()
        .map(|&y| (2.0 * y - ay - by) / (by - ay))
        .collect();

    // Build 2D Chebyshev Vandermonde matrix
    // Row per data point, column per (i, j) coefficient pair
    // Column index = i * n_y + j
    // Entry = T_i(t_x) * T_j(t_y)
    let m = x_data.len();
    let mut vandermonde = vec![vec![0.0; n_coeffs]; m];

    for p in 0..m {
        // Evaluate all T_i(t_x[p]) and T_j(t_y[p])
        let ti_x = chebyshev_values(t_x[p], options.degree_x);
        let tj_y = chebyshev_values(t_y[p], options.degree_y);

        for i in 0..n_x {
            for j in 0..n_y {
                vandermonde[p][i * n_y + j] = ti_x[i] * tj_y[j];
            }
        }
    }

    // Solve least-squares: min ||V * c_raw - z||²
    let c_raw = fit::householder_qr_solve(&vandermonde, z_data);

    // Convert from raw basis to Clenshaw convention:
    // Raw: f = Σ c_raw_ij T_i T_j
    // Clenshaw: c_00 *= 4 (halved in both x and y), c_i0 *= 2 (halved in y only),
    //           c_0j *= 2 (halved in x only), c_ij unchanged for i>0,j>0
    let mut coeffs = c_raw;
    for i in 0..n_x {
        for j in 0..n_y {
            let idx = i * n_y + j;
            if i == 0 && j == 0 {
                coeffs[idx] *= 4.0; // halved twice
            } else if i == 0 {
                coeffs[idx] *= 2.0; // halved in x only
            } else if j == 0 {
                coeffs[idx] *= 2.0; // halved in y only
            }
        }
    }

    let seg = ChebyshevSegment2D {
        coeffs,
        degree_x: options.degree_x,
        degree_y: options.degree_y,
        a_x: ax,
        b_x: bx,
        a_y: ay,
        b_y: by,
    };
    let rel = reliability2d::estimate_reliability_2d(&seg);

    Ok(FitResult2D {
        fit: PiecewiseFit2D::new(vec![seg], vec![ax, bx], vec![ay, by], 1, 1),
        reliability: vec![rel],
    })
}

/// Evaluate all Chebyshev polynomials T_0(t) through T_n(t).
fn chebyshev_values(t: f64, degree: usize) -> Vec<f64> {
    let n = degree + 1;
    let mut vals = vec![0.0; n];
    vals[0] = 1.0;
    if n > 1 {
        vals[1] = t;
    }
    for k in 2..n {
        vals[k] = 2.0 * t * vals[k - 1] - vals[k - 2];
    }
    vals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_2d_eval_constant() {
        // f(x,y) = 5.0 everywhere
        // In Clenshaw convention: c_00/2 in y, then /2 in x → c_00 = 20.0
        let seg: ChebyshevSegment2D<f64> = ChebyshevSegment2D {
            coeffs: vec![20.0],
            degree_x: 0,
            degree_y: 0,
            a_x: 0.0,
            b_x: 1.0,
            a_y: 0.0,
            b_y: 1.0,
        };
        let val = seg.eval(0.5, 0.5);
        assert!((val - 5.0).abs() < 1e-14, "expected 5.0, got {val}");
    }

    #[test]
    fn segment_2d_eval_linear_x() {
        // f(x,y) = x on [-1,1]×[-1,1]
        // T_1(t_x) = t_x = x (identity map on [-1,1])
        // Coefficients: c_00 = 0, c_10 = 1 (with c_0j/2 convention)
        // In Clenshaw convention c_10 row: c_10 = 1 for the T_1 coeff
        // c_00 row: c_00 = 0
        // Layout: [c_00, c_10] = [0.0, 1.0] (n_x=2, n_y=1, degree_x=1, degree_y=0)
        // Inner Clenshaw(y) for row i=0: coeffs=[0.0] → 0/2 = 0
        // Inner Clenshaw(y) for row i=1: coeffs=[1.0] → 1/2 = 0.5
        // r = [0.0, 0.5]
        // r_for_outer = [0.0, 0.5] (r_0 doubled: 0*2=0)
        // Outer Clenshaw: coeffs=[0.0, 0.5], t_x=x
        //   b1 = 0.5 (from k=1: c_1 = 0.5)
        //   result = t_x * 0.5 - 0 + 0.0/2 = 0.5*t_x
        // That gives 0.5*x, not x. The issue is the y convention.
        //
        // For n_y=1, the inner Clenshaw returns c_0/2. So if we want
        // the inner result to be the actual coefficient value (not halved),
        // we'd need c_00 = 0*2 = 0, c_10 = 1*2 = 2.
        //
        // Let's verify: with c_10=2:
        // Inner(i=1): clenshaw([2.0], t_y) = 2/2 = 1
        // r = [0.0, 1.0]
        // r_for_outer = [0.0, 1.0]
        // Outer: coeffs=[0.0, 1.0], result = t_x*1 + 0 = t_x = x ✓
        let seg: ChebyshevSegment2D<f64> = ChebyshevSegment2D {
            coeffs: vec![0.0, 2.0],
            degree_x: 1,
            degree_y: 0,
            a_x: -1.0,
            b_x: 1.0,
            a_y: -1.0,
            b_y: 1.0,
        };

        assert!((seg.eval(0.5, 0.3) - 0.5).abs() < 1e-14);
        assert!((seg.eval(-0.7, 0.0) - (-0.7)).abs() < 1e-14);
    }

    #[test]
    fn segment_2d_eval_linear_y() {
        // f(x,y) = y on [-1,1]×[-1,1]
        // Raw: a_01 = 1 (coefficient of T_0(x)*T_1(y))
        // Convention: i=0 row is doubled → c_01 = 2*a_01 = 2
        // c_00 = 4*a_00 = 0
        // Layout: [c_00, c_01] = [0.0, 2.0] (n_x=1, n_y=2)
        // Inner(i=0): clenshaw([0.0, 2.0], t_y) = t_y*2 + 0/2 = 2y
        // Outer: clenshaw([2y], t_x) = 2y/2 = y ✓
        let seg: ChebyshevSegment2D<f64> = ChebyshevSegment2D {
            coeffs: vec![0.0, 2.0],
            degree_x: 0,
            degree_y: 1,
            a_x: -1.0,
            b_x: 1.0,
            a_y: -1.0,
            b_y: 1.0,
        };

        assert!((seg.eval(0.3, 0.5) - 0.5).abs() < 1e-14);
        assert!((seg.eval(0.0, -0.7) - (-0.7)).abs() < 1e-14);
    }

    #[test]
    fn segment_2d_eval_x_times_y() {
        // f(x,y) = x*y on [-1,1]×[-1,1]
        // = T_1(t_x) * T_1(t_y)
        // Raw basis: c_11 = 1, all others 0
        // Clenshaw convention: c_11 stays 1 (both i>0, j>0)
        // Layout (n_x=2, n_y=2): [c_00, c_01, c_10, c_11] = [0, 0, 0, 1]
        let seg: ChebyshevSegment2D<f64> = ChebyshevSegment2D {
            coeffs: vec![0.0, 0.0, 0.0, 1.0],
            degree_x: 1,
            degree_y: 1,
            a_x: -1.0,
            b_x: 1.0,
            a_y: -1.0,
            b_y: 1.0,
        };

        assert!((seg.eval(0.5, 0.3) - 0.15).abs() < 1e-14);
        assert!((seg.eval(-0.5, 0.4) - (-0.2)).abs() < 1e-14);
        assert!((seg.eval(1.0, 1.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn sparse_2d_polynomial_recovery() {
        // f(x,y) = 3x² + 2xy - y + 5 on [-1,1]×[-1,1]
        // This is degree 2 in x, degree 1 in y. Sparse QR with degree (2,1) should
        // recover exact coefficients (Tier 2).
        let n = 20;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        let mut z_data = Vec::new();

        for i in 0..n {
            for j in 0..n {
                let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
                let y = -1.0 + 2.0 * j as f64 / (n - 1) as f64;
                x_data.push(x);
                y_data.push(y);
                z_data.push(3.0 * x * x + 2.0 * x * y - y + 5.0);
            }
        }

        let result = fit_sparse_2d(
            &x_data,
            &y_data,
            &z_data,
            [-1.0, 1.0],
            [-1.0, 1.0],
            &FitOptions2D {
                degree_x: 3,
                degree_y: 2,
            },
        )
        .unwrap();

        // Tier 2: exact polynomial recovery, near machine epsilon
        let seg = result.fit.segment(0, 0);
        for &(x, y) in &[(0.3, 0.7), (-0.5, 0.2), (0.0, 0.0), (0.8, -0.6)] {
            let val = seg.eval(x, y);
            let exact = 3.0 * x * x + 2.0 * x * y - y + 5.0;
            assert!(
                (val - exact).abs() < 1e-10,
                "f({x}, {y}): got {val}, expected {exact}, err = {:.2e}",
                (val - exact).abs()
            );
        }
    }

    #[test]
    fn dense_2d_separable_function() {
        // f(x,y) = sin(x) * cos(y) on [0, π] × [0, π]
        // Dense fit on a grid, then check accuracy (Tier 3)
        // h ≈ π/100 ≈ 0.031, O(h²) ≈ 1e-3
        let nx = 100;
        let ny = 100;
        let pi = std::f64::consts::PI;

        let x_data: Vec<f64> = (0..=nx).map(|i| pi * i as f64 / nx as f64).collect();
        let y_data: Vec<f64> = (0..=ny).map(|j| pi * j as f64 / ny as f64).collect();
        let z_data: Vec<Vec<f64>> = y_data
            .iter()
            .map(|&y| x_data.iter().map(|&x| x.sin() * y.cos()).collect())
            .collect();

        let result = fit_dense_2d(
            &x_data,
            &y_data,
            &z_data,
            &[0.0, pi],
            &[0.0, pi],
            &FitOptions2D {
                degree_x: 16,
                degree_y: 16,
            },
        )
        .unwrap();

        // Tier 3: O(h²) resampling error, h ≈ π/50 ≈ 0.063
        for &(x, y) in &[(0.5, 0.5), (1.0, 1.0), (2.0, 0.5), (1.5, 2.0)] {
            let val = result.fit.eval(x, y);
            let exact = x.sin() * y.cos();
            assert!(
                (val - exact).abs() < 5e-4,
                "f({x}, {y}): got {val}, expected {exact}, err = {:.2e}",
                (val - exact).abs()
            );
        }
    }

    #[test]
    fn dense_vs_sparse_2d_cross_validation() {
        // Both paths should give similar results for x*y on [0,1]×[0,1]
        let nx = 30;
        let ny = 30;
        let x_data: Vec<f64> = (0..=nx).map(|i| i as f64 / nx as f64).collect();
        let y_data: Vec<f64> = (0..=ny).map(|j| j as f64 / ny as f64).collect();
        let z_data: Vec<Vec<f64>> = y_data
            .iter()
            .map(|&y| x_data.iter().map(|&x| x * y).collect())
            .collect();

        let dense_result = fit_dense_2d(
            &x_data,
            &y_data,
            &z_data,
            &[0.0, 1.0],
            &[0.0, 1.0],
            &FitOptions2D {
                degree_x: 4,
                degree_y: 4,
            },
        )
        .unwrap();

        // Flatten grid data for sparse path
        let mut sx = Vec::new();
        let mut sy = Vec::new();
        let mut sz = Vec::new();
        for (jy, &y) in y_data.iter().enumerate() {
            for (ix, &x) in x_data.iter().enumerate() {
                sx.push(x);
                sy.push(y);
                sz.push(z_data[jy][ix]);
            }
        }

        let sparse_result = fit_sparse_2d(
            &sx,
            &sy,
            &sz,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 4,
                degree_y: 4,
            },
        )
        .unwrap();

        for &(x, y) in &[(0.3, 0.3), (0.5, 0.5), (0.7, 0.8)] {
            let dense_val = dense_result.fit.eval(x, y);
            let sparse_val = sparse_result.fit.eval(x, y);
            assert!(
                (dense_val - sparse_val).abs() < 5e-4,
                "at ({x}, {y}): dense={dense_val}, sparse={sparse_val}, diff={:.2e}",
                (dense_val - sparse_val).abs()
            );
        }
    }

    #[test]
    fn sparse_2d_error_cases() {
        let opts = FitOptions2D {
            degree_x: 2,
            degree_y: 2,
        };

        // Length mismatch
        assert!(fit_sparse_2d(&[0.0], &[0.0, 1.0], &[0.0], [0.0, 1.0], [0.0, 1.0], &opts).is_err());

        // Insufficient data
        assert!(
            fit_sparse_2d(
                &[0.0, 1.0],
                &[0.0, 1.0],
                &[0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                &opts,
            )
            .is_err()
        );

        // Invalid degree
        let bad_opts = FitOptions2D {
            degree_x: 0,
            degree_y: 2,
        };
        assert!(
            fit_sparse_2d(
                &[0.0; 20],
                &[0.0; 20],
                &[0.0; 20],
                [0.0, 1.0],
                [0.0, 1.0],
                &bad_opts,
            )
            .is_err()
        );
    }
}
