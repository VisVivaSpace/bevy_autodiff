//! Piecewise Chebyshev fit management.
//!
//! A [`PiecewiseFit`] holds multiple [`ChebyshevSegment`]s covering a domain,
//! with automatic segment lookup and coordinate mapping.

use bevy_autodiff::Float;

use crate::chebyshev;
use crate::error::FitError;
use crate::fit::ChebyshevSegment;

/// A piecewise Chebyshev fit: multiple segments covering a contiguous domain.
///
/// Each segment is an independent Chebyshev polynomial on its own sub-interval.
/// Function values are continuous at segment boundaries (both segments agree at
/// the shared endpoint), but derivatives may be discontinuous.
#[derive(Clone, Debug)]
pub struct PiecewiseFit<F: Float> {
    segments: Vec<ChebyshevSegment<F>>,
    /// Breakpoints: len = segments.len() + 1. Strictly increasing.
    breakpoints: Vec<F>,
}

impl<F: Float> PiecewiseFit<F> {
    /// Create a new piecewise fit from segments and breakpoints.
    ///
    /// # Panics
    ///
    /// Panics if `breakpoints.len() != segments.len() + 1`.
    pub(crate) fn new(segments: Vec<ChebyshevSegment<F>>, breakpoints: Vec<F>) -> Self {
        assert_eq!(
            breakpoints.len(),
            segments.len() + 1,
            "breakpoints.len() must be segments.len() + 1"
        );
        Self {
            segments,
            breakpoints,
        }
    }

    /// Number of segments in the fit.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Access a segment by index.
    pub fn segment(&self, i: usize) -> &ChebyshevSegment<F> {
        &self.segments[i]
    }

    /// The breakpoints defining segment boundaries.
    pub fn breakpoints(&self) -> &[F] {
        &self.breakpoints
    }

    /// The left endpoint of the fitted domain.
    pub fn domain_min(&self) -> F {
        self.breakpoints[0]
    }

    /// The right endpoint of the fitted domain.
    pub fn domain_max(&self) -> F {
        *self.breakpoints.last().unwrap()
    }
}

impl<F: Float + PartialOrd> PiecewiseFit<F> {
    /// Find which segment contains x.
    ///
    /// Returns the segment index. Points exactly at an internal breakpoint
    /// belong to the right segment (the one starting at that breakpoint).
    /// The last segment includes the right endpoint.
    ///
    /// # Errors
    ///
    /// Returns `FitError::OutOfDomain` if x is outside [domain_min, domain_max].
    pub fn segment_index(&self, x: F) -> Result<usize, FitError> {
        let a = self.domain_min();
        let b = self.domain_max();

        if x < a || x > b {
            return Err(FitError::out_of_domain(x, a, b));
        }

        let n = self.segments.len();
        if n == 1 {
            return Ok(0);
        }

        // Reverse linear scan: find the rightmost breakpoint <= x
        for i in (0..n).rev() {
            if x >= self.breakpoints[i] {
                return Ok(i);
            }
        }

        Ok(0)
    }

    /// Evaluate the piecewise fit at x.
    ///
    /// Automatically selects the correct segment. Points outside the domain
    /// are clamped to the nearest segment — use [`try_eval`](Self::try_eval)
    /// to get an error instead.
    pub fn eval(&self, x: F) -> F {
        let idx = self.segment_index(x).unwrap_or_else(|_| {
            if x <= self.domain_min() {
                0
            } else {
                self.segments.len() - 1
            }
        });
        self.segments[idx].eval(x)
    }

    /// Evaluate the piecewise fit at x, returning an error if out of domain.
    ///
    /// Unlike [`eval`](Self::eval), this does not clamp out-of-domain points.
    ///
    /// # Errors
    ///
    /// Returns `FitError::OutOfDomain` if x is outside the fitted domain.
    pub fn try_eval(&self, x: F) -> Result<F, FitError> {
        let idx = self.segment_index(x)?;
        Ok(self.segments[idx].eval(x))
    }

    /// Evaluate the k-th derivative at x using the Chebyshev derivative recurrence.
    ///
    /// This is a standalone evaluation (no autodiff graph involved). The derivative
    /// coefficients are computed from the segment's Chebyshev coefficients, then
    /// evaluated via Clenshaw.
    ///
    /// The domain mapping Jacobian (2/(b-a))^order is applied automatically.
    pub fn eval_derivative(&self, x: f64, order: usize) -> f64 {
        let idx = self.segment_index_f64(x);
        let seg = &self.segments[idx];
        let a: f64 = seg.a.to_f64();
        let b: f64 = seg.b.to_f64();

        // Get f64 coefficients
        let mut coeffs: Vec<f64> = seg.coeffs.iter().map(|c| c.to_f64()).collect();

        // Apply derivative recurrence `order` times
        let jacobian = 2.0 / (b - a);
        for _ in 0..order {
            coeffs = chebyshev::derivative_coefficients(&coeffs);
        }

        // Map x to [-1, 1]
        let t = (2.0 * x - a - b) / (b - a);

        // Evaluate and scale by Jacobian^order
        chebyshev::clenshaw_eval(&coeffs, t) * jacobian.powi(order as i32)
    }

    /// Find segment index for an f64 x value (clamped to domain).
    fn segment_index_f64(&self, x: f64) -> usize {
        let n = self.segments.len();
        let a = self.breakpoints[0].to_f64();
        let b = self.breakpoints.last().unwrap().to_f64();

        if x <= a {
            return 0;
        }
        if x >= b {
            return n - 1;
        }

        for i in (0..n).rev() {
            if x >= self.breakpoints[i].to_f64() {
                return i;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{FitOptions, fit_dense, uniform_breakpoints};

    #[test]
    fn segment_index_single_segment() {
        let seg = ChebyshevSegment {
            coeffs: vec![2.0, 1.0],
            a: 0.0,
            b: 1.0,
        };
        let pw = PiecewiseFit::new(vec![seg], vec![0.0, 1.0]);
        assert_eq!(pw.segment_index(0.0).unwrap(), 0);
        assert_eq!(pw.segment_index(0.5).unwrap(), 0);
        assert_eq!(pw.segment_index(1.0).unwrap(), 0);
    }

    #[test]
    fn segment_index_two_segments() {
        let seg0 = ChebyshevSegment {
            coeffs: vec![2.0],
            a: 0.0,
            b: 1.0,
        };
        let seg1 = ChebyshevSegment {
            coeffs: vec![4.0],
            a: 1.0,
            b: 2.0,
        };
        let pw = PiecewiseFit::new(vec![seg0, seg1], vec![0.0, 1.0, 2.0]);
        assert_eq!(pw.segment_index(0.0).unwrap(), 0);
        assert_eq!(pw.segment_index(0.5).unwrap(), 0);
        assert_eq!(pw.segment_index(1.0).unwrap(), 1);
        assert_eq!(pw.segment_index(1.5).unwrap(), 1);
        assert_eq!(pw.segment_index(2.0).unwrap(), 1);
    }

    #[test]
    fn segment_index_out_of_domain() {
        let seg = ChebyshevSegment {
            coeffs: vec![2.0],
            a: 0.0,
            b: 1.0,
        };
        let pw = PiecewiseFit::new(vec![seg], vec![0.0, 1.0]);
        assert!(pw.segment_index(-0.1).is_err());
        assert!(pw.segment_index(1.1).is_err());
    }

    #[test]
    fn eval_derivative_of_polynomial() {
        // Fit f(x) = x² on [0, 2], check f'(x) = 2x
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();
        let result = fit_dense(&x_data, &y_data, &[0.0, 2.0], &FitOptions { degree: 10 }).unwrap();

        for &x in &[0.5, 1.0, 1.5] {
            let deriv = result.fit.eval_derivative(x, 1);
            let exact = 2.0 * x;
            assert!(
                (deriv - exact).abs() < 1e-3,
                "f'({x}): got {deriv}, expected {exact}"
            );
        }
    }

    #[test]
    fn eval_derivative_second_order() {
        // f(x) = x³ on [0, 1], f''(x) = 6x
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x * x).collect();
        let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 10 }).unwrap();

        // Dense fit of x³ degree 10 on [0,1]: h=1/50=0.02
        // f'' err dominated by twice-amplified resampling noise
        // Observed: ~1e-2 at interior points
        for &x in &[0.25, 0.5, 0.75] {
            let d2 = result.fit.eval_derivative(x, 2);
            let exact = 6.0 * x;
            assert!(
                (d2 - exact).abs() < 5e-2,
                "f''({x}): got {d2}, expected {exact}, err = {:.2e}",
                (d2 - exact).abs()
            );
        }
    }

    #[test]
    fn try_eval_returns_error_out_of_domain() {
        let seg = ChebyshevSegment {
            coeffs: vec![2.0, 1.0],
            a: 0.0,
            b: 1.0,
        };
        let pw = PiecewiseFit::new(vec![seg], vec![0.0, 1.0]);
        // try_eval returns error for out-of-domain
        assert!(pw.try_eval(-0.1).is_err());
        assert!(pw.try_eval(1.1).is_err());
        // try_eval returns Ok for in-domain
        assert!(pw.try_eval(0.5).is_ok());
    }

    #[test]
    fn eval_clamps_out_of_domain() {
        let seg = ChebyshevSegment {
            coeffs: vec![2.0, 1.0],
            a: 0.0,
            b: 1.0,
        };
        let pw = PiecewiseFit::new(vec![seg], vec![0.0, 1.0]);
        // eval does not panic for out-of-domain, it clamps
        let _ = pw.eval(-0.1);
        let _ = pw.eval(1.1);
    }

    #[test]
    fn piecewise_continuity_at_boundary() {
        // Fit sin(x) on [0, π] with 2 segments — should be continuous at π/2
        let n = 100;
        let x_data: Vec<f64> = (0..=n)
            .map(|i| std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
        let bp = uniform_breakpoints(0.0, std::f64::consts::PI, 2).unwrap();
        let result = fit_dense(&x_data, &y_data, &bp, &FitOptions { degree: 16 }).unwrap();

        // Evaluate just left and just right of the boundary
        let boundary = std::f64::consts::PI / 2.0;
        let left = result.fit.eval(boundary - 1e-10);
        let right = result.fit.eval(boundary + 1e-10);
        assert!(
            (left - right).abs() < 1e-3,
            "discontinuity at boundary: left={left}, right={right}"
        );
    }
}
