//! Pre-compiled piecewise evaluation.
//!
//! [`PiecewiseCompiled`] wraps multiple [`CompiledGraph`]s (one per segment)
//! for fast repeated evaluation with automatic segment selection.

use bevy_autodiff::{AutoDiff, CompiledGraph, Float};

use crate::error::FitError;
use crate::piecewise::PiecewiseFit;

/// Pre-compiled piecewise fit for fast repeated evaluation.
///
/// Holds one [`CompiledGraph`] per segment. Each graph includes the domain
/// mapping, so the input is x in physical coordinates — the chain rule through
/// the mapping is automatic.
///
/// # Example
///
/// ```ignore
/// let compiled = PiecewiseCompiled::new(&fit_result.fit, 2)?;
/// compiled.eval(1.5)?;
/// let value = compiled.value();
/// let first_deriv = compiled.partial(&[1])?;
/// ```
pub struct PiecewiseCompiled<F: Float> {
    graphs: Vec<CompiledGraph<F>>,
    breakpoints: Vec<F>,
    /// Index of the segment used in the most recent eval().
    active_segment: usize,
}

impl<F: Float + PartialOrd> PiecewiseCompiled<F> {
    /// Compile a [`PiecewiseFit`] with derivatives up to the given order.
    ///
    /// For each segment, creates an `AutoDiff<F>`, builds the Clenshaw graph
    /// (including domain mapping), and compiles with `compile_order`.
    pub fn new(fit: &PiecewiseFit<F>, derivative_order: usize) -> Result<Self, FitError> {
        let mut graphs = Vec::with_capacity(fit.num_segments());

        for i in 0..fit.num_segments() {
            let mut ad = AutoDiff::<F>::new();
            let x = ad.var(F::zero()).map_err(FitError::AutoDiff)?;
            let f = fit.segment(i).build_graph(&mut ad, x);
            let cg = ad
                .compile_order(f, &[x], derivative_order)
                .map_err(FitError::AutoDiff)?;
            graphs.push(cg);
        }

        Ok(Self {
            graphs,
            breakpoints: fit.breakpoints().to_vec(),
            active_segment: 0,
        })
    }

    /// Evaluate at x in physical coordinates.
    ///
    /// Selects the correct segment via binary search, then evaluates.
    /// The domain mapping is inside the graph, so x is passed directly.
    pub fn eval(&mut self, x: F) -> Result<(), FitError> {
        let idx = self.find_segment(x)?;
        self.active_segment = idx;
        self.graphs[idx].eval(&[x]).map_err(FitError::AutoDiff)?;
        Ok(())
    }

    /// Get the function value after `eval()`.
    pub fn value(&self) -> F {
        self.graphs[self.active_segment].value()
    }

    /// Get a partial derivative after `eval()`.
    ///
    /// For 1D fits, use `&[n]` to get the n-th derivative.
    pub fn partial(&self, multi_index: &[usize]) -> Result<F, FitError> {
        self.graphs[self.active_segment]
            .partial(multi_index)
            .map_err(FitError::AutoDiff)
    }

    /// Number of compiled segments.
    pub fn num_segments(&self) -> usize {
        self.graphs.len()
    }

    /// Find segment index for x (same logic as PiecewiseFit).
    fn find_segment(&self, x: F) -> Result<usize, FitError> {
        let a = self.breakpoints[0];
        let b = *self.breakpoints.last().unwrap();

        if x < a || x > b {
            return Err(FitError::out_of_domain(x, a, b));
        }

        let n = self.graphs.len();
        if n == 1 {
            return Ok(0);
        }

        for i in (0..n).rev() {
            if x >= self.breakpoints[i] {
                return Ok(i);
            }
        }
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::{FitOptions, fit_dense, uniform_breakpoints};

    #[test]
    fn compiled_matches_standalone() {
        // Fit sin(x) on [0, π] with 2 segments, compare compiled vs standalone
        let n = 100;
        let x_data: Vec<f64> = (0..=n)
            .map(|i| std::f64::consts::PI * i as f64 / n as f64)
            .collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x.sin()).collect();
        let bp = uniform_breakpoints(0.0, std::f64::consts::PI, 2);
        let result = fit_dense(&x_data, &y_data, &bp, &FitOptions { degree: 16 }).unwrap();

        let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();

        for &x in &[0.1, 0.5, 1.0, 1.5, 2.0, 3.0] {
            compiled.eval(x).unwrap();
            let compiled_val = compiled.value();
            let standalone_val = result.fit.eval(x);
            assert!(
                (compiled_val - standalone_val).abs() < 1e-12,
                "at {x}: compiled={compiled_val}, standalone={standalone_val}"
            );
        }
    }

    #[test]
    fn compiled_derivatives() {
        // Fit x² on [0, 2], check first derivative
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| 2.0 * i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();
        let result = fit_dense(&x_data, &y_data, &[0.0, 2.0], &FitOptions { degree: 10 }).unwrap();

        let mut compiled = PiecewiseCompiled::new(&result.fit, 2).unwrap();

        for &x in &[0.5, 1.0, 1.5] {
            compiled.eval(x).unwrap();
            let val = compiled.value();
            let d1 = compiled.partial(&[1]).unwrap();
            let d2 = compiled.partial(&[2]).unwrap();

            assert!(
                (val - x * x).abs() < 1e-3,
                "f({x}): got {val}, expected {}",
                x * x
            );
            assert!(
                (d1 - 2.0 * x).abs() < 1e-2,
                "f'({x}): got {d1}, expected {}",
                2.0 * x
            );
            assert!((d2 - 2.0).abs() < 0.5, "f''({x}): got {d2}, expected 2.0");
        }
    }

    #[test]
    fn compiled_out_of_domain() {
        let n = 50;
        let x_data: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&x| x * x).collect();
        let result = fit_dense(&x_data, &y_data, &[0.0, 1.0], &FitOptions { degree: 5 }).unwrap();
        let mut compiled = PiecewiseCompiled::new(&result.fit, 0).unwrap();

        assert!(compiled.eval(-0.1).is_err());
        assert!(compiled.eval(1.1).is_err());
    }
}
