//! Pre-compiled 2D piecewise evaluation.
//!
//! [`PiecewiseCompiled2D`] wraps multiple [`CompiledGraph`]s (one per segment)
//! for fast repeated evaluation of 2D tensor product fits.

use bevy_autodiff::{AutoDiff, CompiledGraph, Float};

use crate::error::FitError;
use crate::piecewise2d::PiecewiseFit2D;

/// Pre-compiled 2D piecewise fit for fast repeated evaluation.
///
/// Holds one [`CompiledGraph`] per segment (row-major). Each graph includes
/// both domain mappings (x and y), so inputs are in physical coordinates.
/// Partial derivatives use multi-indices: `&[1, 0]` = ∂f/∂x, `&[0, 1]` = ∂f/∂y.
pub struct PiecewiseCompiled2D<F: Float> {
    graphs: Vec<CompiledGraph<F>>,
    breakpoints_x: Vec<F>,
    breakpoints_y: Vec<F>,
    n_segments_x: usize,
    active_segment: usize,
}

impl<F: Float + PartialOrd> PiecewiseCompiled2D<F> {
    /// Compile a [`PiecewiseFit2D`] with derivatives up to the given order.
    ///
    /// For each segment, creates an `AutoDiff<F>`, builds the nested Clenshaw
    /// graph, and compiles with `compile_order(f, &[x, y], order)`.
    pub fn new(fit: &PiecewiseFit2D<F>, derivative_order: usize) -> Result<Self, FitError> {
        let nx = fit.num_segments_x();
        let ny = fit.num_segments_y();
        let mut graphs = Vec::with_capacity(nx * ny);

        for iy in 0..ny {
            for ix in 0..nx {
                let mut ad = AutoDiff::<F>::new();
                let x = ad.var(F::zero()).map_err(FitError::AutoDiff)?;
                let y = ad.var(F::zero()).map_err(FitError::AutoDiff)?;
                let f = fit.segment(ix, iy).build_graph(&mut ad, x, y);
                let cg = ad
                    .compile_order(f, &[x, y], derivative_order)
                    .map_err(FitError::AutoDiff)?;
                graphs.push(cg);
            }
        }

        Ok(Self {
            graphs,
            breakpoints_x: fit.breakpoints_x().to_vec(),
            breakpoints_y: fit.breakpoints_y().to_vec(),
            n_segments_x: nx,
            active_segment: 0,
        })
    }

    /// Evaluate at (x, y) in physical coordinates.
    ///
    /// Selects the correct segment, then evaluates.
    pub fn eval(&mut self, x: F, y: F) -> Result<(), FitError> {
        let (ix, iy) = self.find_segment(x, y)?;
        let idx = iy * self.n_segments_x + ix;
        self.active_segment = idx;
        self.graphs[idx]
            .eval(&[x, y])
            .map_err(FitError::AutoDiff)?;
        Ok(())
    }

    /// Get the function value after `eval()`.
    pub fn value(&self) -> F {
        self.graphs[self.active_segment].value()
    }

    /// Get a partial derivative after `eval()`.
    ///
    /// Multi-index format: `&[1, 0]` = ∂f/∂x, `&[0, 1]` = ∂f/∂y,
    /// `&[1, 1]` = ∂²f/∂x∂y, `&[2, 0]` = ∂²f/∂x².
    pub fn partial(&self, multi_index: &[usize]) -> Result<F, FitError> {
        self.graphs[self.active_segment]
            .partial(multi_index)
            .map_err(FitError::AutoDiff)
    }

    /// Number of compiled segments.
    pub fn num_segments(&self) -> usize {
        self.graphs.len()
    }

    /// Find segment (ix, iy) for point (x, y).
    fn find_segment(&self, x: F, y: F) -> Result<(usize, usize), FitError> {
        let ix = find_segment_1d(&self.breakpoints_x, x)?;
        let iy = find_segment_1d(&self.breakpoints_y, y)?;
        Ok((ix, iy))
    }
}

/// Find the segment index for a 1D coordinate.
fn find_segment_1d<F: Float + PartialOrd>(breakpoints: &[F], x: F) -> Result<usize, FitError> {
    let a = breakpoints[0];
    let b = *breakpoints.last().unwrap();
    let n = breakpoints.len() - 1;

    if x < a || x > b {
        return Err(FitError::out_of_domain(x, a, b));
    }

    if n == 1 {
        return Ok(0);
    }

    for i in (0..n).rev() {
        if x >= breakpoints[i] {
            return Ok(i);
        }
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::{FitOptions2D, fit_sparse_2d};

    #[test]
    fn compiled_2d_matches_standalone() {
        // f(x,y) = x² + 2xy + 3y² via sparse fit
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x * x + 2.0 * x * y + 3.0 * y * y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 3,
                degree_y: 3,
            },
        )
        .unwrap();

        let mut compiled = PiecewiseCompiled2D::new(&result.fit, 0).unwrap();

        for &(tx, ty) in &[(0.1, 0.2), (0.5, 0.5), (0.8, 0.3), (0.0, 1.0)] {
            compiled.eval(tx, ty).unwrap();
            let compiled_val = compiled.value();
            let standalone_val = result.fit.eval(tx, ty);
            assert!(
                (compiled_val - standalone_val).abs() < 1e-12,
                "at ({tx}, {ty}): compiled={compiled_val}, standalone={standalone_val}, err={:.2e}",
                (compiled_val - standalone_val).abs()
            );
        }
    }

    #[test]
    fn compiled_2d_derivatives() {
        // f(x,y) = x*y via sparse fit
        // ∂f/∂x = y, ∂f/∂y = x, ∂²f/∂x∂y = 1
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x * y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 2,
                degree_y: 2,
            },
        )
        .unwrap();

        let mut compiled = PiecewiseCompiled2D::new(&result.fit, 2).unwrap();

        let test_x = 0.4;
        let test_y = 0.6;
        compiled.eval(test_x, test_y).unwrap();

        let val = compiled.value();
        assert!(
            (val - test_x * test_y).abs() < 1e-12,
            "f = {val}, expected {}, err = {:.2e}",
            test_x * test_y,
            (val - test_x * test_y).abs()
        );

        let dx = compiled.partial(&[1, 0]).unwrap();
        assert!(
            (dx - test_y).abs() < 1e-10,
            "∂f/∂x = {dx}, expected {test_y}, err = {:.2e}",
            (dx - test_y).abs()
        );

        let dy = compiled.partial(&[0, 1]).unwrap();
        assert!(
            (dy - test_x).abs() < 1e-10,
            "∂f/∂y = {dy}, expected {test_x}, err = {:.2e}",
            (dy - test_x).abs()
        );

        let dxy = compiled.partial(&[1, 1]).unwrap();
        assert!(
            (dxy - 1.0).abs() < 1e-9,
            "∂²f/∂x∂y = {dxy}, expected 1.0, err = {:.2e}",
            (dxy - 1.0).abs()
        );
    }

    #[test]
    fn compiled_2d_out_of_domain() {
        let n = 10;
        let mut xd = Vec::new();
        let mut yd = Vec::new();
        let mut zd = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let x = i as f64 / (n - 1) as f64;
                let y = j as f64 / (n - 1) as f64;
                xd.push(x);
                yd.push(y);
                zd.push(x + y);
            }
        }

        let result = fit_sparse_2d(
            &xd,
            &yd,
            &zd,
            [0.0, 1.0],
            [0.0, 1.0],
            &FitOptions2D {
                degree_x: 2,
                degree_y: 2,
            },
        )
        .unwrap();

        let mut compiled = PiecewiseCompiled2D::new(&result.fit, 0).unwrap();
        assert!(compiled.eval(-0.1, 0.5).is_err());
        assert!(compiled.eval(0.5, 1.1).is_err());
    }
}
