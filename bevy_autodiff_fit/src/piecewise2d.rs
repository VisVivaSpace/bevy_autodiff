//! Piecewise 2D Chebyshev fit management.
//!
//! A [`PiecewiseFit2D`] holds a rectangular grid of [`ChebyshevSegment2D`]s,
//! with automatic segment lookup and coordinate mapping.

use bevy_autodiff::Float;

use crate::error::FitError;
use crate::fit2d::ChebyshevSegment2D;

/// A piecewise 2D Chebyshev fit: rectangular grid of segments.
///
/// Segments are stored row-major: `segments[iy * n_segments_x + ix]`.
/// Each segment is an independent tensor product Chebyshev polynomial on its sub-rectangle.
#[derive(Clone, Debug)]
pub struct PiecewiseFit2D<F: Float> {
    segments: Vec<ChebyshevSegment2D<F>>,
    breakpoints_x: Vec<F>,
    breakpoints_y: Vec<F>,
    n_segments_x: usize,
    n_segments_y: usize,
}

impl<F: Float> PiecewiseFit2D<F> {
    /// Create a new piecewise 2D fit.
    pub(crate) fn new(
        segments: Vec<ChebyshevSegment2D<F>>,
        breakpoints_x: Vec<F>,
        breakpoints_y: Vec<F>,
        n_segments_x: usize,
        n_segments_y: usize,
    ) -> Self {
        assert_eq!(
            segments.len(),
            n_segments_x * n_segments_y,
            "segments.len() must equal n_segments_x * n_segments_y"
        );
        assert_eq!(breakpoints_x.len(), n_segments_x + 1);
        assert_eq!(breakpoints_y.len(), n_segments_y + 1);
        Self {
            segments,
            breakpoints_x,
            breakpoints_y,
            n_segments_x,
            n_segments_y,
        }
    }

    /// Number of segments in the x direction.
    pub fn num_segments_x(&self) -> usize {
        self.n_segments_x
    }

    /// Number of segments in the y direction.
    pub fn num_segments_y(&self) -> usize {
        self.n_segments_y
    }

    /// Access a segment by (ix, iy) indices.
    pub fn segment(&self, ix: usize, iy: usize) -> &ChebyshevSegment2D<F> {
        &self.segments[iy * self.n_segments_x + ix]
    }

    /// The x breakpoints.
    pub fn breakpoints_x(&self) -> &[F] {
        &self.breakpoints_x
    }

    /// The y breakpoints.
    pub fn breakpoints_y(&self) -> &[F] {
        &self.breakpoints_y
    }

    /// The left endpoint of the x domain.
    pub fn domain_min_x(&self) -> F {
        self.breakpoints_x[0]
    }

    /// The right endpoint of the x domain.
    pub fn domain_max_x(&self) -> F {
        *self.breakpoints_x.last().unwrap()
    }

    /// The left endpoint of the y domain.
    pub fn domain_min_y(&self) -> F {
        self.breakpoints_y[0]
    }

    /// The right endpoint of the y domain.
    pub fn domain_max_y(&self) -> F {
        *self.breakpoints_y.last().unwrap()
    }
}

impl<F: Float + PartialOrd> PiecewiseFit2D<F> {
    /// Find which segment contains (x, y).
    ///
    /// Returns (ix, iy) segment indices.
    ///
    /// # Errors
    ///
    /// Returns `FitError::OutOfDomain` if (x, y) is outside the domain.
    pub fn segment_index(&self, x: F, y: F) -> Result<(usize, usize), FitError> {
        let ix = find_segment_1d(&self.breakpoints_x, x, self.n_segments_x)?;
        let iy = find_segment_1d(&self.breakpoints_y, y, self.n_segments_y)?;
        Ok((ix, iy))
    }

    /// Evaluate the piecewise 2D fit at (x, y).
    ///
    /// Points outside the domain are clamped to the nearest segment —
    /// use [`try_eval`](Self::try_eval) to get an error instead.
    pub fn eval(&self, x: F, y: F) -> F {
        let (ix, iy) = self.segment_index(x, y).unwrap_or_else(|_| {
            let ix = clamp_segment(&self.breakpoints_x, x, self.n_segments_x);
            let iy = clamp_segment(&self.breakpoints_y, y, self.n_segments_y);
            (ix, iy)
        });
        self.segments[iy * self.n_segments_x + ix].eval(x, y)
    }

    /// Evaluate the piecewise 2D fit at (x, y), returning an error if out of domain.
    ///
    /// Unlike [`eval`](Self::eval), this does not clamp out-of-domain points.
    ///
    /// # Errors
    ///
    /// Returns `FitError::OutOfDomain` if (x, y) is outside the fitted domain.
    pub fn try_eval(&self, x: F, y: F) -> Result<F, FitError> {
        let (ix, iy) = self.segment_index(x, y)?;
        Ok(self.segments[iy * self.n_segments_x + ix].eval(x, y))
    }

    /// Build a graph for a specific segment.
    ///
    /// # Errors
    ///
    /// Returns `FitError::SegmentOutOfRange` if the segment index is invalid.
    pub fn build_segment_graph(
        &self,
        ad: &mut bevy_autodiff::AutoDiff<F>,
        x: bevy_autodiff::Var,
        y: bevy_autodiff::Var,
        ix: usize,
        iy: usize,
    ) -> Result<bevy_autodiff::Var, FitError> {
        if ix >= self.n_segments_x || iy >= self.n_segments_y {
            return Err(FitError::SegmentOutOfRange {
                index: iy * self.n_segments_x + ix,
                count: self.n_segments_x * self.n_segments_y,
            });
        }
        // build_graph is provided by the graph2d module
        Ok(self.segments[iy * self.n_segments_x + ix].build_graph(ad, x, y))
    }

}

/// Find the segment index for a 1D coordinate within breakpoints.
fn find_segment_1d<F: Float + PartialOrd>(
    breakpoints: &[F],
    x: F,
    n_segments: usize,
) -> Result<usize, FitError> {
    let a = breakpoints[0];
    let b = *breakpoints.last().unwrap();

    if x < a || x > b {
        return Err(FitError::out_of_domain(x, a, b));
    }

    if n_segments == 1 {
        return Ok(0);
    }

    for i in (0..n_segments).rev() {
        if x >= breakpoints[i] {
            return Ok(i);
        }
    }

    Ok(0)
}

/// Clamp to valid segment index.
fn clamp_segment<F: Float + PartialOrd>(breakpoints: &[F], x: F, n_segments: usize) -> usize {
    if x <= breakpoints[0] {
        return 0;
    }
    if x >= *breakpoints.last().unwrap() {
        return n_segments - 1;
    }
    for i in (0..n_segments).rev() {
        if x >= breakpoints[i] {
            return i;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use crate::fit2d::{FitOptions2D, fit_sparse_2d};

    #[test]
    fn segment_index_single_segment() {
        // f(x,y) = x*y via sparse fit on [0,1]×[0,1]
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

        assert_eq!(result.fit.segment_index(0.5, 0.5).unwrap(), (0, 0));
        assert_eq!(result.fit.segment_index(0.0, 0.0).unwrap(), (0, 0));
        assert_eq!(result.fit.segment_index(1.0, 1.0).unwrap(), (0, 0));
    }

    #[test]
    fn segment_index_out_of_domain() {
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

        assert!(result.fit.segment_index(-0.1, 0.5).is_err());
        assert!(result.fit.segment_index(0.5, 1.1).is_err());
    }
}
