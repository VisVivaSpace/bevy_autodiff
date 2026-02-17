//! Piecewise Chebyshev polynomial fitting for [`bevy_autodiff`].
//!
//! This crate fits smooth piecewise Chebyshev polynomials to tabulated data
//! and integrates them with `bevy_autodiff`'s symbolic differentiation.
//! Derivatives are computed exactly via the chain rule — the Clenshaw evaluation
//! is built as graph nodes, so `differentiate()` just works.
//!
//! # Important
//!
//! The fit is an **approximation**, not exact. Derivatives of the fit approximate
//! derivatives of the underlying function, with accuracy limited by:
//! - Polynomial degree (higher = more accurate, but amplifies noise)
//! - Data quality (noisy data → unreliable higher derivatives)
//! - Segment width (narrower segments → less polynomial bending)
//!
//! Use [`FitResult::reliability`] to check how many derivatives are trustworthy.
//!
//! # Quick Start
//!
//! ```
//! use bevy_autodiff_fit::{fit_dense, FitOptions, uniform_breakpoints, PiecewiseCompiled};
//!
//! // Your data
//! let x: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
//! let y: Vec<f64> = x.iter().map(|&x| x.sin()).collect();
//!
//! // Fit with a single segment, degree 20
//! let result = fit_dense(&x, &y, &[0.0, 1.0], &FitOptions { degree: 20 }).unwrap();
//!
//! // Check reliability
//! println!("reliable up to order {}", result.reliability[0].max_reliable_order);
//!
//! // Compile for fast evaluation with first derivatives
//! let mut compiled = PiecewiseCompiled::new(&result.fit, 1).unwrap();
//! compiled.eval(0.5).unwrap();
//! let value = compiled.value();
//! let derivative = compiled.partial(&[1]).unwrap();
//! ```
//!
//! # Integration with bevy_autodiff
//!
//! For composing fits with other AD operations:
//!
//! ```ignore
//! use bevy_autodiff::AutoDiff;
//!
//! let mut ad = AutoDiff::new();
//! let x = ad.var(0.5).unwrap();
//!
//! // Build Clenshaw graph for segment 0
//! let f = fit_result.fit.build_segment_graph(&mut ad, x, 0).unwrap();
//!
//! // Compose with other operations
//! let g = ad.exp(f);  // g = exp(fit(x))
//!
//! // Differentiate — chain rule is automatic
//! let dg = ad.differentiate(g, x).unwrap();
//! ```

pub mod chebyshev;
mod compiled;
pub mod error;
mod fit;
mod graph;
pub mod piecewise;
pub mod reliability;

// Re-exports: primary public API
pub use compiled::PiecewiseCompiled;
pub use error::FitError;
pub use fit::{
    ChebyshevSegment, FitOptions, FitResult, fit_dense, fit_sparse, uniform_breakpoints,
};
pub use piecewise::PiecewiseFit;
pub use reliability::SegmentReliability;
