//! Taylor coefficient propagation.
//!
//! This module implements the core Taylor-mode automatic differentiation:
//! - Polynomial arithmetic (add, sub, mul, div, neg)
//! - Elementary function rules (sin, cos, exp, ln, sqrt, sinh, cosh)
//! - Graph traversal and coefficient propagation
//! - Incremental order extension
//!
//! All coefficient operations are pure functions that take `&[f64]` and return `Vec<f64>`.
//! This ensures referential transparency and makes the code easy to test.

pub mod extend;
pub mod polynomial;
pub mod propagate;
pub mod rules;

pub use extend::{ensure_taylor_order, extend_taylor_order};
pub use polynomial::{add_taylor, div_taylor, mul_taylor, neg_taylor, sub_taylor};
pub use propagate::{compute_taylor_coeffs, propagate_taylor};

// Re-export commonly used rules
pub use rules::{
    cos_taylor, cosh_taylor, exp_taylor, ln_taylor, sin_cos_taylor, sin_taylor, sinh_cosh_taylor,
    sinh_taylor, sqrt_taylor, CoupledRecurrence, SinCos, SinhCosh,
};
