//! Taylor coefficient computation rules.
//!
//! This module provides pure functions for computing Taylor coefficients
//! of elementary functions. All rules are O(n²) for order n.
//!
//! ## Arithmetic Rules
//! Re-exported from `taylor::polynomial`: add, sub, mul, div, neg
//!
//! ## Elementary Function Rules
//! - `exp_taylor`: Exponential e^u
//! - `ln_taylor`: Natural logarithm ln(u)
//! - `sqrt_taylor`: Square root √u
//!
//! ## Power Function Rules
//! - `pow_taylor`: General power u^v
//! - `pow_int_taylor`: Integer power u^n
//! - `pow_const_taylor`: Constant power u^p
//!
//! ## Coupled Recurrence Rules
//! For functions like sin/cos that satisfy coupled differential equations:
//! - `CoupledRecurrence` trait for extensibility
//! - `SinCos` implementation for sin/cos pair
//! - `SinhCosh` implementation for sinh/cosh pair
//!
//! ## Inverse Trigonometric and Hyperbolic Rules
//! - `tan_taylor`, `tanh_taylor`: Tangent functions
//! - `asin_taylor`, `acos_taylor`, `atan_taylor`: Inverse trig
//! - `asinh_taylor`, `acosh_taylor`, `atanh_taylor`: Inverse hyperbolic

pub mod coupled;
pub mod elementary;
pub mod inverse;
pub mod power;

pub use coupled::{
    cos_taylor, cosh_taylor, coupled_taylor, sin_cos_taylor, sin_taylor, sinh_cosh_taylor,
    sinh_taylor, CoupledRecurrence, SinCos, SinhCosh,
};
pub use elementary::{exp_taylor, ln_taylor, pow_const_taylor, sqrt_taylor};
pub use inverse::{
    acos_taylor, acosh_taylor, asin_taylor, asinh_taylor, atan_taylor, atanh_taylor, tan_taylor,
    tanh_taylor,
};
pub use power::{pow_int_taylor, pow_taylor};

// Re-export arithmetic rules from polynomial module
pub use super::polynomial::{
    add_taylor, constant_taylor, div_taylor, identity_taylor, mul_taylor, neg_taylor, scale_taylor,
    sub_taylor,
};
