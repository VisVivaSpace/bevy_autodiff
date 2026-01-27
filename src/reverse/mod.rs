//! Reverse mode automatic differentiation.
//!
//! Reverse mode AD (also called backpropagation or adjoint mode) computes
//! gradients efficiently when there are many inputs but few outputs.
//!
//! ## How It Works
//!
//! 1. **Forward pass**: Compute function value and store intermediate Taylor coefficients
//! 2. **Backward pass**: Propagate adjoint polynomials from output to inputs
//!
//! ## Adjoint Rules
//!
//! For each operation, we have adjoint rules that describe how to backpropagate
//! sensitivities through the operation.
//!
//! For y = f(u, v):
//! - ū += ∂f/∂u · ȳ  (adjoint of u accumulates contribution from y)
//! - v̄ += ∂f/∂v · ȳ  (adjoint of v accumulates contribution from y)
//!
//! ## Polynomial Adjoints
//!
//! In Taylor mode, adjoints are polynomials (not just scalars). This allows
//! computing higher-order derivatives via reverse mode.

mod adjoint_rules;
mod gradient;

pub use adjoint_rules::{
    adjoint_add, adjoint_div, adjoint_exp, adjoint_ln, adjoint_mul, adjoint_neg, adjoint_sin_cos,
    adjoint_sqrt, adjoint_sub,
};
pub use gradient::{compute_gradient_reverse, reverse_accumulate};
