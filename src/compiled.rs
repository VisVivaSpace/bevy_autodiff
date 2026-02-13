//! Compiled computation graph for fast repeated evaluation.
//!
//! The ECS graph handles expression building; `NodeOp` flattens it
//! into a representation that can be re-evaluated without touching the ECS.

use crate::components::{BinaryOp, UnaryOp};

/// A node in the flattened computation graph.
#[derive(Clone, Copy, Debug)]
pub enum NodeOp {
    /// Read from inputs[index].
    Input(usize),
    /// Fixed constant value.
    Constant(f64),
    /// Unary operation on node at index `src`.
    Unary { op: UnaryOp, src: usize },
    /// Binary operation on nodes at indices `lhs`, `rhs`.
    Binary { op: BinaryOp, lhs: usize, rhs: usize },
}

// =============================================================================
// Value-level helpers (compute f(x) for each operation, no derivatives)
// =============================================================================

/// Apply a unary operation to a value.
pub fn apply_unary_value(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Asin => x.asin(),
        UnaryOp::Acos => x.acos(),
        UnaryOp::Atan => x.atan(),
        UnaryOp::Asinh => x.asinh(),
        UnaryOp::Acosh => x.acosh(),
        UnaryOp::Atanh => x.atanh(),
    }
}

/// Apply a binary operation to two values.
pub fn apply_binary_value(op: BinaryOp, x: f64, y: f64) -> f64 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        BinaryOp::Pow => x.powf(y),
    }
}
