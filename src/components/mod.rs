//! ECS components for the autodiff computation graph.
//!
//! Variables are entities with components that define their role in the graph:
//! - Marker components: `Variable`, `IsInput`, `IsConstant`
//! - Value storage: `Value`
//! - Operation definitions: `UnaryOp`, `BinaryOp`, `UnaryInput`, `BinaryInputs`
//! - Taylor data: `TaylorData`, `Direction`, `MultiIndex`
//! - Adjoint data: `AdjointTaylor` for reverse mode
//! - Dependency tracking: `Dependencies`

mod adjoint;
mod operations;
mod taylor;
mod variable;

pub use adjoint::AdjointTaylor;
pub use operations::{BinaryInputs, BinaryOp, UnaryInput, UnaryOp};
pub use taylor::{Direction, MultiIndex, TaylorData};
pub use variable::{Dependencies, IsConstant, IsInput, Value, Variable};
