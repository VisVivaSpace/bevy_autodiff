//! ECS components for the autodiff computation graph.
//!
//! Variables are entities with components that define their role in the graph:
//! - Marker components: `Variable`, `IsInput`, `IsConstant`
//! - Value storage: `Value`
//! - Operation definitions: `UnaryOp`, `BinaryOp`, `UnaryInput`, `BinaryInputs`
//! - Dependency tracking: `Dependencies`

mod operations;
mod variable;

pub use operations::{BinaryInputs, BinaryOp, BinaryOpMarker, UnaryInput, UnaryOp, UnaryOpMarker};
pub use variable::{Dependencies, IsConstant, IsInput, Value, Variable};
