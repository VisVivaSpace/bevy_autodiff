//! Graph optimization utilities.
//!
//! This module provides optimization passes for the computation graph:
//! - Constant folding: evaluate operations on constants at build time
//! - Algebraic simplification: x+0=x, x*1=x, x*0=0, etc.
//! - Common subexpression elimination (CSE)

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use std::collections::HashMap;

use crate::components::{BinaryInputs, IsConstant, IsInput, Value};
use crate::context::{BinaryOpMarker, UnaryOpMarker};
use crate::graph::topological_order;
use crate::var::Var;

/// Result of attempting to simplify a binary operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimplifyResult {
    /// Keep the operation as-is
    Keep,
    /// Replace with the left operand
    UseLeft,
    /// Replace with the right operand
    UseRight,
    /// Replace with a constant value
    Constant(f64),
}

/// Attempts to simplify a binary operation based on algebraic identities.
///
/// # Simplification rules
///
/// - Addition: x + 0 = x, 0 + x = x
/// - Subtraction: x - 0 = x, x - x = 0
/// - Multiplication: x * 1 = x, 1 * x = x, x * 0 = 0, 0 * x = 0
/// - Division: x / 1 = x, 0 / x = 0 (if x != 0)
pub fn simplify_binary(
    world: &World,
    op_name: &str,
    left: Entity,
    right: Entity,
) -> SimplifyResult {
    let left_const = get_constant_value(world, left);
    let right_const = get_constant_value(world, right);

    match op_name {
        "add" => {
            // x + 0 = x
            if right_const == Some(0.0) {
                return SimplifyResult::UseLeft;
            }
            // 0 + x = x
            if left_const == Some(0.0) {
                return SimplifyResult::UseRight;
            }
            // Fold constants
            if let (Some(l), Some(r)) = (left_const, right_const) {
                return SimplifyResult::Constant(l + r);
            }
        }
        "sub" => {
            // x - 0 = x
            if right_const == Some(0.0) {
                return SimplifyResult::UseLeft;
            }
            // x - x = 0
            if left == right {
                return SimplifyResult::Constant(0.0);
            }
            // Fold constants
            if let (Some(l), Some(r)) = (left_const, right_const) {
                return SimplifyResult::Constant(l - r);
            }
        }
        "mul" => {
            // x * 0 = 0
            if right_const == Some(0.0) {
                return SimplifyResult::Constant(0.0);
            }
            // 0 * x = 0
            if left_const == Some(0.0) {
                return SimplifyResult::Constant(0.0);
            }
            // x * 1 = x
            if right_const == Some(1.0) {
                return SimplifyResult::UseLeft;
            }
            // 1 * x = x
            if left_const == Some(1.0) {
                return SimplifyResult::UseRight;
            }
            // Fold constants
            if let (Some(l), Some(r)) = (left_const, right_const) {
                return SimplifyResult::Constant(l * r);
            }
        }
        "div" => {
            // x / 1 = x
            if right_const == Some(1.0) {
                return SimplifyResult::UseLeft;
            }
            // 0 / x = 0 (when x is constant and non-zero)
            if left_const == Some(0.0) {
                if let Some(r) = right_const {
                    if r != 0.0 {
                        return SimplifyResult::Constant(0.0);
                    }
                }
            }
            // Fold constants (avoid division by zero)
            if let (Some(l), Some(r)) = (left_const, right_const) {
                if r != 0.0 {
                    return SimplifyResult::Constant(l / r);
                }
            }
        }
        _ => {}
    }

    SimplifyResult::Keep
}

/// Gets the constant value of an entity if it's a constant.
fn get_constant_value(world: &World, entity: Entity) -> Option<f64> {
    let entity_ref = world.entity(entity);
    if entity_ref.contains::<IsConstant>() {
        entity_ref.get::<Value>().map(|v| v.get())
    } else {
        None
    }
}

/// Attempts to simplify a unary operation based on algebraic identities.
///
/// # Simplification rules
///
/// - neg(neg(x)) = x (double negation)
/// - exp(0) = 1, ln(1) = 0, sqrt(1) = 1
/// - sin(0) = 0, cos(0) = 1
pub fn simplify_unary(world: &World, op_name: &str, input: Entity) -> SimplifyResult {
    let input_const = get_constant_value(world, input);

    match op_name {
        "neg" => {
            // neg(0) = 0
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(0.0);
            }
            // Fold constant
            if let Some(v) = input_const {
                return SimplifyResult::Constant(-v);
            }
        }
        "exp" => {
            // exp(0) = 1
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(1.0);
            }
            // Fold constant
            if let Some(v) = input_const {
                return SimplifyResult::Constant(v.exp());
            }
        }
        "ln" => {
            // ln(1) = 0
            if input_const == Some(1.0) {
                return SimplifyResult::Constant(0.0);
            }
            // Fold constant (only for positive values)
            if let Some(v) = input_const {
                if v > 0.0 {
                    return SimplifyResult::Constant(v.ln());
                }
            }
        }
        "sqrt" => {
            // sqrt(0) = 0
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(0.0);
            }
            // sqrt(1) = 1
            if input_const == Some(1.0) {
                return SimplifyResult::Constant(1.0);
            }
            // Fold constant (only for non-negative values)
            if let Some(v) = input_const {
                if v >= 0.0 {
                    return SimplifyResult::Constant(v.sqrt());
                }
            }
        }
        "sin" => {
            // sin(0) = 0
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(0.0);
            }
            // Fold constant
            if let Some(v) = input_const {
                return SimplifyResult::Constant(v.sin());
            }
        }
        "cos" => {
            // cos(0) = 1
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(1.0);
            }
            // Fold constant
            if let Some(v) = input_const {
                return SimplifyResult::Constant(v.cos());
            }
        }
        "sinh" => {
            // sinh(0) = 0
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(0.0);
            }
            // Fold constant
            if let Some(v) = input_const {
                return SimplifyResult::Constant(v.sinh());
            }
        }
        "cosh" => {
            // cosh(0) = 1
            if input_const == Some(0.0) {
                return SimplifyResult::Constant(1.0);
            }
            // Fold constant
            if let Some(v) = input_const {
                return SimplifyResult::Constant(v.cosh());
            }
        }
        _ => {}
    }

    SimplifyResult::Keep
}

/// Signature for common subexpression elimination.
///
/// Two operations with the same signature can be merged.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpSignature {
    /// Unary operation with (op_name, input_entity)
    Unary(&'static str, Entity),
    /// Binary operation with (op_name, left_entity, right_entity)
    Binary(&'static str, Entity, Entity),
    /// Binary operation that is commutative (uses sorted entity order)
    CommutativeBinary(&'static str, Entity, Entity),
}

impl OpSignature {
    /// Creates a signature for a unary operation.
    pub fn unary(op_name: &'static str, input: Entity) -> Self {
        Self::Unary(op_name, input)
    }

    /// Creates a signature for a binary operation.
    ///
    /// For commutative operations (add, mul), normalizes the order.
    pub fn binary(op_name: &'static str, left: Entity, right: Entity) -> Self {
        match op_name {
            "add" | "mul" => {
                // Normalize order for commutative operations
                let (a, b) = if left.index() <= right.index() {
                    (left, right)
                } else {
                    (right, left)
                };
                Self::CommutativeBinary(op_name, a, b)
            }
            _ => Self::Binary(op_name, left, right),
        }
    }
}

/// A table for finding existing operations (common subexpression elimination).
#[derive(Debug, Default)]
pub struct CseTable {
    signatures: HashMap<OpSignature, Entity>,
}

impl CseTable {
    /// Creates a new CSE table.
    pub fn new() -> Self {
        Self {
            signatures: HashMap::new(),
        }
    }

    /// Looks up an existing operation by signature.
    pub fn find(&self, signature: &OpSignature) -> Option<Entity> {
        self.signatures.get(signature).copied()
    }

    /// Registers an operation in the CSE table.
    pub fn register(&mut self, signature: OpSignature, entity: Entity) {
        self.signatures.insert(signature, entity);
    }

    /// Finds or registers an operation, returning the existing entity if found.
    pub fn find_or_register(&mut self, signature: OpSignature, entity: Entity) -> Entity {
        *self.signatures.entry(signature).or_insert(entity)
    }
}

/// Builds a CSE table from an existing computation graph.
pub fn build_cse_table(world: &World, output: Var) -> CseTable {
    let mut table = CseTable::new();
    let entities = topological_order(world, output.entity());

    for &entity in &entities {
        let entity_ref = world.entity(entity);

        // Skip inputs and constants
        if entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>() {
            continue;
        }

        // Register unary operations
        if let Some(op) = entity_ref.get::<UnaryOpMarker>() {
            if let Some(input) = entity_ref.get::<crate::components::UnaryInput>() {
                let sig = OpSignature::unary(op.0.name(), input.get().entity());
                table.register(sig, entity);
            }
        }

        // Register binary operations
        if let Some(op) = entity_ref.get::<BinaryOpMarker>() {
            if let Some(inputs) = entity_ref.get::<BinaryInputs>() {
                let sig = OpSignature::binary(op.0.name(), inputs.left.entity(), inputs.right.entity());
                table.register(sig, entity);
            }
        }
    }

    table
}

/// Counts potential CSE opportunities in a computation graph.
///
/// Returns the number of duplicate operations that could be eliminated.
pub fn count_cse_opportunities(world: &World, output: Var) -> usize {
    let mut signatures: HashMap<OpSignature, usize> = HashMap::new();
    let entities = topological_order(world, output.entity());

    for &entity in &entities {
        let entity_ref = world.entity(entity);

        if entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>() {
            continue;
        }

        if let Some(op) = entity_ref.get::<UnaryOpMarker>() {
            if let Some(input) = entity_ref.get::<crate::components::UnaryInput>() {
                let sig = OpSignature::unary(op.0.name(), input.get().entity());
                *signatures.entry(sig).or_insert(0) += 1;
            }
        }

        if let Some(op) = entity_ref.get::<BinaryOpMarker>() {
            if let Some(inputs) = entity_ref.get::<BinaryInputs>() {
                let sig = OpSignature::binary(op.0.name(), inputs.left.entity(), inputs.right.entity());
                *signatures.entry(sig).or_insert(0) += 1;
            }
        }
    }

    // Count duplicates (entries with count > 1)
    signatures.values().filter(|&&count| count > 1).map(|&c| c - 1).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;

    #[test]
    fn test_simplify_add_zero() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let zero = ad.constant(0.0);

        // x + 0 = x
        let result = simplify_binary(ad.world(), "add", x.entity(), zero.entity());
        assert_eq!(result, SimplifyResult::UseLeft);

        // 0 + x = x
        let result = simplify_binary(ad.world(), "add", zero.entity(), x.entity());
        assert_eq!(result, SimplifyResult::UseRight);
    }

    #[test]
    fn test_simplify_mul_zero_one() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);
        let zero = ad.constant(0.0);
        let one = ad.constant(1.0);

        // x * 0 = 0
        let result = simplify_binary(ad.world(), "mul", x.entity(), zero.entity());
        assert_eq!(result, SimplifyResult::Constant(0.0));

        // x * 1 = x
        let result = simplify_binary(ad.world(), "mul", x.entity(), one.entity());
        assert_eq!(result, SimplifyResult::UseLeft);

        // 1 * x = x
        let result = simplify_binary(ad.world(), "mul", one.entity(), x.entity());
        assert_eq!(result, SimplifyResult::UseRight);
    }

    #[test]
    fn test_simplify_sub_self() {
        let mut ad = AutoDiff::new();
        let x = ad.var(5.0);

        // x - x = 0
        let result = simplify_binary(ad.world(), "sub", x.entity(), x.entity());
        assert_eq!(result, SimplifyResult::Constant(0.0));
    }

    #[test]
    fn test_constant_folding() {
        let mut ad = AutoDiff::new();
        let a = ad.constant(3.0);
        let b = ad.constant(4.0);

        // 3 + 4 = 7
        let result = simplify_binary(ad.world(), "add", a.entity(), b.entity());
        assert_eq!(result, SimplifyResult::Constant(7.0));

        // 3 * 4 = 12
        let result = simplify_binary(ad.world(), "mul", a.entity(), b.entity());
        assert_eq!(result, SimplifyResult::Constant(12.0));
    }

    #[test]
    fn test_simplify_unary_special_values() {
        let mut ad = AutoDiff::new();
        let zero = ad.constant(0.0);
        let one = ad.constant(1.0);

        // exp(0) = 1
        assert_eq!(simplify_unary(ad.world(), "exp", zero.entity()), SimplifyResult::Constant(1.0));

        // ln(1) = 0
        assert_eq!(simplify_unary(ad.world(), "ln", one.entity()), SimplifyResult::Constant(0.0));

        // sin(0) = 0
        assert_eq!(simplify_unary(ad.world(), "sin", zero.entity()), SimplifyResult::Constant(0.0));

        // cos(0) = 1
        assert_eq!(simplify_unary(ad.world(), "cos", zero.entity()), SimplifyResult::Constant(1.0));
    }

    #[test]
    fn test_cse_table() {
        let mut table = CseTable::new();

        // Create dummy entity IDs for testing
        let e1 = Entity::from_raw(1);
        let e2 = Entity::from_raw(2);
        let e3 = Entity::from_raw(3);

        let sig1 = OpSignature::binary("add", e1, e2);
        table.register(sig1.clone(), e3);

        assert_eq!(table.find(&sig1), Some(e3));

        // Commutative operations should match regardless of order
        let sig2 = OpSignature::binary("add", e2, e1);
        assert_eq!(table.find(&sig2), Some(e3));
    }

    #[test]
    fn test_op_signature_commutativity() {
        let e1 = Entity::from_raw(1);
        let e2 = Entity::from_raw(2);

        // add(e1, e2) should equal add(e2, e1)
        let sig1 = OpSignature::binary("add", e1, e2);
        let sig2 = OpSignature::binary("add", e2, e1);
        assert_eq!(sig1, sig2);

        // mul(e1, e2) should equal mul(e2, e1)
        let sig3 = OpSignature::binary("mul", e1, e2);
        let sig4 = OpSignature::binary("mul", e2, e1);
        assert_eq!(sig3, sig4);

        // sub(e1, e2) should NOT equal sub(e2, e1)
        let sig5 = OpSignature::binary("sub", e1, e2);
        let sig6 = OpSignature::binary("sub", e2, e1);
        assert_ne!(sig5, sig6);
    }

    #[test]
    fn test_count_cse_opportunities() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);

        // Create duplicate expressions
        let xy1 = ad.mul(x, y);
        let xy2 = ad.mul(x, y);  // Duplicate!
        let result = ad.add(xy1, xy2);

        let count = count_cse_opportunities(ad.world(), result);
        assert_eq!(count, 1);  // One duplicate mul(x, y)
    }

    #[test]
    fn test_build_cse_table() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x);  // x * x

        let table = build_cse_table(ad.world(), y);

        // The mul operation should be registered
        let sig = OpSignature::binary("mul", x.entity(), x.entity());
        assert!(table.find(&sig).is_some());
    }
}
