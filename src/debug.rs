//! Debugging utilities for computation graphs.
//!
//! This module provides tools for inspecting and visualizing computation graphs:
//! - DOT graph output for visualization
//! - Graph validation

use crate::diff_num::Float;
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use std::fmt::{Display, Write};

use crate::components::{
    BinaryInputs, BinaryOpMarker, IsConstant, IsInput, UnaryInput, UnaryOpMarker, Value,
};
use crate::graph::topological_order;
use crate::var::Var;

/// Generates a DOT graph representation of the computation graph.
///
/// The output can be visualized using Graphviz or similar tools.
pub fn to_dot<F: Float + Display>(world: &World, output: Var) -> String {
    let mut dot = String::from("digraph {\n    rankdir=BT;\n");

    // Get all entities in topological order
    let entities = topological_order(world, output.entity())
        .expect("debug: cycle detected in computation graph");
    let entity_to_id: std::collections::HashMap<Entity, usize> =
        entities.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    // Generate node definitions
    for (i, &entity) in entities.iter().enumerate() {
        let entity_ref = world.entity(entity);
        let label = get_node_label::<F>(world, entity);
        let shape = if entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>() {
            "ellipse"
        } else {
            "box"
        };

        let _ = writeln!(dot, "    node_{} [label=\"{}\" shape={}];", i, label, shape);
    }

    // Generate edges
    for &entity in &entities {
        let entity_ref = world.entity(entity);
        let node_id = entity_to_id[&entity];

        if let Some(unary_input) = entity_ref.get::<UnaryInput>() {
            let input_entity = unary_input.get().entity();
            if let Some(&input_id) = entity_to_id.get(&input_entity) {
                let _ = writeln!(dot, "    node_{} -> node_{};", input_id, node_id);
            }
        }

        if let Some(binary_inputs) = entity_ref.get::<BinaryInputs>() {
            let left_entity = binary_inputs.left.entity();
            let right_entity = binary_inputs.right.entity();

            if let Some(&left_id) = entity_to_id.get(&left_entity) {
                let _ = writeln!(
                    dot,
                    "    node_{} -> node_{} [label=\"L\"];",
                    left_id, node_id
                );
            }
            if let Some(&right_id) = entity_to_id.get(&right_entity) {
                let _ = writeln!(
                    dot,
                    "    node_{} -> node_{} [label=\"R\"];",
                    right_id, node_id
                );
            }
        }
    }

    dot.push_str("}\n");
    dot
}

/// Gets a human-readable label for a node.
fn get_node_label<F: Float + Display>(world: &World, entity: Entity) -> String {
    let entity_ref = world.entity(entity);

    // Input or constant
    if entity_ref.contains::<IsInput>() {
        let value = entity_ref
            .get::<Value<F>>()
            .map(|v| v.get())
            .unwrap_or(F::zero());
        return format!("input={:.4}", value);
    }

    if entity_ref.contains::<IsConstant>() {
        let value = entity_ref
            .get::<Value<F>>()
            .map(|v| v.get())
            .unwrap_or(F::zero());
        return format!("const={:.4}", value);
    }

    // Unary operation
    if let Some(op) = entity_ref.get::<UnaryOpMarker>() {
        return op.op().name().to_string();
    }

    // Binary operation
    if let Some(op) = entity_ref.get::<BinaryOpMarker>() {
        return op.op().name().to_string();
    }

    "???".to_string()
}

/// Validates the computation graph for common issues.
///
/// Checks:
/// - All operation nodes have required inputs
/// - No cycles in the graph
/// - All referenced entities exist
///
/// Returns Ok(()) if valid, or Err with a description of the issue.
pub fn validate_graph<F: Float>(world: &World, output: Var) -> Result<(), String> {
    let entities = topological_order(world, output.entity())
        .map_err(|_| "cycle detected in computation graph".to_string())?;

    for &entity in &entities {
        let entity_ref = world.entity(entity);

        // Skip inputs and constants
        if entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>() {
            continue;
        }

        // Check unary operations have input
        if entity_ref.contains::<UnaryOpMarker>() {
            if let Some(input) = entity_ref.get::<UnaryInput>() {
                if world.get_entity(input.get().entity()).is_err() {
                    return Err(format!(
                        "Unary operation {:?} references non-existent entity {:?}",
                        entity,
                        input.get().entity()
                    ));
                }
            } else {
                return Err(format!("Unary operation {:?} missing UnaryInput", entity));
            }
        }

        // Check binary operations have inputs
        if entity_ref.contains::<BinaryOpMarker>() {
            if let Some(inputs) = entity_ref.get::<BinaryInputs>() {
                if world.get_entity(inputs.left.entity()).is_err() {
                    return Err(format!(
                        "Binary operation {:?} references non-existent left entity {:?}",
                        entity,
                        inputs.left.entity()
                    ));
                }
                if world.get_entity(inputs.right.entity()).is_err() {
                    return Err(format!(
                        "Binary operation {:?} references non-existent right entity {:?}",
                        entity,
                        inputs.right.entity()
                    ));
                }
            } else {
                return Err(format!(
                    "Binary operation {:?} missing BinaryInputs",
                    entity
                ));
            }
        }

        // Check all nodes have Value component
        if entity_ref.get::<Value<F>>().is_none() {
            return Err(format!("Node {:?} missing Value component", entity));
        }
    }

    Ok(())
}

/// Counts the number of operations in a computation graph.
pub fn count_operations(world: &World, output: Var) -> (usize, usize, usize) {
    let entities = topological_order(world, output.entity())
        .expect("debug: cycle detected in computation graph");
    let mut inputs = 0;
    let mut constants = 0;
    let mut operations = 0;

    for &entity in &entities {
        let entity_ref = world.entity(entity);
        if entity_ref.contains::<IsInput>() {
            inputs += 1;
        } else if entity_ref.contains::<IsConstant>() {
            constants += 1;
        } else {
            operations += 1;
        }
    }

    (inputs, constants, operations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;

    #[test]
    fn test_to_dot_simple() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.square(x);

        let dot = to_dot::<f64>(ad.world(), y);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("input="));
        assert!(dot.contains("mul")); // square is x * x
    }

    #[test]
    fn test_validate_graph_valid() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);

        assert!(validate_graph::<f64>(ad.world(), f).is_ok());
    }

    #[test]
    fn test_count_operations() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let c = ad.constant(1.0);

        let xy = ad.mul(x, y);
        let f = ad.add(xy, c);

        let (inputs, constants, ops) = count_operations(ad.world(), f);
        assert_eq!(inputs, 2);
        assert_eq!(constants, 1);
        assert_eq!(ops, 2); // mul and add
    }
}
