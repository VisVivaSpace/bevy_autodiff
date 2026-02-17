//! Graph traversal utilities using EntityHandle.
//!
//! This module provides ergonomic traversal helpers for navigating the
//! computation graph. All functions take explicit `&World` parameters
//! following the EntityHandle pattern.

// Internal utility functions with test coverage — not all are used outside tests.
#![allow(dead_code)]

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use crate::components::{BinaryInputs, IsConstant, IsInput, UnaryInput, Value};
use crate::components::{BinaryOpMarker, UnaryOpMarker};

/// Gets the input entity for a unary operation.
pub fn get_unary_input(world: &World, entity: Entity) -> Option<Entity> {
    world.get::<UnaryInput>(entity).map(|ui| ui.get().entity())
}

/// Gets the input entities for a binary operation as (left, right).
pub fn get_binary_inputs(world: &World, entity: Entity) -> Option<(Entity, Entity)> {
    world
        .get::<BinaryInputs>(entity)
        .map(|bi| (bi.left.entity(), bi.right.entity()))
}

/// Gets all direct inputs to a node.
pub fn get_inputs(world: &World, entity: Entity) -> Vec<Entity> {
    let mut inputs = Vec::new();

    if let Some(ui) = world.get::<UnaryInput>(entity) {
        inputs.push(ui.get().entity());
    }

    if let Some(bi) = world.get::<BinaryInputs>(entity) {
        inputs.push(bi.left.entity());
        inputs.push(bi.right.entity());
    }

    inputs
}

/// Checks if an entity is a leaf node (input or constant).
pub fn is_leaf(world: &World, entity: Entity) -> bool {
    let entity_ref = world.entity(entity);
    entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>()
}

/// Gets the numerical value of an entity.
pub fn get_value<F: crate::diff_num::Float>(world: &World, entity: Entity) -> Option<F> {
    world.get::<Value<F>>(entity).map(|v| v.get())
}

/// Gets the operation name for a node, if it's an operation.
pub fn get_operation_name(world: &World, entity: Entity) -> Option<&'static str> {
    if let Some(op) = world.get::<UnaryOpMarker>(entity) {
        return Some(op.op().name());
    }
    if let Some(op) = world.get::<BinaryOpMarker>(entity) {
        return Some(op.op().name());
    }
    None
}

/// Visits all nodes in the computation graph in topological order.
///
/// The visitor function receives (entity, depth) where depth is the distance
/// from the output node.
pub fn visit_topological<F>(world: &World, output: Entity, mut visitor: F)
where
    F: FnMut(Entity, usize),
{
    let order = crate::graph::topological_order(world, output)
        .expect("cycle detected in computation graph");
    let depth_map: std::collections::HashMap<Entity, usize> = compute_depths(world, output);

    for entity in order {
        let depth = depth_map.get(&entity).copied().unwrap_or(0);
        visitor(entity, depth);
    }
}

/// Computes the depth of each node from the output.
fn compute_depths(world: &World, output: Entity) -> std::collections::HashMap<Entity, usize> {
    use std::collections::HashMap;

    let mut depths = HashMap::new();
    let mut queue = std::collections::VecDeque::new();

    depths.insert(output, 0);
    queue.push_back(output);

    while let Some(entity) = queue.pop_front() {
        let current_depth = depths[&entity];

        for input in get_inputs(world, entity) {
            let new_depth = current_depth + 1;
            let entry = depths.entry(input).or_insert(usize::MAX);
            if new_depth < *entry {
                *entry = new_depth;
                // Only re-enqueue if we found a shorter path
                queue.push_back(input);
            }
        }
    }

    depths
}

/// Collects all unique entities reachable from an output.
pub fn collect_all_entities(world: &World, output: Entity) -> Vec<Entity> {
    crate::graph::topological_order(world, output).expect("cycle detected in computation graph")
}

/// Finds the maximum depth of the computation graph.
pub fn max_depth(world: &World, output: Entity) -> usize {
    let depths = compute_depths(world, output);
    depths.values().copied().max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;

    #[test]
    fn test_get_unary_input() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.sin(x);

        let input = get_unary_input(ad.world(), y.entity());
        assert_eq!(input, Some(x.entity()));
    }

    #[test]
    fn test_get_binary_inputs() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.var(3.0).unwrap();
        let f = ad.mul(x, y);

        let inputs = get_binary_inputs(ad.world(), f.entity());
        assert_eq!(inputs, Some((x.entity(), y.entity())));
    }

    #[test]
    fn test_is_leaf() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let c = ad.constant(1.0);
        let y = ad.square(x);

        assert!(is_leaf(ad.world(), x.entity()));
        assert!(is_leaf(ad.world(), c.entity()));
        assert!(!is_leaf(ad.world(), y.entity()));
    }

    #[test]
    fn test_get_value() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.square(x);

        assert_eq!(get_value(ad.world(), x.entity()), Some(2.0));
        assert_eq!(get_value(ad.world(), y.entity()), Some(4.0));
    }

    #[test]
    fn test_get_operation_name() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let s = ad.sin(x);
        let y = ad.mul(x, x);

        assert_eq!(get_operation_name(ad.world(), x.entity()), None);
        assert_eq!(get_operation_name(ad.world(), s.entity()), Some("sin"));
        assert_eq!(get_operation_name(ad.world(), y.entity()), Some("mul"));
    }

    #[test]
    fn test_max_depth() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.square(x);
        let z = ad.sin(y);

        // x -> y -> z (depth 2 from z)
        assert_eq!(max_depth(ad.world(), z.entity()), 2);
    }

    #[test]
    fn test_free_functions_on_graph() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0).unwrap();
        let y = ad.square(x);

        assert!(is_leaf(ad.world(), x.entity()));
        assert!(!is_leaf(ad.world(), y.entity()));
        assert_eq!(get_value(ad.world(), x.entity()), Some(2.0));
        assert_eq!(get_value(ad.world(), y.entity()), Some(4.0));
    }
}
