//! Topological sorting of the computation graph.

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use bevy_entity_ptr::EntityHandle;
use std::collections::{HashSet, VecDeque};

use crate::components::{BinaryInputs, IsConstant, IsInput, UnaryInput};

/// Computes a topological ordering of all entities that affect the given output.
///
/// Returns entities in dependency order: inputs first, then operations that
/// depend only on already-visited entities, and so on. The output entity is last.
pub fn topological_order(world: &World, output: Entity) -> Vec<Entity> {
    topological_order_multi(world, &[output])
}

/// Gets the direct dependencies (input entities) of an entity.
fn get_dependencies(world: &World, entity: Entity) -> Vec<Entity> {
    let entity_ref = world.entity(entity);

    // Input variables and constants have no dependencies
    if entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>() {
        return Vec::new();
    }

    let mut deps = Vec::new();

    // Check for unary input
    if let Some(unary_input) = entity_ref.get::<UnaryInput>() {
        deps.push(unary_input.get().entity());
    }

    // Check for binary inputs
    if let Some(binary_inputs) = entity_ref.get::<BinaryInputs>() {
        deps.push(binary_inputs.left.entity());
        deps.push(binary_inputs.right.entity());
    }

    deps
}

/// Computes a topological ordering of all entities that affect any of the given outputs.
///
/// Like `topological_order` but accepts multiple root outputs. Useful when
/// compiling a graph that includes both a function and its derivative outputs.
pub fn topological_order_multi(world: &World, outputs: &[Entity]) -> Vec<Entity> {
    let mut result = Vec::new();
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();

    // Start from all outputs and work backwards to find all dependencies
    for &output in outputs {
        queue.push_back(output);
    }

    while let Some(entity) = queue.pop_front() {
        if reachable.contains(&entity) {
            continue;
        }
        reachable.insert(entity);

        let deps = get_dependencies(world, entity);
        for dep in deps {
            if !reachable.contains(&dep) {
                queue.push_back(dep);
            }
        }
    }

    // Proper topological sort via DFS
    let nodes: Vec<Entity> = reachable.into_iter().collect();
    let node_set: HashSet<Entity> = nodes.iter().copied().collect();
    let mut visited = HashSet::new();
    let mut temp_mark = HashSet::new();

    fn visit(
        entity: Entity,
        world: &World,
        visited: &mut HashSet<Entity>,
        temp_mark: &mut HashSet<Entity>,
        result: &mut Vec<Entity>,
        nodes: &HashSet<Entity>,
    ) {
        if visited.contains(&entity) {
            return;
        }
        if temp_mark.contains(&entity) {
            panic!("Cycle detected in computation graph");
        }

        temp_mark.insert(entity);

        for dep in get_dependencies(world, entity) {
            if nodes.contains(&dep) {
                visit(dep, world, visited, temp_mark, result, nodes);
            }
        }

        temp_mark.remove(&entity);
        visited.insert(entity);
        result.push(entity);
    }

    for entity in nodes {
        visit(
            entity,
            world,
            &mut visited,
            &mut temp_mark,
            &mut result,
            &node_set,
        );
    }

    result
}

/// Checks if an entity is a leaf node (input or constant).
pub fn is_leaf(world: &World, entity: Entity) -> bool {
    let entity_ref = world.entity(entity);
    entity_ref.contains::<IsInput>() || entity_ref.contains::<IsConstant>()
}

/// Gets all input variable entities in the graph (not constants).
pub fn get_inputs(world: &World, output: Entity) -> Vec<Entity> {
    let order = topological_order(world, output);
    order
        .into_iter()
        .filter(|&e| world.entity(e).contains::<IsInput>())
        .collect()
}

/// Finds the entity by handle, returning None if the entity doesn't exist.
pub fn resolve_handle(world: &World, handle: EntityHandle) -> Option<Entity> {
    let entity = handle.entity();
    if world.get_entity(entity).is_ok() {
        Some(entity)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Value, Variable};
    use crate::context::{BinaryOpMarker, UnaryOpMarker};
    use crate::BinaryOp;
    use crate::UnaryOp;

    fn create_input(world: &mut World, value: f64) -> Entity {
        world.spawn((Variable, IsInput, Value::new(value))).id()
    }

    fn create_constant(world: &mut World, value: f64) -> Entity {
        world.spawn((Variable, IsConstant, Value::new(value))).id()
    }

    fn create_binary(world: &mut World, op: BinaryOp, left: Entity, right: Entity) -> Entity {
        world
            .spawn((
                Variable,
                BinaryOpMarker(op),
                BinaryInputs::new(EntityHandle::new(left), EntityHandle::new(right)),
            ))
            .id()
    }

    fn create_unary(world: &mut World, op: UnaryOp, input: Entity) -> Entity {
        world
            .spawn((
                Variable,
                UnaryOpMarker(op),
                UnaryInput::new(EntityHandle::new(input)),
            ))
            .id()
    }

    #[test]
    fn test_topological_single_input() {
        let mut world = World::new();
        let x = create_input(&mut world, 1.0);

        let order = topological_order(&world, x);
        assert_eq!(order, vec![x]);
    }

    #[test]
    fn test_topological_binary_op() {
        let mut world = World::new();
        let x = create_input(&mut world, 1.0);
        let y = create_input(&mut world, 2.0);
        let sum = create_binary(&mut world, BinaryOp::Add, x, y);

        let order = topological_order(&world, sum);

        // Inputs should come before the sum
        let x_pos = order.iter().position(|&e| e == x).unwrap();
        let y_pos = order.iter().position(|&e| e == y).unwrap();
        let sum_pos = order.iter().position(|&e| e == sum).unwrap();

        assert!(x_pos < sum_pos);
        assert!(y_pos < sum_pos);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_topological_unary_op() {
        let mut world = World::new();
        let x = create_input(&mut world, 1.0);
        let neg_x = create_unary(&mut world, UnaryOp::Neg, x);

        let order = topological_order(&world, neg_x);

        let x_pos = order.iter().position(|&e| e == x).unwrap();
        let neg_pos = order.iter().position(|&e| e == neg_x).unwrap();

        assert!(x_pos < neg_pos);
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_topological_chain() {
        let mut world = World::new();
        // x -> neg(x) -> neg(neg(x))
        let x = create_input(&mut world, 1.0);
        let neg1 = create_unary(&mut world, UnaryOp::Neg, x);
        let neg2 = create_unary(&mut world, UnaryOp::Neg, neg1);

        let order = topological_order(&world, neg2);

        let x_pos = order.iter().position(|&e| e == x).unwrap();
        let neg1_pos = order.iter().position(|&e| e == neg1).unwrap();
        let neg2_pos = order.iter().position(|&e| e == neg2).unwrap();

        assert!(x_pos < neg1_pos);
        assert!(neg1_pos < neg2_pos);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_topological_dag() {
        let mut world = World::new();
        // x used twice: x + x
        let x = create_input(&mut world, 1.0);
        let sum = create_binary(&mut world, BinaryOp::Add, x, x);

        let order = topological_order(&world, sum);

        // x should appear only once
        assert_eq!(order.iter().filter(|&&e| e == x).count(), 1);

        let x_pos = order.iter().position(|&e| e == x).unwrap();
        let sum_pos = order.iter().position(|&e| e == sum).unwrap();
        assert!(x_pos < sum_pos);
        assert_eq!(order.len(), 2);
    }

    #[test]
    fn test_topological_with_constant() {
        let mut world = World::new();
        let x = create_input(&mut world, 1.0);
        let c = create_constant(&mut world, 5.0);
        let sum = create_binary(&mut world, BinaryOp::Add, x, c);

        let order = topological_order(&world, sum);

        let x_pos = order.iter().position(|&e| e == x).unwrap();
        let c_pos = order.iter().position(|&e| e == c).unwrap();
        let sum_pos = order.iter().position(|&e| e == sum).unwrap();

        assert!(x_pos < sum_pos);
        assert!(c_pos < sum_pos);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_is_leaf() {
        let mut world = World::new();
        let x = create_input(&mut world, 1.0);
        let c = create_constant(&mut world, 2.0);
        let sum = create_binary(&mut world, BinaryOp::Add, x, c);

        assert!(is_leaf(&world, x));
        assert!(is_leaf(&world, c));
        assert!(!is_leaf(&world, sum));
    }

    #[test]
    fn test_get_inputs() {
        let mut world = World::new();
        let x = create_input(&mut world, 1.0);
        let y = create_input(&mut world, 2.0);
        let c = create_constant(&mut world, 3.0);
        let sum1 = create_binary(&mut world, BinaryOp::Add, x, y);
        let sum2 = create_binary(&mut world, BinaryOp::Add, sum1, c);

        let inputs = get_inputs(&world, sum2);

        // Should contain x and y but not c
        assert_eq!(inputs.len(), 2);
        assert!(inputs.contains(&x));
        assert!(inputs.contains(&y));
        assert!(!inputs.contains(&c));
    }

    #[test]
    fn test_complex_graph() {
        let mut world = World::new();
        // Build: f = (x + y) * (x - y) = x² - y²
        let x = create_input(&mut world, 3.0);
        let y = create_input(&mut world, 2.0);
        let sum = create_binary(&mut world, BinaryOp::Add, x, y);
        let diff = create_binary(&mut world, BinaryOp::Sub, x, y);
        let prod = create_binary(&mut world, BinaryOp::Mul, sum, diff);

        let order = topological_order(&world, prod);

        // Verify ordering constraints
        let x_pos = order.iter().position(|&e| e == x).unwrap();
        let y_pos = order.iter().position(|&e| e == y).unwrap();
        let sum_pos = order.iter().position(|&e| e == sum).unwrap();
        let diff_pos = order.iter().position(|&e| e == diff).unwrap();
        let prod_pos = order.iter().position(|&e| e == prod).unwrap();

        // x and y before everything else
        assert!(x_pos < sum_pos);
        assert!(x_pos < diff_pos);
        assert!(y_pos < sum_pos);
        assert!(y_pos < diff_pos);

        // sum and diff before prod
        assert!(sum_pos < prod_pos);
        assert!(diff_pos < prod_pos);

        assert_eq!(order.len(), 5);
    }
}
