//! Cache management for Taylor coefficient data.
//!
//! When input values change, cached Taylor coefficients become invalid and
//! need to be recomputed. This module provides utilities for managing cache
//! validity and invalidation.

use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;

use crate::components::{Direction, TaylorData};

/// Checks if the Taylor cache for an entity is valid for a given direction and order.
///
/// Returns true if the entity has cached Taylor coefficients for the specified
/// direction with at least the required order.
pub fn is_cache_valid(world: &World, entity: Entity, direction: &Direction, order: usize) -> bool {
    world
        .get::<TaylorData>(entity)
        .and_then(|td| td.get_directional(direction))
        .map(|coeffs| coeffs.len() > order)
        .unwrap_or(false)
}

/// Invalidates (clears) all cached Taylor data for an entity.
///
/// This should be called when the entity's value changes (e.g., input update).
pub fn invalidate_cache(world: &mut World, entity: Entity) {
    if let Some(mut td) = world.get_mut::<TaylorData>(entity) {
        td.clear_directional();
        td.clear_partials();
    }
}

/// Invalidates caches for all entities that depend on a given input.
///
/// When an input value changes, all variables that depend on it (directly or
/// transitively) need their caches invalidated.
pub fn invalidate_dependents(world: &mut World, input_entity: Entity) {
    use crate::components::Dependencies;

    // Get the input's dependency bit
    let input_deps = world
        .get::<Dependencies>(input_entity)
        .map(|d| d.mask)
        .unwrap_or(0);

    if input_deps == 0 {
        return;
    }

    // Find all entities that depend on this input
    let entities_to_invalidate: Vec<Entity> = world
        .query::<(Entity, &Dependencies)>()
        .iter(world)
        .filter(|(_, deps)| (deps.mask & input_deps) != 0)
        .map(|(e, _)| e)
        .collect();

    // Invalidate each dependent entity
    for entity in entities_to_invalidate {
        invalidate_cache(world, entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;

    #[test]
    fn test_is_cache_valid_empty() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);

        let direction = Direction::basis(1, 0);
        // Initially, no Taylor data beyond constant
        assert!(!is_cache_valid(ad.world(), x.entity(), &direction, 1));
    }

    #[test]
    fn test_is_cache_valid_after_propagation() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x);

        // Compute derivative to populate cache
        ad.derivative(y, x, 2);

        let direction = Direction::basis(1, 0);
        // Now cache should be valid for order 2
        assert!(is_cache_valid(ad.world(), y.entity(), &direction, 2));
        // But not for higher orders
        assert!(!is_cache_valid(ad.world(), y.entity(), &direction, 5));
    }

    #[test]
    fn test_invalidate_cache() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x);

        // Populate cache
        ad.derivative(y, x, 2);

        let direction = Direction::basis(1, 0);
        assert!(is_cache_valid(ad.world(), y.entity(), &direction, 2));

        // Invalidate
        invalidate_cache(ad.world_mut(), y.entity());

        // Cache should now be invalid
        assert!(!is_cache_valid(ad.world(), y.entity(), &direction, 1));
    }

    #[test]
    fn test_invalidate_dependents() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.square(x);

        // Populate caches
        ad.derivative(y, x, 2);

        // Invalidate all dependents of x
        invalidate_dependents(ad.world_mut(), x.entity());

        // y's cache should be invalidated (it depends on x)
        let direction = Direction::basis(1, 0);
        assert!(!is_cache_valid(ad.world(), y.entity(), &direction, 1));
    }
}
