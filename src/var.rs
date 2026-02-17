//! Variable handle type for the autodiff context.

use bevy_ecs::entity::Entity;
use bevy_entity_ptr::EntityHandle;
use std::hash::{Hash, Hasher};

/// A lightweight handle to a variable in the computation graph.
///
/// `Var` wraps an ECS entity, providing a type-safe way to reference
/// variables without exposing the underlying entity system.
///
/// Vars are Copy and cheap to pass around. They are only valid within
/// the `AutoDiff` context that created them.
#[derive(Debug, Clone, Copy)]
pub struct Var {
    /// The underlying entity in the computation graph.
    pub(crate) entity: Entity,
}

impl Var {
    /// Creates a new Var from an entity.
    /// This is internal - users should create Vars through AutoDiff methods.
    #[inline]
    pub(crate) const fn new(entity: Entity) -> Self {
        Self { entity }
    }

    /// Returns the underlying entity.
    /// Useful for advanced operations on the ECS world directly.
    #[inline]
    pub const fn entity(&self) -> Entity {
        self.entity
    }

    /// Converts to an EntityHandle for storage in components.
    #[inline]
    pub fn handle(&self) -> EntityHandle {
        EntityHandle::new(self.entity)
    }

    /// Creates a Var from an EntityHandle.
    #[inline]
    pub fn from_handle(handle: EntityHandle) -> Self {
        Self {
            entity: handle.entity(),
        }
    }
}

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Var({})", self.entity.index())
    }
}

impl PartialEq for Var {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.entity == other.entity
    }
}

impl Eq for Var {}

impl PartialOrd for Var {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Var {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.entity.cmp(&other.entity)
    }
}

impl Hash for Var {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.entity.hash(state);
    }
}

impl From<Var> for Entity {
    #[inline]
    fn from(var: Var) -> Self {
        var.entity
    }
}

impl From<EntityHandle> for Var {
    #[inline]
    fn from(handle: EntityHandle) -> Self {
        Self::from_handle(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_var_from_entity() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let var = Var::new(entity);
        assert_eq!(var.entity(), entity);
    }

    #[test]
    fn test_var_equality() {
        let mut world = World::new();
        let e1 = world.spawn_empty().id();
        let e2 = world.spawn_empty().id();

        let v1a = Var::new(e1);
        let v1b = Var::new(e1);
        let v2 = Var::new(e2);

        assert_eq!(v1a, v1b);
        assert_ne!(v1a, v2);
    }

    #[test]
    fn test_var_hash() {
        let mut world = World::new();
        let e1 = world.spawn_empty().id();
        let e2 = world.spawn_empty().id();

        let v1a = Var::new(e1);
        let v1b = Var::new(e1);
        let v2 = Var::new(e2);

        let mut set = HashSet::new();
        set.insert(v1a);

        assert!(set.contains(&v1b)); // Same entity
        assert!(!set.contains(&v2)); // Different entity
    }

    #[test]
    fn test_var_in_hashmap() {
        let mut world = World::new();
        let e1 = world.spawn_empty().id();
        let e2 = world.spawn_empty().id();

        let v1 = Var::new(e1);
        let v2 = Var::new(e2);

        let mut map = HashMap::new();
        map.insert(v1, "first");
        map.insert(v2, "second");

        assert_eq!(map.get(&v1), Some(&"first"));
        assert_eq!(map.get(&v2), Some(&"second"));
    }

    #[test]
    fn test_var_round_trip_entity() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();

        let var = Var::new(entity);
        let back: Entity = var.into();

        assert_eq!(entity, back);
    }

    #[test]
    fn test_var_round_trip_handle() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let handle = EntityHandle::new(entity);

        let var = Var::from_handle(handle);
        let back = var.handle();

        assert_eq!(handle, back);
    }

    #[test]
    fn test_var_is_copy() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let var = Var::new(entity);

        // Copy should work without moving
        let copy = var;
        let another_copy = var;

        assert_eq!(var, copy);
        assert_eq!(var, another_copy);
    }

    #[test]
    fn test_var_debug() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let var = Var::new(entity);

        let debug_str = format!("{:?}", var);
        assert!(debug_str.contains("Var"));
    }
}
