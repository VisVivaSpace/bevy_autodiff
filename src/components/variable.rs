//! Variable marker components and value storage.

use bevy_ecs::component::Component;

/// Marker component indicating an entity is a variable in the computation graph.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Variable;

/// Marker component indicating a variable is an input (leaf node).
/// Input variables have user-specified values and are the sources for derivative computation.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IsInput;

/// Marker component indicating a variable is a constant.
/// Constants have fixed values and zero derivatives with respect to all inputs.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IsConstant;

/// Stores the numerical value of a variable.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Value(pub(crate) f64);

impl Value {
    /// Creates a new Value component.
    #[inline]
    pub const fn new(v: f64) -> Self {
        Self(v)
    }

    /// Returns the stored value.
    #[inline]
    pub const fn get(&self) -> f64 {
        self.0
    }

    /// Sets the stored value.
    #[inline]
    pub fn set(&mut self, v: f64) {
        self.0 = v;
    }
}

/// Bitmask tracking which input variables affect this variable.
///
/// Each bit corresponds to an input variable index. If bit i is set,
/// input i affects this variable's value (transitively through the graph).
///
/// This is used for efficient pruning during derivative computation:
/// if an output doesn't depend on an input, its derivative is trivially zero.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dependencies {
    /// Bitmask of input variable indices that affect this variable.
    pub(crate) mask: u64,
}

impl Dependencies {
    /// Creates an empty dependency set.
    #[inline]
    pub const fn none() -> Self {
        Self { mask: 0 }
    }

    /// Creates a dependency set with a single input index.
    #[inline]
    pub const fn single(input_index: usize) -> Self {
        Self {
            mask: 1 << input_index,
        }
    }

    /// Returns true if this variable depends on the given input index.
    #[inline]
    pub const fn depends_on(&self, input_index: usize) -> bool {
        (self.mask & (1 << input_index)) != 0
    }

    /// Returns true if this variable has no dependencies (constant or isolated).
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.mask == 0
    }

    /// Returns the union of two dependency sets.
    #[inline]
    pub const fn union(&self, other: &Self) -> Self {
        Self {
            mask: self.mask | other.mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::world::World;

    #[test]
    fn test_variable_marker() {
        let mut world = World::new();
        let entity = world.spawn(Variable).id();
        assert!(world.entity(entity).contains::<Variable>());
    }

    #[test]
    fn test_is_input_marker() {
        let mut world = World::new();
        let entity = world.spawn((Variable, IsInput)).id();
        assert!(world.entity(entity).contains::<Variable>());
        assert!(world.entity(entity).contains::<IsInput>());
    }

    #[test]
    fn test_is_constant_marker() {
        let mut world = World::new();
        let entity = world.spawn((Variable, IsConstant)).id();
        assert!(world.entity(entity).contains::<Variable>());
        assert!(world.entity(entity).contains::<IsConstant>());
    }

    #[test]
    fn test_value_component() {
        let mut world = World::new();
        let entity = world.spawn(Value::new(3.14)).id();
        let value = world.entity(entity).get::<Value>().unwrap();
        assert_eq!(value.get(), 3.14);
    }

    #[test]
    fn test_value_special_values() {
        assert!(Value::new(f64::INFINITY).get().is_infinite());
        assert!(Value::new(f64::NAN).get().is_nan());
        assert_eq!(Value::new(0.0).get(), 0.0);
        assert_eq!(Value::new(-0.0).get(), -0.0);
    }

    #[test]
    fn test_dependencies_none() {
        let deps = Dependencies::none();
        assert!(deps.is_empty());
        assert!(!deps.depends_on(0));
        assert!(!deps.depends_on(63));
    }

    #[test]
    fn test_dependencies_single() {
        let deps = Dependencies::single(5);
        assert!(!deps.is_empty());
        assert!(!deps.depends_on(4));
        assert!(deps.depends_on(5));
        assert!(!deps.depends_on(6));
    }

    #[test]
    fn test_dependencies_union() {
        let deps1 = Dependencies::single(2);
        let deps2 = Dependencies::single(7);
        let combined = deps1.union(&deps2);
        assert!(combined.depends_on(2));
        assert!(combined.depends_on(7));
        assert!(!combined.depends_on(3));
    }

    #[test]
    fn test_dependencies_all_bits() {
        // Test edge case at bit 63
        let deps = Dependencies::single(63);
        assert!(deps.depends_on(63));
        assert!(!deps.depends_on(62));
    }
}
