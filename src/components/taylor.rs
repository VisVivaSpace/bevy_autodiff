//! Taylor coefficient storage and multi-index types.

use bevy_ecs::component::Component;
use std::collections::HashMap;
use std::fmt;

use crate::util::factorial;

/// A direction vector for directional derivatives.
///
/// For n input variables, a direction d = (d₁, d₂, ..., dₙ) specifies
/// the derivative D_d f = d₁·∂f/∂x₁ + d₂·∂f/∂x₂ + ... + dₙ·∂f/∂xₙ
///
/// The k-th directional derivative along d is:
/// D_d^k f = (d₁·∂/∂x₁ + ... + dₙ·∂/∂xₙ)^k f
///
/// Stored as integers for exact comparison and hashing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Direction(pub Vec<i32>);

impl Direction {
    /// Creates a new direction from a vector of components.
    #[inline]
    pub fn new(components: Vec<i32>) -> Self {
        Self(components)
    }

    /// Creates a unit direction along the i-th coordinate axis.
    /// e_i = (0, ..., 0, 1, 0, ..., 0) with 1 at position i.
    #[inline]
    pub fn basis(dim: usize, index: usize) -> Self {
        let mut components = vec![0; dim];
        if index < dim {
            components[index] = 1;
        }
        Self(components)
    }

    /// Creates the sum of two basis vectors: e_i + e_j.
    /// Used for mixed partial extraction via polarization.
    #[inline]
    pub fn sum_of_basis(dim: usize, i: usize, j: usize) -> Self {
        let mut components = vec![0; dim];
        if i < dim {
            components[i] += 1;
        }
        if j < dim {
            components[j] += 1;
        }
        Self(components)
    }

    /// Creates the zero direction vector.
    #[inline]
    pub fn zero(dim: usize) -> Self {
        Self(vec![0; dim])
    }

    /// Returns the dimension (number of input variables).
    #[inline]
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Returns true if this is the zero direction.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&c| c == 0)
    }

    /// Returns the component along the i-th axis.
    #[inline]
    pub fn get(&self, i: usize) -> i32 {
        self.0.get(i).copied().unwrap_or(0)
    }
}

impl From<Vec<i32>> for Direction {
    fn from(v: Vec<i32>) -> Self {
        Self(v)
    }
}

impl From<&[i32]> for Direction {
    fn from(v: &[i32]) -> Self {
        Self(v.to_vec())
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, &c) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", c)?;
        }
        write!(f, ")")
    }
}

/// A multi-index for partial derivatives.
///
/// α = (α₁, α₂, ..., αₙ) represents the partial derivative:
/// ∂^|α| f / ∂x₁^α₁ ∂x₂^α₂ ... ∂xₙ^αₙ
///
/// where |α| = α₁ + α₂ + ... + αₙ is the total derivative order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct MultiIndex(pub Vec<usize>);

impl MultiIndex {
    /// Creates a new multi-index from a vector of exponents.
    #[inline]
    pub fn new(exponents: Vec<usize>) -> Self {
        Self(exponents)
    }

    /// Creates a multi-index for a pure partial: ∂^k f / ∂x_i^k
    #[inline]
    pub fn pure(dim: usize, var_index: usize, order: usize) -> Self {
        let mut exponents = vec![0; dim];
        if var_index < dim {
            exponents[var_index] = order;
        }
        Self(exponents)
    }

    /// Creates a multi-index for first partial: ∂f/∂x_i
    #[inline]
    pub fn first(dim: usize, var_index: usize) -> Self {
        Self::pure(dim, var_index, 1)
    }

    /// Creates the zero multi-index (for function value).
    #[inline]
    pub fn zero(dim: usize) -> Self {
        Self(vec![0; dim])
    }

    /// Returns the total order |α| = Σ αᵢ.
    #[inline]
    pub fn order(&self) -> usize {
        self.0.iter().sum()
    }

    /// Returns α! = α₁! · α₂! · ... · αₙ!
    ///
    /// This appears in the multinomial theorem and mixed partial formulas.
    #[inline]
    pub fn factorial_product(&self) -> f64 {
        self.0.iter().map(|&a| factorial(a)).product()
    }

    /// Returns the dimension (number of variables).
    #[inline]
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// Returns the exponent for variable i.
    #[inline]
    pub fn get(&self, i: usize) -> usize {
        self.0.get(i).copied().unwrap_or(0)
    }

    /// Returns true if this is the zero multi-index.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&a| a == 0)
    }
}

impl From<Vec<usize>> for MultiIndex {
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for MultiIndex {
    fn from(v: &[usize]) -> Self {
        Self(v.to_vec())
    }
}

impl fmt::Display for MultiIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, &e) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", e)?;
        }
        write!(f, ")")
    }
}

/// Taylor coefficient storage for a variable.
///
/// Stores both:
/// - Directional Taylor coefficients: f_k = D_d^k f(a) / k!
/// - Extracted partial derivatives: ∂^α f / ∂x^α
///
/// Coefficients are stored **normalized** (divided by k! or α!):
/// - This keeps intermediate values small for numerical stability
/// - Recurrence formulas naturally produce normalized coefficients
/// - We multiply by k! only when extracting the actual derivative
#[derive(Component, Debug, Clone, Default)]
pub struct TaylorData {
    /// Directional Taylor coefficients indexed by direction.
    /// For each direction d, stores [f_0, f_1, ..., f_k] where f_k = D_d^k f(a) / k!
    pub directional: HashMap<Direction, Vec<f64>>,

    /// Cached partial derivatives indexed by multi-index.
    /// Stores the actual derivative value (not normalized).
    pub partials: HashMap<MultiIndex, f64>,
}

impl TaylorData {
    /// Creates empty Taylor data.
    #[inline]
    pub fn new() -> Self {
        Self {
            directional: HashMap::new(),
            partials: HashMap::new(),
        }
    }

    /// Creates Taylor data with a single constant value.
    /// Used for input variables and constants.
    pub fn constant(value: f64) -> Self {
        let mut data = Self::new();
        // Store value at zero direction with order 0
        data.directional.insert(Direction::zero(0), vec![value]);
        data
    }

    /// Gets the Taylor coefficients for a given direction, if computed.
    #[inline]
    pub fn get_directional(&self, direction: &Direction) -> Option<&Vec<f64>> {
        self.directional.get(direction)
    }

    /// Gets a specific Taylor coefficient for a direction and order.
    #[inline]
    pub fn get_coefficient(&self, direction: &Direction, order: usize) -> Option<f64> {
        self.directional
            .get(direction)
            .and_then(|coeffs| coeffs.get(order).copied())
    }

    /// Gets the current maximum order computed for a direction.
    #[inline]
    pub fn max_order(&self, direction: &Direction) -> Option<usize> {
        self.directional
            .get(direction)
            .map(|coeffs| coeffs.len().saturating_sub(1))
    }

    /// Gets a cached partial derivative, if computed.
    #[inline]
    pub fn get_partial(&self, index: &MultiIndex) -> Option<f64> {
        self.partials.get(index).copied()
    }

    /// Sets the Taylor coefficients for a direction.
    /// This is the only way to write directional data (immutable pattern).
    #[inline]
    pub fn set_directional(&mut self, direction: Direction, coefficients: Vec<f64>) {
        self.directional.insert(direction, coefficients);
    }

    /// Sets a cached partial derivative.
    #[inline]
    pub fn set_partial(&mut self, index: MultiIndex, value: f64) {
        self.partials.insert(index, value);
    }

    /// Returns the function value f(a) if computed.
    /// This is the 0th Taylor coefficient (same for all directions).
    pub fn value(&self) -> Option<f64> {
        // Try to get from any direction, coefficient 0
        self.directional
            .values()
            .next()
            .and_then(|coeffs| coeffs.first().copied())
    }

    /// Clears all cached directional Taylor coefficients.
    #[inline]
    pub fn clear_directional(&mut self) {
        self.directional.clear();
    }

    /// Clears all cached partial derivatives.
    #[inline]
    pub fn clear_partials(&mut self) {
        self.partials.clear();
    }

    /// Clears all cached data (both directional and partials).
    #[inline]
    pub fn clear(&mut self) {
        self.directional.clear();
        self.partials.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Direction tests
    #[test]
    fn test_direction_basis() {
        let e0 = Direction::basis(3, 0);
        assert_eq!(e0.0, vec![1, 0, 0]);

        let e1 = Direction::basis(3, 1);
        assert_eq!(e1.0, vec![0, 1, 0]);

        let e2 = Direction::basis(3, 2);
        assert_eq!(e2.0, vec![0, 0, 1]);
    }

    #[test]
    fn test_direction_sum_of_basis() {
        let e01 = Direction::sum_of_basis(3, 0, 1);
        assert_eq!(e01.0, vec![1, 1, 0]);

        // Same index sums to 2
        let e00 = Direction::sum_of_basis(3, 0, 0);
        assert_eq!(e00.0, vec![2, 0, 0]);
    }

    #[test]
    fn test_direction_zero() {
        let zero = Direction::zero(4);
        assert_eq!(zero.0, vec![0, 0, 0, 0]);
        assert!(zero.is_zero());
        assert_eq!(zero.dim(), 4);
    }

    #[test]
    fn test_direction_get() {
        let d = Direction::new(vec![1, 2, 3]);
        assert_eq!(d.get(0), 1);
        assert_eq!(d.get(1), 2);
        assert_eq!(d.get(2), 3);
        assert_eq!(d.get(10), 0); // Out of bounds returns 0
    }

    #[test]
    fn test_direction_equality_and_hash() {
        use std::collections::HashSet;

        let d1 = Direction::basis(3, 0);
        let d2 = Direction::basis(3, 0);
        let d3 = Direction::basis(3, 1);

        assert_eq!(d1, d2);
        assert_ne!(d1, d3);

        let mut set = HashSet::new();
        set.insert(d1.clone());
        assert!(set.contains(&d2));
        assert!(!set.contains(&d3));
    }

    // MultiIndex tests
    #[test]
    fn test_multi_index_order() {
        let alpha = MultiIndex::new(vec![2, 1, 0, 3]);
        assert_eq!(alpha.order(), 6); // 2+1+0+3
    }

    #[test]
    fn test_multi_index_factorial_product() {
        // α = (2, 3) → α! = 2! * 3! = 2 * 6 = 12
        let alpha = MultiIndex::new(vec![2, 3]);
        assert_eq!(alpha.factorial_product(), 12.0);

        // α = (0, 0) → α! = 0! * 0! = 1
        let zero = MultiIndex::zero(2);
        assert_eq!(zero.factorial_product(), 1.0);
    }

    #[test]
    fn test_multi_index_pure() {
        let alpha = MultiIndex::pure(3, 1, 4);
        assert_eq!(alpha.0, vec![0, 4, 0]);
        assert_eq!(alpha.order(), 4);
    }

    #[test]
    fn test_multi_index_first() {
        let alpha = MultiIndex::first(3, 2);
        assert_eq!(alpha.0, vec![0, 0, 1]);
        assert_eq!(alpha.order(), 1);
    }

    #[test]
    fn test_multi_index_get() {
        let alpha = MultiIndex::new(vec![1, 2, 3]);
        assert_eq!(alpha.get(0), 1);
        assert_eq!(alpha.get(1), 2);
        assert_eq!(alpha.get(2), 3);
        assert_eq!(alpha.get(10), 0); // Out of bounds returns 0
    }

    // TaylorData tests
    #[test]
    fn test_taylor_data_empty() {
        let data = TaylorData::new();
        assert!(data.directional.is_empty());
        assert!(data.partials.is_empty());
        assert!(data.value().is_none());
    }

    #[test]
    fn test_taylor_data_constant() {
        let data = TaylorData::constant(5.0);
        assert_eq!(data.value(), Some(5.0));
    }

    #[test]
    fn test_taylor_data_directional() {
        let mut data = TaylorData::new();
        let dir = Direction::basis(2, 0);
        let coeffs = vec![1.0, 2.0, 3.0];

        data.set_directional(dir.clone(), coeffs.clone());

        assert_eq!(data.get_directional(&dir), Some(&coeffs));
        assert_eq!(data.get_coefficient(&dir, 0), Some(1.0));
        assert_eq!(data.get_coefficient(&dir, 1), Some(2.0));
        assert_eq!(data.get_coefficient(&dir, 2), Some(3.0));
        assert_eq!(data.get_coefficient(&dir, 3), None);
        assert_eq!(data.max_order(&dir), Some(2));
    }

    #[test]
    fn test_taylor_data_partials() {
        let mut data = TaylorData::new();
        let idx = MultiIndex::pure(2, 0, 2);

        data.set_partial(idx.clone(), 42.0);

        assert_eq!(data.get_partial(&idx), Some(42.0));
        assert_eq!(data.get_partial(&MultiIndex::zero(2)), None);
    }

    #[test]
    fn test_taylor_data_as_component() {
        use bevy_ecs::world::World;

        let mut world = World::new();
        let mut data = TaylorData::new();
        data.set_directional(Direction::basis(1, 0), vec![1.0, 2.0]);

        let entity = world.spawn(data).id();

        let retrieved = world.entity(entity).get::<TaylorData>().unwrap();
        assert_eq!(
            retrieved.get_coefficient(&Direction::basis(1, 0), 1),
            Some(2.0)
        );
    }
}
