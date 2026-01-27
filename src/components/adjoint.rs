//! Adjoint (reverse mode) Taylor coefficient storage.
//!
//! In reverse mode AD, we backpropagate "adjoint polynomials" which represent
//! the sensitivity of the output with respect to intermediate values.

use bevy_ecs::component::Component;
use std::collections::HashMap;

use super::Direction;

/// Adjoint Taylor polynomial coefficients for reverse mode.
///
/// Stores the adjoint (sensitivity) polynomial for each direction.
/// The adjoint of a variable y with respect to output f is:
/// ȳ = ∂f/∂y (as a polynomial in the perturbation parameter t)
///
/// During backpropagation, these adjoint polynomials are accumulated
/// and transformed according to the chain rule.
#[derive(Component, Debug, Clone, Default)]
pub struct AdjointTaylor {
    /// Adjoint Taylor coefficients per direction.
    /// adjoint[direction][k] = k-th coefficient of adjoint polynomial
    pub adjoint: HashMap<Direction, Vec<f64>>,
}

impl AdjointTaylor {
    /// Creates an empty adjoint storage.
    pub fn new() -> Self {
        Self {
            adjoint: HashMap::new(),
        }
    }

    /// Creates adjoint storage initialized for the output variable.
    ///
    /// For the output variable f, the adjoint is df/df = 1 (constant polynomial).
    pub fn output(direction: Direction, order: usize) -> Self {
        let mut adjoint = HashMap::new();
        let mut coeffs = vec![0.0; order + 1];
        coeffs[0] = 1.0; // df/df = 1
        adjoint.insert(direction, coeffs);
        Self { adjoint }
    }

    /// Gets the adjoint coefficients for a direction, if computed.
    pub fn get(&self, direction: &Direction) -> Option<&Vec<f64>> {
        self.adjoint.get(direction)
    }

    /// Gets mutable adjoint coefficients for a direction.
    pub fn get_mut(&mut self, direction: &Direction) -> Option<&mut Vec<f64>> {
        self.adjoint.get_mut(direction)
    }

    /// Inserts or replaces adjoint coefficients for a direction.
    pub fn insert(&mut self, direction: Direction, coeffs: Vec<f64>) {
        self.adjoint.insert(direction, coeffs);
    }

    /// Gets or creates adjoint coefficients for a direction.
    pub fn get_or_insert(&mut self, direction: Direction, order: usize) -> &mut Vec<f64> {
        self.adjoint
            .entry(direction)
            .or_insert_with(|| vec![0.0; order + 1])
    }

    /// Accumulates adjoint coefficients for a direction.
    ///
    /// This is used during backpropagation when a variable has multiple consumers.
    pub fn accumulate(&mut self, direction: &Direction, coeffs: &[f64]) {
        if let Some(existing) = self.adjoint.get_mut(direction) {
            // Ensure vectors are same length, extending with zeros if needed
            let max_len = existing.len().max(coeffs.len());
            existing.resize(max_len, 0.0);
            for (i, &c) in coeffs.iter().enumerate() {
                existing[i] += c;
            }
        } else {
            self.adjoint.insert(direction.clone(), coeffs.to_vec());
        }
    }

    /// Clears all stored adjoint coefficients.
    pub fn clear(&mut self) {
        self.adjoint.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjoint_new() {
        let adj = AdjointTaylor::new();
        assert!(adj.adjoint.is_empty());
    }

    #[test]
    fn test_adjoint_output() {
        let dir = Direction::basis(2, 0);
        let adj = AdjointTaylor::output(dir.clone(), 3);

        let coeffs = adj.get(&dir).unwrap();
        assert_eq!(coeffs.len(), 4);
        assert_eq!(coeffs[0], 1.0);
        assert_eq!(coeffs[1], 0.0);
        assert_eq!(coeffs[2], 0.0);
        assert_eq!(coeffs[3], 0.0);
    }

    #[test]
    fn test_adjoint_insert_get() {
        let mut adj = AdjointTaylor::new();
        let dir = Direction::basis(2, 1);

        adj.insert(dir.clone(), vec![1.0, 2.0, 3.0]);

        let coeffs = adj.get(&dir).unwrap();
        assert_eq!(coeffs, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_adjoint_accumulate() {
        let mut adj = AdjointTaylor::new();
        let dir = Direction::basis(2, 0);

        adj.accumulate(&dir, &[1.0, 2.0]);
        assert_eq!(adj.get(&dir).unwrap(), &vec![1.0, 2.0]);

        adj.accumulate(&dir, &[3.0, 4.0, 5.0]);
        assert_eq!(adj.get(&dir).unwrap(), &vec![4.0, 6.0, 5.0]);
    }

    #[test]
    fn test_adjoint_get_or_insert() {
        let mut adj = AdjointTaylor::new();
        let dir = Direction::basis(1, 0);

        {
            let coeffs = adj.get_or_insert(dir.clone(), 2);
            assert_eq!(coeffs, &vec![0.0, 0.0, 0.0]);
            coeffs[0] = 1.0;
        }

        assert_eq!(adj.get(&dir).unwrap()[0], 1.0);
    }

    #[test]
    fn test_adjoint_clear() {
        let mut adj = AdjointTaylor::new();
        let dir = Direction::basis(1, 0);
        adj.insert(dir, vec![1.0, 2.0]);

        adj.clear();
        assert!(adj.adjoint.is_empty());
    }
}
