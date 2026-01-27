//! Partial derivative extraction from Taylor coefficients.
//!
//! This module extracts partial derivatives using the polarization identity
//! and directional derivative relationships.
//!
//! ## Pure Partial Derivatives
//! For ∂ⁿf/∂xᵢⁿ (all derivatives with respect to a single variable),
//! we simply extract the n-th Taylor coefficient in direction eᵢ.
//!
//! ## Mixed Partial Derivatives
//! For ∂²f/∂xᵢ∂xⱼ with i ≠ j, we use the polarization identity:
//! ```text
//! ∂²f/∂xᵢ∂xⱼ = D²f(eᵢ + eⱼ) - D²f(eᵢ) - D²f(eⱼ)
//! ```
//!
//! Higher-order mixed partials use generalized polarization formulas.

mod directional;
mod interpolate;

pub use directional::{extract_directional_derivative, get_pure_partial};
pub use interpolate::{get_mixed_partial, get_mixed_partial_2};

use crate::components::{Direction, MultiIndex};
use crate::taylor::propagate::propagate_taylor;
use crate::var::Var;
use bevy_ecs::world::World;

/// Computes a partial derivative of a variable with respect to input variables.
///
/// # Arguments
/// - `world`: The ECS world containing the computation graph
/// - `output`: The variable to differentiate
/// - `index`: The multi-index specifying which partial to compute
/// - `num_inputs`: Total number of input variables
///
/// # Returns
/// The partial derivative ∂^|α|f / ∂x^α where α = index
pub fn compute_partial(
    world: &mut World,
    output: Var,
    index: &MultiIndex,
    num_inputs: usize,
) -> f64 {
    let order = index.order();

    // Check if this is a pure partial (only one variable involved)
    let nonzero_vars: Vec<usize> = index
        .0
        .iter()
        .enumerate()
        .filter(|(_, &exp)| exp > 0)
        .map(|(i, _)| i)
        .collect();

    match nonzero_vars.len() {
        0 => {
            // Zero multi-index: return function value
            // Propagate with zero direction to get value
            let direction = Direction::zero(num_inputs);
            let coeffs = propagate_taylor(world, output.entity(), &direction, 0);
            coeffs.first().copied().unwrap_or(0.0)
        }
        1 => {
            // Pure partial: ∂^n f / ∂xᵢ^n
            let i = nonzero_vars[0];
            get_pure_partial(world, output, i, order, num_inputs)
        }
        2 if order == 2 => {
            // Mixed second partial: ∂²f/∂xᵢ∂xⱼ
            let i = nonzero_vars[0];
            let j = nonzero_vars[1];
            get_mixed_partial_2(world, output, i, j, num_inputs)
        }
        _ => {
            // General mixed partial - use generalized polarization
            get_mixed_partial(world, output, index, num_inputs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AutoDiff;
    use approx::assert_relative_eq;

    #[test]
    fn test_compute_partial_value() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = x * y = 6

        let index = MultiIndex::zero(2);
        let value = compute_partial(ad.world_mut(), f, &index, 2);

        assert_eq!(value, 6.0);
    }

    #[test]
    fn test_compute_partial_first() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = x * y

        // ∂f/∂x = y = 3
        let index_x = MultiIndex::first(2, 0);
        assert_eq!(compute_partial(ad.world_mut(), f, &index_x, 2), 3.0);

        // ∂f/∂y = x = 2
        let index_y = MultiIndex::first(2, 1);
        assert_eq!(compute_partial(ad.world_mut(), f, &index_y, 2), 2.0);
    }

    #[test]
    fn test_compute_partial_pure_second() {
        let mut ad = AutoDiff::new();
        let x = ad.var(3.0);
        let f = ad.square(x); // f = x²

        // ∂²f/∂x² = 2
        let index = MultiIndex::pure(1, 0, 2);
        assert_eq!(compute_partial(ad.world_mut(), f, &index, 1), 2.0);
    }

    #[test]
    fn test_compute_partial_mixed_second() {
        let mut ad = AutoDiff::new();
        let x = ad.var(2.0);
        let y = ad.var(3.0);
        let f = ad.mul(x, y); // f = x * y

        // ∂²f/∂x∂y = 1
        let index = MultiIndex::new(vec![1, 1]);
        assert_relative_eq!(compute_partial(ad.world_mut(), f, &index, 2), 1.0, epsilon = 1e-10);
    }
}
